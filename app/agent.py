from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from livekit import agents
from livekit.agents import APIConnectOptions, JobProcess, RunContext
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.llm import function_tool
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import openai, silero
from dotenv import load_dotenv

from app.config import (
    GOOGLE_CALENDAR_ID,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    GOOGLE_TIMEZONE,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT_SECONDS,
    MIN_SLOT_MINUTES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_TTS_VOICE,
    STT_PROVIDER,
    TTS_PROVIDER,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_STT_API_KEY,
    LOCAL_STT_BASE_URL,
    LOCAL_STT_MODEL,
    LOCAL_TTS_API_KEY,
    LOCAL_TTS_BASE_URL,
    LOCAL_TTS_MODEL,
    LOCAL_TTS_VOICE,
)
from app.calendar_service import GoogleCalendarService
from app.date_utils import (
    events_to_summary,
    missing_fields,
    parse_calendar_date,
    parse_dt,
    parse_dt_or_error,
    preset_from_natural_query,
    validate_future_window,
    window_for_preset,
)

load_dotenv('.env')
logger = logging.getLogger(__name__)

EFFECTIVE_LLM_REQUEST_TIMEOUT_SECONDS = LLM_REQUEST_TIMEOUT_SECONDS
CALENDAR_TIMEOUT_SECONDS = 12.0


def prewarm(proc: JobProcess):
    proc.userdata['vad'] = silero.VAD.load()


def build_llm():
    import openai as openai_client
    import httpx
    if LLM_PROVIDER == 'local':
        return openai.LLM(
            model=LLM_MODEL,
            client=openai_client.AsyncOpenAI(
                api_key=LOCAL_LLM_API_KEY,
                base_url=LOCAL_LLM_BASE_URL,
                timeout=httpx.Timeout(connect=5.0, read=EFFECTIVE_LLM_REQUEST_TIMEOUT_SECONDS, write=EFFECTIVE_LLM_REQUEST_TIMEOUT_SECONDS, pool=5.0),
                max_retries=1
            ),
            temperature=0.1,
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
        )
    return openai.LLM(
        model=LLM_MODEL,
        client=openai_client.AsyncOpenAI(
            api_key=OPENAI_API_KEY or 'dummy',
            base_url=OPENAI_BASE_URL or None,
            timeout=httpx.Timeout(EFFECTIVE_LLM_REQUEST_TIMEOUT_SECONDS),
        ),
        temperature=0.1,
        max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
    )


def build_stt():
    if STT_PROVIDER == 'local':
        return openai.STT(
            model=LOCAL_STT_MODEL,
            api_key=LOCAL_STT_API_KEY,
            base_url=LOCAL_STT_BASE_URL,
        )
    from livekit.plugins import deepgram
    return deepgram.STT(model='nova-2', language='en')


def build_tts():
    if TTS_PROVIDER == 'local':
        return openai.TTS(
            model=LOCAL_TTS_MODEL,
            voice=LOCAL_TTS_VOICE,
            api_key=LOCAL_TTS_API_KEY,
            base_url=LOCAL_TTS_BASE_URL,
            response_format='wav',
        )
    return openai.TTS(voice=OPENAI_TTS_VOICE, api_key=OPENAI_API_KEY or None, base_url=OPENAI_BASE_URL or None)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting and emojis so TTS reads clean text."""
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    text = re.sub(r'[✅📅🎉🗓️📝❌⚠️🔗📌💡🎯🔔]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class CalendarBookingAssistant(Agent):
    def __init__(self):
        # Pass empty string; the property below provides fresh instructions on every access
        super().__init__(instructions='')
        self.calendar = GoogleCalendarService(
            service_account_file=GOOGLE_SERVICE_ACCOUNT_FILE,
            calendar_id=GOOGLE_CALENDAR_ID,
            timezone=GOOGLE_TIMEZONE,
        )
        self.last_booked_event_id: str | None = None
        self.last_booked_title: str | None = None

    @property
    def instructions(self):
        """Recomputed on every access so the LLM always sees the current date/time."""
        now = datetime.now(ZoneInfo(GOOGLE_TIMEZONE))
        today_iso = now.date().isoformat()

        # Only include /no_think for Qwen-family models that support it
        prefix = '/no_think\n' if 'qwen' in LLM_MODEL.lower() else ''

        return (
            f'{prefix}'
            f'You are a voice calendar assistant. Today is {now.strftime("%A %B %d %Y")}, '
            f'time is {now.strftime("%I:%M %p")}, timezone {GOOGLE_TIMEZONE}. '
            f'Calendar year for user dates without a year is {now.year}. '
            f'Use "{today_iso}" for today in tool arguments.\n'
            'OUTPUT RULES: Plain spoken text only. No markdown, no bold, no headers, no tables, '
            'no bullets, no emojis, no links. Speak naturally in short sentences.\n'
            'TOOL RULES (critical):\n'
            '- When the user asks if they are free, busy, what is on the calendar, or mentions '
            '"this weekend", "Saturday", "Sunday", "today", "tomorrow", "this week", '
            '"this month", "next month", or rolling windows like "next 10 days", call '
            'smart_calendar_lookup with their phrase directly. Do not ask them for ISO dates.\n'
            '- When they give concrete calendar days (e.g. May 15 through May 17), call '
            'calendar_date_range_lookup with start_date and end_date as YYYY-MM-DD using the '
            'year above if they did not say the year.\n'
            '- For a specific time slot, use check_availability with full ISO datetimes.\n'
            '- Do not ask for an email address unless you are about to book a meeting with an invitee.\n'
            '- After a tool returns, summarize the result in one or two short sentences.\n'
            '- IMPORTANT: When tool results include date ranges (e.g. "Queried: Saturday May 16 '
            'through Sunday May 17"), use THOSE EXACT DATES for any follow-up booking. '
            'Never guess or compute dates yourself.\n'
            'BOOKING RULES:\n'
            '- Default duration 30 minutes. Confirm title and time before booking.\n'
            '- For all-day or multi-day events (like "block my weekend" or "mark as busy"), '
            'use book_all_day_event with YYYY-MM-DD dates. Do NOT use book_meeting for these.\n'
            '- For timed meetings at specific hours, use book_meeting with ISO datetimes.\n'
            '- Attendee email is optional for booking.\n'
            '- Never say "I will check" unless you are calling the calendar tool in this same turn. '
            'For weekend/today/tomorrow/this week availability, call smart_calendar_lookup first, '
            'then answer with the actual result.'
        )

    @instructions.setter
    def instructions(self, value):
        # Ignore writes from the parent __init__; the getter always returns fresh instructions.
        pass

    def _tz(self) -> ZoneInfo:
        return ZoneInfo(GOOGLE_TIMEZONE)

    async def _calendar_call(self, label: str, func, *args):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args),
                timeout=CALENDAR_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f'{label} timed out after {CALENDAR_TIMEOUT_SECONDS:.0f}s') from exc

    # ------------------------------------------------------------------
    # Lookup tools
    # ------------------------------------------------------------------

    @function_tool
    async def smart_calendar_lookup(self, context: RunContext, query: str) -> str:
        """Look up calendar events using natural language time references.

        Use this for any availability or schedule query. Accepts natural phrases like:
        - "today", "tomorrow", "this weekend", "next weekend"
        - "this week", "this month", "next month", "last month"
        - "next 10 days", "past 14 days", "last 7 days"
        - "am I free this weekend?", "what does next week look like?"

        Args:
            query: The user's time-related phrase (pass their words directly).
        """
        preset = preset_from_natural_query(query)
        if not preset:
            # Try treating the query itself as a preset token
            maybe = (query or '').strip().lower().replace(' ', '_').replace('-', '_')
            if maybe and window_for_preset(maybe, GOOGLE_TIMEZONE):
                preset = maybe

        if not preset:
            return (
                f'I could not map "{query}" to a calendar range yet. '
                'Try phrasing like: today, this weekend, this month, next month, '
                'next 10 days, or past 14 days.'
            )

        window = window_for_preset(preset, GOOGLE_TIMEZONE)
        if window is None:
            return f'Unsupported timeframe "{preset}".'

        start, end = window
        label = preset.replace('_', ' ')
        try:
            events = await self._calendar_call(
                'calendar lookup',
                self.calendar.find_events,
                start, end, None,
            )
        except Exception as exc:
            logger.error('calendar lookup failed: %s', exc)
            return 'Calendar lookup failed. Please try again.'

        return events_to_summary(events, f'for {label}', GOOGLE_TIMEZONE, start_dt=start, end_dt=end)

    @function_tool
    async def calendar_date_range_lookup(
        self,
        context: RunContext,
        start_date: str,
        end_date: str,
    ) -> str:
        """List calendar events across calendar days (inclusive). Use YYYY-MM-DD only.

        When the user says a range like May 15 to May 17, convert to dates using the
        current calendar year from the system message if they did not specify a year.

        Args:
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD (inclusive)
        """
        sd = parse_calendar_date(start_date)
        ed = parse_calendar_date(end_date)
        if not sd or not ed:
            return 'Dates must be YYYY-MM-DD, for example 2026-05-15.'

        if ed < sd:
            sd, ed = ed, sd

        tz = self._tz()
        window_start = datetime.combine(sd, time.min, tzinfo=tz)
        window_end = datetime.combine(ed, time(23, 59, 59), tzinfo=tz)

        try:
            events = await self._calendar_call(
                'date range lookup',
                self.calendar.find_events,
                window_start, window_end, None,
            )
        except Exception as exc:
            logger.error('date range lookup failed: %s', exc)
            return 'Calendar lookup failed. Please try again.'

        label = f'from {sd.isoformat()} through {ed.isoformat()}'
        return events_to_summary(events, label, GOOGLE_TIMEZONE, start_dt=window_start, end_dt=window_end)

    # ------------------------------------------------------------------
    # Availability tools
    # ------------------------------------------------------------------

    @function_tool
    async def check_availability(
        self,
        context: RunContext,
        requested_start_iso: str,
        requested_end_iso: str,
    ) -> str:
        """Check if requested time window is free.

        Args:
            requested_start_iso: ISO datetime like 2026-05-13T15:00:00+05:30
            requested_end_iso: ISO datetime like 2026-05-13T15:30:00+05:30
        """
        miss = missing_fields(requested_start_iso=requested_start_iso, requested_end_iso=requested_end_iso)
        if miss:
            return f"Missing required details: {', '.join(miss)}."

        start, start_err = parse_dt_or_error(requested_start_iso, 'requested_start_iso', GOOGLE_TIMEZONE)
        end, end_err = parse_dt_or_error(requested_end_iso, 'requested_end_iso', GOOGLE_TIMEZONE)
        if start_err or end_err:
            return start_err or end_err or 'Invalid requested time.'

        val_err = validate_future_window(start, end, GOOGLE_TIMEZONE)
        if val_err:
            return val_err

        try:
            free = await self._calendar_call('check availability', self.calendar.is_free, start, end)
        except Exception as exc:
            logger.error('availability check failed: %s', exc)
            return 'Calendar availability check failed. Please try again.'
        return 'AVAILABLE' if free else 'BUSY'

    @function_tool
    async def suggest_time_slots(
        self,
        context: RunContext,
        window_start_iso: str,
        window_end_iso: str,
        meeting_minutes: int = MIN_SLOT_MINUTES,
    ) -> str:
        """Suggest free slots in a window when a requested time is busy.

        Args:
            window_start_iso: ISO datetime start for search window
            window_end_iso: ISO datetime end for search window
            meeting_minutes: meeting duration in minutes
        """
        miss = missing_fields(window_start_iso=window_start_iso, window_end_iso=window_end_iso)
        if miss:
            return f"Missing required details: {', '.join(miss)}."

        ws, ws_err = parse_dt_or_error(window_start_iso, 'window_start_iso', GOOGLE_TIMEZONE)
        we, we_err = parse_dt_or_error(window_end_iso, 'window_end_iso', GOOGLE_TIMEZONE)
        if ws_err or we_err:
            return ws_err or we_err or 'Invalid search window.'

        val_err = validate_future_window(ws, we, GOOGLE_TIMEZONE)
        if val_err:
            return val_err

        try:
            slots = await self._calendar_call(
                'suggest time slots',
                self.calendar.find_free_slots, ws, we, meeting_minutes, 5,
            )
        except Exception as exc:
            logger.error('slot suggestion failed: %s', exc)
            return 'Calendar slot search failed. Please try again.'
        if not slots:
            return 'No free slots found in that window.'

        lines = []
        for i, s in enumerate(slots, start=1):
            candidate_end = s.start + timedelta(minutes=meeting_minutes)
            lines.append(
                f"{i}. {s.start.strftime('%A, %B %d %Y, %I:%M %p')} to "
                f"{candidate_end.strftime('%I:%M %p')}"
            )
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Booking tools
    # ------------------------------------------------------------------

    @function_tool
    async def book_meeting(
        self,
        context: RunContext,
        title: str,
        start_iso: str,
        end_iso: str,
        description: str = '',
        attendee_email: str = '',
    ) -> str:
        """Create a calendar meeting after user confirms details.

        Args:
            title: Meeting title
            start_iso: Meeting start (ISO datetime)
            end_iso: Meeting end (ISO datetime)
            description: Optional meeting notes/agenda
            attendee_email: Optional invitee email
        """
        miss = missing_fields(title=title, start_iso=start_iso, end_iso=end_iso)
        if miss:
            return (
                f"Cannot book yet. Missing required details: {', '.join(miss)}. "
                "Ask the user for the missing details before trying again."
            )

        start, start_err = parse_dt_or_error(start_iso, 'start_iso', GOOGLE_TIMEZONE)
        end, end_err = parse_dt_or_error(end_iso, 'end_iso', GOOGLE_TIMEZONE)
        if start_err or end_err:
            return start_err or end_err or 'Invalid meeting time.'

        val_err = validate_future_window(start, end, GOOGLE_TIMEZONE)
        if val_err:
            return val_err

        try:
            is_free = await self._calendar_call('check availability', self.calendar.is_free, start, end)
        except Exception as exc:
            logger.error('pre-booking availability check failed: %s', exc)
            return 'Could not check the calendar right now. Please try again.'

        if not is_free:
            return 'Could not book: selected slot is no longer free.'

        try:
            event = await self._calendar_call(
                'create event',
                self.calendar.create_event, title, description, start, end, attendee_email,
            )
        except Exception as exc:
            logger.error('event creation failed: %s', exc)
            return 'Could not create the calendar event. Please try again.'

        self.last_booked_event_id = event.get('id')
        self.last_booked_title = event.get('summary', title)

        return (
            f'Meeting booked: {self.last_booked_title}, '
            f'{start.strftime("%A %B %d")} from {start.strftime("%I:%M %p")} to {end.strftime("%I:%M %p")}.'
        )

    @function_tool
    async def book_all_day_event(
        self,
        context: RunContext,
        title: str,
        start_date: str,
        end_date: str,
        description: str = '',
    ) -> str:
        """Create an all-day or multi-day calendar event. Use for blocking whole days.

        Use this instead of book_meeting when the user wants to block entire days
        (e.g. "mark my weekend as busy", "block Friday through Sunday").

        Args:
            title: Event title (e.g. "Busy", "Out of Office")
            start_date: First day YYYY-MM-DD (inclusive)
            end_date: Last day YYYY-MM-DD (inclusive)
            description: Optional event notes
        """
        miss = missing_fields(title=title, start_date=start_date, end_date=end_date)
        if miss:
            return (
                f"Cannot book yet. Missing required details: {', '.join(miss)}. "
                "Ask the user for the missing details before trying again."
            )

        sd = parse_calendar_date(start_date)
        ed = parse_calendar_date(end_date)
        if not sd or not ed:
            return 'Dates must be YYYY-MM-DD, for example 2026-05-16.'

        if ed < sd:
            sd, ed = ed, sd

        today = datetime.now(self._tz()).date()
        if ed < today:
            return (
                f"The requested dates are in the past. Today is {today.isoformat()}. "
                "Ask the user to confirm future dates."
            )

        # Google Calendar all-day events use exclusive end date
        exclusive_end = ed + timedelta(days=1)

        try:
            event = await self._calendar_call(
                'create all-day event',
                self.calendar.create_all_day_event,
                title, description, sd.isoformat(), exclusive_end.isoformat(),
            )
        except Exception as exc:
            logger.error('all-day event creation failed: %s', exc)
            return 'Could not create the calendar event. Please try again.'

        self.last_booked_event_id = event.get('id')
        self.last_booked_title = event.get('summary', title)

        return (
            f'All-day event booked: {self.last_booked_title}, '
            f'{sd.strftime("%A %B %d")} through {ed.strftime("%A %B %d")}.'
        )

    # ------------------------------------------------------------------
    # Mutation tools
    # ------------------------------------------------------------------

    @function_tool
    async def cancel_last_event(self, context: RunContext) -> str:
        """Cancel the most recently booked event during this session."""
        if not self.last_booked_event_id:
            return "There is no recently booked event to cancel in this session."

        try:
            await self._calendar_call('cancel event', self.calendar.delete_event, self.last_booked_event_id)
            title = self.last_booked_title or 'the event'
            self.last_booked_event_id = None
            self.last_booked_title = None
            return f"Successfully cancelled {title}."
        except Exception as exc:
            logger.error('event cancellation failed: %s', exc)
            return 'I could not cancel the event on the calendar.'

    @function_tool
    async def rename_last_event(self, context: RunContext, new_title: str) -> str:
        """Rename the most recently booked event during this session.

        Args:
            new_title: The new title for the event.
        """
        if not self.last_booked_event_id:
            return "There is no recently booked event to rename in this session."

        if not new_title:
            return "You must specify a new title."

        try:
            event = await self._calendar_call(
                'rename event',
                self.calendar.update_event_summary,
                self.last_booked_event_id, new_title,
            )
            self.last_booked_title = event.get('summary') or new_title
            return f"Successfully renamed the event to {self.last_booked_title}."
        except Exception as exc:
            logger.error('event rename failed: %s', exc)
            return 'I could not rename the event on the calendar.'

    @function_tool
    async def reschedule_meeting(
        self,
        context: RunContext,
        search_query: str,
        search_window_start_iso: str,
        search_window_end_iso: str,
        new_start_iso: str,
        new_end_iso: str,
    ) -> str:
        """Find an existing meeting by title and move it to a new time.

        Args:
            search_query: Title or keyword to find the existing meeting
            search_window_start_iso: ISO datetime start of window to search for the meeting
            search_window_end_iso: ISO datetime end of window to search for the meeting
            new_start_iso: New meeting start (ISO datetime)
            new_end_iso: New meeting end (ISO datetime)
        """
        miss = missing_fields(
            search_query=search_query,
            search_window_start_iso=search_window_start_iso,
            search_window_end_iso=search_window_end_iso,
            new_start_iso=new_start_iso,
            new_end_iso=new_end_iso,
        )
        if miss:
            return f"Missing required details: {', '.join(miss)}."

        ws, ws_err = parse_dt_or_error(search_window_start_iso, 'search_window_start_iso', GOOGLE_TIMEZONE)
        we, we_err = parse_dt_or_error(search_window_end_iso, 'search_window_end_iso', GOOGLE_TIMEZONE)
        ns, ns_err = parse_dt_or_error(new_start_iso, 'new_start_iso', GOOGLE_TIMEZONE)
        ne, ne_err = parse_dt_or_error(new_end_iso, 'new_end_iso', GOOGLE_TIMEZONE)
        for err in [ws_err, we_err, ns_err, ne_err]:
            if err:
                return err

        val_err = validate_future_window(ns, ne, GOOGLE_TIMEZONE)
        if val_err:
            return val_err

        try:
            events = await self._calendar_call(
                'find event', self.calendar.find_events, ws, we, search_query,
            )
        except Exception as exc:
            logger.error('event search failed: %s', exc)
            return 'Calendar search failed. Please try again.'

        if not events:
            return f'No event matching "{search_query}" found in that time window.'

        target = events[0]
        event_id = target.get('id')
        event_title = target.get('summary', 'Untitled')

        try:
            is_free = await self._calendar_call('check new time', self.calendar.is_free, ns, ne)
        except Exception as exc:
            logger.error('reschedule availability check failed: %s', exc)
            return 'Could not check the new time slot. Please try again.'

        if not is_free:
            return 'The new time slot is not available. Try "suggest_time_slots" to find open times.'

        try:
            await self._calendar_call(
                'reschedule event', self.calendar.update_event_time, event_id, ns, ne,
            )
        except Exception as exc:
            logger.error('reschedule failed: %s', exc)
            return 'Could not reschedule the event. Please try again.'

        self.last_booked_event_id = event_id
        self.last_booked_title = event_title

        return (
            f'Successfully rescheduled "{event_title}" to '
            f'{ns.strftime("%A, %B %d %Y, %I:%M %p")} - '
            f'{ne.strftime("%I:%M %p")}.'
        )


# ======================================================================
# LiveKit session entrypoint
# ======================================================================

server = agents.AgentServer()
server.prewarm = prewarm


@server.rtc_session(agent_name='calendar-agent')
async def entrypoint(ctx: agents.JobContext):
    # Use prewarmed VAD if available, otherwise load fresh
    vad = ctx.proc.userdata.get('vad') or silero.VAD.load()
    session = AgentSession(
        stt=build_stt(),
        llm=build_llm(),
        tts=build_tts(),
        vad=vad,
        aec_warmup_duration=0.8,
        conn_options=SessionConnectOptions(
            stt_conn_options=APIConnectOptions(timeout=60.0, max_retry=2),
            llm_conn_options=APIConnectOptions(timeout=EFFECTIVE_LLM_REQUEST_TIMEOUT_SECONDS, max_retry=1),
            tts_conn_options=APIConnectOptions(timeout=60.0, max_retry=2),
        ),
        turn_handling={
            'endpointing': {'mode': 'fixed', 'min_delay': 0.45, 'max_delay': 2.0},
            'interruption': {
                'enabled': True,
                'discard_audio_if_uninterruptible': True,
                'min_duration': 0.25,
                'min_words': 0,
                'resume_false_interruption': True,
            },
            'preemptive_generation': {'enabled': False},
        },
    )

    assistant = CalendarBookingAssistant()

    def on_user_input(ev):
        if ev.is_final:
            logger.info('transcript final: %r', ev.transcript)

    def on_speech_created(ev):
        logger.info('speech created source=%s', ev.source)

    def on_agent_state_changed(ev):
        logger.info('state %s -> %s', ev.old_state, ev.new_state)

    def on_error(ev):
        logger.error('runtime error: %s', ev.error)

    session.on('user_input_transcribed', on_user_input)
    session.on('speech_created', on_speech_created)
    session.on('agent_state_changed', on_agent_state_changed)
    session.on('error', on_error)

    await session.start(room=ctx.room, agent=assistant)
    await session.generate_reply(
        instructions=(
            'Greet briefly. Say you can check their calendar or book meetings. '
            'If they ask about availability or weekends, use smart_calendar_lookup or '
            'calendar_date_range_lookup right away instead of asking for date formats.'
        )
    )


if __name__ == '__main__':
    agents.cli.run_app(server)
