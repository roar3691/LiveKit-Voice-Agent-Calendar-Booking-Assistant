from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
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

load_dotenv('.env')

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


def strip_markdown(text: str) -> str:
    """Remove Markdown formatting so TTS speaks clean text."""
    text = re.sub(r'#{1,6}\s*', '', text)           # ## headers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)     # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)          # *italic*
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)   # [link](url)
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)   # numbered lists
    text = re.sub(r'\|', ', ', text)                  # table separators
    text = re.sub(r'^[\-=]{3,}$', '', text, flags=re.MULTILINE)  # horizontal rules
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)  # `code`
    text = re.sub(r'[✅📅🎉🗓️📝❌⚠️🔗📌💡🎯🔔🍽️]', '', text)  # emojis
    text = re.sub(r'\n{3,}', '\n\n', text)            # collapse newlines
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
            f'Use "{today_iso}" for today in tool arguments.\n'
            'OUTPUT RULES: Plain spoken text only. No markdown, no bold, no headers, no tables, '
            'no bullets, no emojis, no links. Speak naturally in short sentences.\n'
            'WORKFLOW: Use tools to check availability then book. Default duration is 30 minutes. '
            'Attendee email is optional. Ask one question at a time. '
            'Confirm details before booking. State outcomes clearly.'
        )

    @instructions.setter
    def instructions(self, value):
        # Ignore writes from the parent __init__; the getter always returns fresh instructions.
        pass

    def _parse_dt(self, iso_text: str) -> datetime:
        dt = datetime.fromisoformat(iso_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(GOOGLE_TIMEZONE))
        return dt

    def _missing_fields(self, **fields: str) -> list[str]:
        return [name for name, value in fields.items() if not str(value or '').strip()]

    async def _calendar_call(self, label: str, func, *args):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args),
                timeout=CALENDAR_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f'{label} timed out after {CALENDAR_TIMEOUT_SECONDS:.0f}s') from exc

    def _parse_dt_or_error(self, iso_text: str, field_name: str) -> tuple[datetime | None, str | None]:
        try:
            return self._parse_dt(iso_text), None
        except ValueError:
            return None, f"Missing or invalid {field_name}. Ask the user for a specific date and time."

    def _validate_future_window(self, start: datetime, end: datetime) -> str | None:
        if end <= start:
            return 'End time must be after start time. Ask the user to clarify the duration or end time.'

        now = datetime.now(ZoneInfo(GOOGLE_TIMEZONE))
        if start < now - timedelta(minutes=5):
            return (
                f"Requested time is in the past. Current date/time is "
                f"{now.strftime('%A, %B %d %Y, %I:%M %p')} {GOOGLE_TIMEZONE}. "
                "Ask the user to confirm a future date and time."
            )
        return None

    def _event_time(self, event: dict, key: str) -> datetime | None:
        raw = event.get(key, {}).get('dateTime') or event.get(key, {}).get('date')
        if not raw:
            return None
        dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(GOOGLE_TIMEZONE))
        return dt.astimezone(ZoneInfo(GOOGLE_TIMEZONE))

    @function_tool
    async def list_calendar_events(
        self,
        context: RunContext,
        window_start_iso: str,
        window_end_iso: str,
        query: str = "",
    ) -> str:
        """List calendar events in a time window.

        Args:
            window_start_iso: ISO datetime start for lookup window
            window_end_iso: ISO datetime end for lookup window
            query: Optional text to search for in event title/details
        """
        missing = self._missing_fields(
            window_start_iso=window_start_iso,
            window_end_iso=window_end_iso,
        )
        if missing:
            return f"Missing required details: {', '.join(missing)}."

        window_start, start_error = self._parse_dt_or_error(window_start_iso, 'window_start_iso')
        window_end, end_error = self._parse_dt_or_error(window_end_iso, 'window_end_iso')
        if start_error or end_error:
            return start_error or end_error or 'Invalid lookup window.'

        try:
            events = await self._calendar_call(
                'list calendar events',
                self.calendar.find_events,
                window_start,
                window_end,
                query or None,
            )
        except Exception as exc:
            print(f'[calendar] unable to list events: {exc}')
            return 'Calendar lookup failed. Please try again.'
        if not events:
            return 'No matching events found.'

        lines = []
        for event in events[:5]:
            title = event.get('summary') or 'Untitled event'
            start = self._event_time(event, 'start')
            end = self._event_time(event, 'end')
            if start and end:
                lines.append(f"{title}: {start.strftime('%I:%M %p')} to {end.strftime('%I:%M %p')}")
            else:
                lines.append(title)
        if len(events) > len(lines):
            lines.append(f"...and {len(events) - len(lines)} more.")
        return '\n'.join(lines)

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
        missing = self._missing_fields(
            requested_start_iso=requested_start_iso,
            requested_end_iso=requested_end_iso,
        )
        if missing:
            return f"Missing required details: {', '.join(missing)}."

        start, start_error = self._parse_dt_or_error(requested_start_iso, 'requested_start_iso')
        end, end_error = self._parse_dt_or_error(requested_end_iso, 'requested_end_iso')
        if start_error or end_error:
            return start_error or end_error or 'Invalid requested time.'

        validation_error = self._validate_future_window(start, end)
        if validation_error:
            return validation_error

        try:
            free = await self._calendar_call('check availability', self.calendar.is_free, start, end)
        except Exception as exc:
            print(f'[calendar] unable to check availability: {exc}')
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
        missing = self._missing_fields(
            window_start_iso=window_start_iso,
            window_end_iso=window_end_iso,
        )
        if missing:
            return f"Missing required details: {', '.join(missing)}."

        window_start, start_error = self._parse_dt_or_error(window_start_iso, 'window_start_iso')
        window_end, end_error = self._parse_dt_or_error(window_end_iso, 'window_end_iso')
        if start_error or end_error:
            return start_error or end_error or 'Invalid search window.'

        validation_error = self._validate_future_window(window_start, window_end)
        if validation_error:
            return validation_error

        try:
            slots = await self._calendar_call(
                'suggest time slots',
                self.calendar.find_free_slots,
                window_start,
                window_end,
                meeting_minutes,
                5,
            )
        except Exception as exc:
            print(f'[calendar] unable to suggest slots: {exc}')
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
        missing = self._missing_fields(
            title=title,
            start_iso=start_iso,
            end_iso=end_iso,
        )
        if missing:
            return (
                f"Cannot book yet. Missing required details: {', '.join(missing)}. "
                "Ask the user for the missing details before trying again."
            )

        start, start_error = self._parse_dt_or_error(start_iso, 'start_iso')
        end, end_error = self._parse_dt_or_error(end_iso, 'end_iso')
        if start_error or end_error:
            return start_error or end_error or 'Invalid meeting time.'

        validation_error = self._validate_future_window(start, end)
        if validation_error:
            return validation_error

        try:
            is_free = await self._calendar_call('check availability', self.calendar.is_free, start, end)
        except Exception as exc:
            print(f'[calendar] unable to check availability before booking: {exc}')
            return 'Could not check the calendar right now. Please try again.'

        if not is_free:
            return 'Could not book: selected slot is no longer free.'

        try:
            event = await self._calendar_call(
                'create event',
                self.calendar.create_event,
                title,
                description,
                start,
                end,
                attendee_email,
            )
        except Exception as exc:
            print(f'[calendar] unable to create event: {exc}')
            return 'Could not create the calendar event. Please try again.'

        self.last_booked_event_id = event.get('id')
        self.last_booked_title = event.get('summary', title)

        html_link = event.get('htmlLink', '')
        return (
            f'Meeting booked: {self.last_booked_title}, '
            f'{start.strftime("%A %B %d")} from {start.strftime("%I:%M %p")} to {end.strftime("%I:%M %p")}.'
        )

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
            print(f'[calendar] unable to cancel event: {exc}')
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
                self.last_booked_event_id,
                new_title,
            )
            self.last_booked_title = event.get('summary') or new_title
            return f"Successfully renamed the event to {self.last_booked_title}."
        except Exception as exc:
            print(f'[calendar] unable to rename event: {exc}')
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
        missing = self._missing_fields(
            search_query=search_query,
            search_window_start_iso=search_window_start_iso,
            search_window_end_iso=search_window_end_iso,
            new_start_iso=new_start_iso,
            new_end_iso=new_end_iso,
        )
        if missing:
            return f"Missing required details: {', '.join(missing)}."

        window_start, ws_err = self._parse_dt_or_error(search_window_start_iso, 'search_window_start_iso')
        window_end, we_err = self._parse_dt_or_error(search_window_end_iso, 'search_window_end_iso')
        new_start, ns_err = self._parse_dt_or_error(new_start_iso, 'new_start_iso')
        new_end, ne_err = self._parse_dt_or_error(new_end_iso, 'new_end_iso')
        for err in [ws_err, we_err, ns_err, ne_err]:
            if err:
                return err

        validation_error = self._validate_future_window(new_start, new_end)
        if validation_error:
            return validation_error

        # Find the event
        try:
            events = await self._calendar_call(
                'find event to reschedule',
                self.calendar.find_events,
                window_start,
                window_end,
                search_query,
            )
        except Exception as exc:
            print(f'[calendar] unable to search events: {exc}')
            return 'Calendar search failed. Please try again.'

        if not events:
            return f'No event matching "{search_query}" found in that time window.'

        target_event = events[0]
        event_id = target_event.get('id')
        event_title = target_event.get('summary', 'Untitled')

        # Check new time is free
        try:
            is_free = await self._calendar_call('check new time', self.calendar.is_free, new_start, new_end)
        except Exception as exc:
            print(f'[calendar] unable to check availability for reschedule: {exc}')
            return 'Could not check the new time slot. Please try again.'

        if not is_free:
            return f'The new time slot is not available. Try "suggest_time_slots" to find open times.'

        # Move the event
        try:
            updated = await self._calendar_call(
                'reschedule event',
                self.calendar.update_event_time,
                event_id,
                new_start,
                new_end,
            )
        except Exception as exc:
            print(f'[calendar] unable to reschedule event: {exc}')
            return 'Could not reschedule the event. Please try again.'

        self.last_booked_event_id = event_id
        self.last_booked_title = event_title

        return (
            f'Successfully rescheduled "{event_title}" to '
            f'{new_start.strftime("%A, %B %d %Y, %I:%M %p")} - '
            f'{new_end.strftime("%I:%M %p")}.'
        )


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
        # Strip markdown/emoji from LLM output before TTS speaks it
        tts_text_transforms=[
            'filter_markdown',
            'filter_emoji',
            strip_markdown,
        ],
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
            print(f'[agent] transcript final: {ev.transcript!r}')

    def on_speech_created(ev):
        print(f'[agent] speech created source={ev.source}')

    def on_agent_state_changed(ev):
        print(f'[agent] state {ev.old_state} -> {ev.new_state}')

    def on_error(ev):
        print(f'[agent] runtime error: {ev.error}')

    session.on('user_input_transcribed', on_user_input)
    session.on('speech_created', on_speech_created)
    session.on('agent_state_changed', on_agent_state_changed)
    session.on('error', on_error)

    await session.start(room=ctx.room, agent=assistant)
    await session.generate_reply(
        instructions='Greet the user and offer to help schedule, reschedule, or manage their calendar appointments.'
    )


if __name__ == '__main__':
    agents.cli.run_app(server)

