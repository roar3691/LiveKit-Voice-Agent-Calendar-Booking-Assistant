"""Date/time utility functions for the calendar booking assistant.

Extracted from agent.py to keep the agent class focused on tool definitions
and session management.  All date parsing, window calculation, and preset
resolution lives here so it can be tested independently without mocking
the entire LiveKit stack.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_dt(iso_text: str, timezone: str) -> datetime:
    """Parse an ISO datetime string, applying *timezone* if the value is naive."""
    dt = datetime.fromisoformat(iso_text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone))
    return dt


def parse_dt_or_error(
    iso_text: str,
    field_name: str,
    timezone: str,
) -> tuple[datetime | None, str | None]:
    """Safely parse an ISO datetime; return ``(dt, None)`` or ``(None, error_msg)``."""
    try:
        return parse_dt(iso_text, timezone), None
    except ValueError:
        return None, (
            f"Missing or invalid {field_name}. "
            "Ask the user for a specific date and time."
        )


def parse_calendar_date(value: str) -> datetime.date | None:
    """Parse a ``YYYY-MM-DD`` string (first 10 chars) into a :class:`date`."""
    raw = (value or '').strip()[:10]
    if not raw:
        return None
    try:
        return datetime.strptime(raw, '%Y-%m-%d').date()
    except ValueError:
        return None


def missing_fields(**fields: str) -> list[str]:
    """Return names of any keyword arguments that are empty / whitespace-only."""
    return [name for name, value in fields.items() if not str(value or '').strip()]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_future_window(
    start: datetime,
    end: datetime,
    timezone: str,
) -> str | None:
    """Return an error string if *start*–*end* is invalid, else ``None``."""
    if end <= start:
        return (
            'End time must be after start time. '
            'Ask the user to clarify the duration or end time.'
        )

    now = datetime.now(ZoneInfo(timezone))
    if start < now - timedelta(minutes=5):
        return (
            f"Requested time is in the past. Current date/time is "
            f"{now.strftime('%A, %B %d %Y, %I:%M %p')} {timezone}. "
            "Ask the user to confirm a future date and time."
        )
    return None


# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------

def event_time(event: dict, key: str, timezone: str) -> datetime | None:
    """Extract a datetime from a Google Calendar event's *start* or *end* dict."""
    tz = ZoneInfo(timezone)
    raw = event.get(key, {}).get('dateTime') or event.get(key, {}).get('date')
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def events_to_summary(
    events: list,
    window_label: str,
    timezone: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> str:
    """Format a list of Google Calendar events into a voice-friendly summary."""
    date_range_note = ''
    if start_dt and end_dt:
        date_range_note = (
            f'Queried: {start_dt.strftime("%A %B %d %Y")} through '
            f'{end_dt.strftime("%A %B %d %Y")} '
            f'(start_date={start_dt.strftime("%Y-%m-%d")}, '
            f'end_date={end_dt.strftime("%Y-%m-%d")}). '
        )

    if not events:
        return (
            f'{date_range_note}No events scheduled {window_label}. '
            'You look free for that whole period.'
        )

    lines = []
    for ev in events[:8]:
        title = ev.get('summary') or 'Untitled event'
        start = event_time(ev, 'start', timezone)
        end = event_time(ev, 'end', timezone)
        if start and end:
            lines.append(
                f"{title}, {start.strftime('%a %b %d %I:%M %p')} "
                f"to {end.strftime('%I:%M %p')}"
            )
        elif start:
            lines.append(f"{title}, starting {start.strftime('%a %b %d %I:%M %p')}")
        else:
            lines.append(title)

    extra = len(events) - len(lines)
    if extra > 0:
        lines.append(f'and {extra} more event{"s" if extra != 1 else ""}.')

    return (
        f'{date_range_note}Here is what is on the calendar {window_label}:\n'
        + '\n'.join(lines)
    )


# ---------------------------------------------------------------------------
# Weekend bounds
# ---------------------------------------------------------------------------

def this_weekend_bounds(local_now: datetime, timezone: str) -> tuple[datetime, datetime]:
    """Return (Saturday 00:00, Sunday 23:59:59) for the current weekend."""
    tz = ZoneInfo(timezone)
    d = local_now.astimezone(tz).date()
    wd = d.weekday()
    if wd < 5:
        sat_date = d + timedelta(days=(5 - wd))
    elif wd == 5:
        sat_date = d
    else:
        sat_date = d - timedelta(days=1)
    sun_date = sat_date + timedelta(days=1)
    start = datetime.combine(sat_date, time.min, tzinfo=tz)
    end = datetime.combine(sun_date, time(23, 59, 59), tzinfo=tz)
    return start, end


# ---------------------------------------------------------------------------
# Preset time windows
# ---------------------------------------------------------------------------

def _month_bounds(
    year: int,
    month: int,
    tz: ZoneInfo,
) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=tz)
    if month == 12:
        next_start = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=tz)
    else:
        next_start = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=tz)
    end = next_start - timedelta(seconds=1)
    return start, end


def window_for_preset(preset: str, timezone: str) -> tuple[datetime, datetime] | None:
    """Convert a preset token (e.g. ``'this_weekend'``) into a datetime range.

    Returns ``None`` for unrecognised tokens.
    """
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    local = now.astimezone(tz)
    d = local.date()
    preset = preset.lower().strip().replace(' ', '_').replace('-', '_')

    if preset == 'today':
        return (
            datetime.combine(d, time.min, tzinfo=tz),
            datetime.combine(d, time(23, 59, 59), tzinfo=tz),
        )

    if preset == 'tomorrow':
        d2 = d + timedelta(days=1)
        return (
            datetime.combine(d2, time.min, tzinfo=tz),
            datetime.combine(d2, time(23, 59, 59), tzinfo=tz),
        )

    if preset in ('this_weekend', 'weekend'):
        return this_weekend_bounds(local, timezone)

    if preset == 'next_weekend':
        s0, _ = this_weekend_bounds(local, timezone)
        sat_next = s0.date() + timedelta(days=7)
        sun_next = sat_next + timedelta(days=1)
        return (
            datetime.combine(sat_next, time.min, tzinfo=tz),
            datetime.combine(sun_next, time(23, 59, 59), tzinfo=tz),
        )

    if preset in ('this_week', 'thisweek'):
        wd = d.weekday()
        mon = d - timedelta(days=wd)
        sun = mon + timedelta(days=6)
        return (
            datetime.combine(mon, time.min, tzinfo=tz),
            datetime.combine(sun, time(23, 59, 59), tzinfo=tz),
        )

    if preset in ('last_7_days', 'past_week', 'last_week'):
        return (
            datetime.combine(d - timedelta(days=7), time.min, tzinfo=tz),
            datetime.combine(d, time(23, 59, 59), tzinfo=tz),
        )

    if preset in ('this_month', 'thismonth', 'current_month'):
        return _month_bounds(d.year, d.month, tz)

    if preset in ('next_month', 'nextmonth'):
        year = d.year + 1 if d.month == 12 else d.year
        month = 1 if d.month == 12 else d.month + 1
        return _month_bounds(year, month, tz)

    if preset in ('last_month', 'previous_month', 'prev_month'):
        year = d.year - 1 if d.month == 1 else d.year
        month = 12 if d.month == 1 else d.month - 1
        return _month_bounds(year, month, tz)

    if preset in ('this_year', 'thisyear', 'current_year'):
        return (
            datetime(d.year, 1, 1, 0, 0, 0, tzinfo=tz),
            datetime(d.year, 12, 31, 23, 59, 59, tzinfo=tz),
        )

    # Fixed rolling windows
    _fixed = {
        'next_7_days': (0, 7),
        'next7days': (0, 7),
        'next_14_days': (0, 14),
        'next14days': (0, 14),
        'next_30_days': (0, 30),
        'next30days': (0, 30),
        'last_30_days': (-30, 0),
        'past_30_days': (-30, 0),
        'last30days': (-30, 0),
    }
    if preset in _fixed:
        back, fwd = _fixed[preset]
        return (
            datetime.combine(d + timedelta(days=back), time.min, tzinfo=tz),
            datetime.combine(d + timedelta(days=fwd), time(23, 59, 59), tzinfo=tz),
        )

    # Generic parser: next_N_days / last_N_days
    parts = [p for p in preset.split('_') if p]
    if len(parts) >= 3 and parts[0] in ('next', 'last', 'past') and parts[1].isdigit():
        count = int(parts[1])
        unit = parts[2]
        if 1 <= count <= 90 and unit in ('day', 'days'):
            if parts[0] == 'next':
                return (
                    datetime.combine(d, time.min, tzinfo=tz),
                    datetime.combine(d + timedelta(days=count), time(23, 59, 59), tzinfo=tz),
                )
            return (
                datetime.combine(d - timedelta(days=count), time.min, tzinfo=tz),
                datetime.combine(d, time(23, 59, 59), tzinfo=tz),
            )

    return None


# ---------------------------------------------------------------------------
# Natural language → preset token
# ---------------------------------------------------------------------------

def preset_from_natural_query(query: str) -> str | None:
    """Map free-form text like ``"am I free this weekend?"`` to a preset token."""
    text = (query or '').strip().lower()
    normalized = text.replace('-', ' ')

    direct_map = {
        'today': 'today',
        'tomorrow': 'tomorrow',
        'this week': 'this_week',
        'weekend': 'this_weekend',
        'this weekend': 'this_weekend',
        'next weekend': 'next_weekend',
        'this month': 'this_month',
        'current month': 'this_month',
        'next month': 'next_month',
        'last month': 'last_month',
        'previous month': 'last_month',
        'this year': 'this_year',
        'current year': 'this_year',
        'last 7 days': 'last_7_days',
        'past 7 days': 'last_7_days',
    }
    # Check longest phrases first so "this weekend" matches before "this week"
    for phrase, preset in sorted(direct_map.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in normalized:
            return preset

    m = re.search(r'\b(next|last|past)\s+(\d{1,2})\s+day[s]?\b', normalized)
    if m:
        direction = m.group(1)
        days = int(m.group(2))
        if 1 <= days <= 90:
            if direction == 'next':
                return f'next_{days}_days'
            return f'last_{days}_days'

    if 'saturday' in normalized or 'sunday' in normalized:
        if 'next' in normalized:
            return 'next_weekend'
        return 'this_weekend'

    return None
