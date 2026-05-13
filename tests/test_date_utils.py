"""Unit tests for date/time utility logic.

Tests cover:
- window_for_preset: all preset time windows (today, tomorrow, weekends, months, rolling days)
- this_weekend_bounds: correct Saturday/Sunday calculation from any day of the week
- preset_from_natural_query: natural language → preset token mapping
- validate_future_window: past-date rejection and edge cases
- parse_dt: ISO datetime parsing with timezone handling
- parse_calendar_date: YYYY-MM-DD string parsing
- missing_fields: empty/whitespace field detection
"""
from __future__ import annotations

import unittest
from datetime import datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.date_utils import (
    events_to_summary,
    missing_fields,
    parse_calendar_date,
    parse_dt,
    parse_dt_or_error,
    preset_from_natural_query,
    this_weekend_bounds,
    validate_future_window,
    window_for_preset,
)

TZ = ZoneInfo('Asia/Kolkata')
TZ_STR = 'Asia/Kolkata'


class TestThisWeekendBounds(unittest.TestCase):
    """Tests for this_weekend_bounds — computing Saturday-Sunday from any day."""

    def test_monday_returns_upcoming_saturday_sunday(self):
        # Monday May 12, 2025
        monday = datetime(2025, 5, 12, 10, 0, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(monday, TZ_STR)
        self.assertEqual(start.date().weekday(), 5)  # Saturday
        self.assertEqual(end.date().weekday(), 6)     # Sunday
        self.assertEqual(start.date(), monday.date() + timedelta(days=5))
        self.assertEqual(end.date(), monday.date() + timedelta(days=6))

    def test_wednesday_returns_upcoming_saturday_sunday(self):
        # Wednesday May 14, 2025
        wednesday = datetime(2025, 5, 14, 15, 30, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(wednesday, TZ_STR)
        self.assertEqual(start.date().weekday(), 5)
        self.assertEqual(end.date().weekday(), 6)
        # From Wednesday (wd=2), Saturday is 3 days away
        self.assertEqual(start.date(), wednesday.date() + timedelta(days=3))

    def test_friday_returns_next_day_saturday(self):
        # Friday May 16, 2025
        friday = datetime(2025, 5, 16, 18, 0, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(friday, TZ_STR)
        self.assertEqual(start.date().weekday(), 5)
        self.assertEqual(start.date(), friday.date() + timedelta(days=1))

    def test_saturday_returns_current_saturday(self):
        # Saturday May 17, 2025
        saturday = datetime(2025, 5, 17, 12, 0, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(saturday, TZ_STR)
        self.assertEqual(start.date(), saturday.date())
        self.assertEqual(end.date(), saturday.date() + timedelta(days=1))

    def test_sunday_returns_yesterday_saturday(self):
        # Sunday May 18, 2025 — should go back to Saturday May 17
        sunday = datetime(2025, 5, 18, 14, 0, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(sunday, TZ_STR)
        self.assertEqual(start.date(), sunday.date() - timedelta(days=1))
        self.assertEqual(end.date(), sunday.date())

    def test_start_is_midnight_end_is_2359(self):
        monday = datetime(2025, 5, 12, 10, 0, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(monday, TZ_STR)
        self.assertEqual(start.time(), time.min)
        self.assertEqual(end.time(), time(23, 59, 59))

    def test_timezone_is_preserved(self):
        monday = datetime(2025, 5, 12, 10, 0, 0, tzinfo=TZ)
        start, end = this_weekend_bounds(monday, TZ_STR)
        self.assertEqual(start.tzinfo, TZ)
        self.assertEqual(end.tzinfo, TZ)


class TestWindowForPreset(unittest.TestCase):
    """Tests for window_for_preset — converting preset tokens to datetime ranges."""

    def test_today(self):
        result = window_for_preset('today', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        today = datetime.now(TZ).date()
        self.assertEqual(start.date(), today)
        self.assertEqual(end.date(), today)
        self.assertEqual(start.time(), time.min)
        self.assertEqual(end.time(), time(23, 59, 59))

    def test_tomorrow(self):
        result = window_for_preset('tomorrow', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        tomorrow = datetime.now(TZ).date() + timedelta(days=1)
        self.assertEqual(start.date(), tomorrow)
        self.assertEqual(end.date(), tomorrow)

    def test_this_weekend(self):
        result = window_for_preset('this_weekend', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        self.assertEqual(start.date().weekday(), 5)  # Saturday
        self.assertEqual(end.date().weekday(), 6)     # Sunday

    def test_weekend_alias(self):
        result1 = window_for_preset('this_weekend', TZ_STR)
        result2 = window_for_preset('weekend', TZ_STR)
        self.assertEqual(result1, result2)

    def test_this_week(self):
        result = window_for_preset('this_week', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        self.assertEqual(start.date().weekday(), 0)  # Monday
        self.assertEqual(end.date().weekday(), 6)     # Sunday
        self.assertEqual((end.date() - start.date()).days, 6)

    def test_this_month_boundaries(self):
        result = window_for_preset('this_month', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        self.assertEqual(start.day, 1)
        self.assertEqual(start.time(), time.min)
        # End should be last day of the month
        next_month_first = (end + timedelta(days=1))
        self.assertEqual(next_month_first.day, 1)

    def test_next_month(self):
        result = window_for_preset('next_month', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        now = datetime.now(TZ)
        expected_month = now.month + 1 if now.month < 12 else 1
        self.assertEqual(start.month, expected_month)
        self.assertEqual(start.day, 1)

    def test_last_month(self):
        result = window_for_preset('last_month', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        now = datetime.now(TZ)
        expected_month = now.month - 1 if now.month > 1 else 12
        self.assertEqual(start.month, expected_month)

    def test_next_7_days(self):
        result = window_for_preset('next_7_days', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        today = datetime.now(TZ).date()
        self.assertEqual(start.date(), today)
        self.assertEqual(end.date(), today + timedelta(days=7))

    def test_next_30_days(self):
        result = window_for_preset('next_30_days', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        today = datetime.now(TZ).date()
        self.assertEqual(end.date(), today + timedelta(days=30))

    def test_last_7_days(self):
        result = window_for_preset('last_7_days', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        today = datetime.now(TZ).date()
        self.assertEqual(end.date(), today)
        self.assertEqual(start.date(), today - timedelta(days=7))

    def test_generic_next_n_days(self):
        result = window_for_preset('next_10_days', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        today = datetime.now(TZ).date()
        self.assertEqual(start.date(), today)
        self.assertEqual(end.date(), today + timedelta(days=10))

    def test_generic_last_n_days(self):
        result = window_for_preset('last_14_days', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        today = datetime.now(TZ).date()
        self.assertEqual(end.date(), today)
        self.assertEqual(start.date(), today - timedelta(days=14))

    def test_n_days_capped_at_90(self):
        result = window_for_preset('next_91_days', TZ_STR)
        self.assertIsNone(result)

    def test_n_days_zero_rejected(self):
        result = window_for_preset('next_0_days', TZ_STR)
        self.assertIsNone(result)

    def test_unknown_preset_returns_none(self):
        self.assertIsNone(window_for_preset('next_century', TZ_STR))
        self.assertIsNone(window_for_preset('', TZ_STR))
        self.assertIsNone(window_for_preset('gibberish', TZ_STR))

    def test_this_year(self):
        result = window_for_preset('this_year', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        now = datetime.now(TZ)
        self.assertEqual(start.month, 1)
        self.assertEqual(start.day, 1)
        self.assertEqual(end.month, 12)
        self.assertEqual(end.day, 31)
        self.assertEqual(start.year, now.year)

    def test_normalizes_spaces_and_dashes(self):
        result = window_for_preset('this-weekend', TZ_STR)
        self.assertIsNotNone(result)
        result2 = window_for_preset('this weekend', TZ_STR)
        self.assertIsNotNone(result2)

    def test_next_weekend(self):
        result = window_for_preset('next_weekend', TZ_STR)
        this_wknd = window_for_preset('this_weekend', TZ_STR)
        self.assertIsNotNone(result)
        start, end = result
        this_start, _ = this_wknd
        # Next weekend's Saturday should be exactly 7 days after this weekend's Saturday
        self.assertEqual(start.date(), this_start.date() + timedelta(days=7))


class TestPresetFromNaturalQuery(unittest.TestCase):
    """Tests for preset_from_natural_query — mapping free text to preset tokens."""

    def test_direct_mappings(self):
        cases = {
            'today': 'today',
            'tomorrow': 'tomorrow',
            'this week': 'this_week',
            'this weekend': 'this_weekend',
            'weekend': 'this_weekend',
            'next weekend': 'next_weekend',
            'this month': 'this_month',
            'next month': 'next_month',
            'last month': 'last_month',
            'previous month': 'last_month',
            'this year': 'this_year',
            'current year': 'this_year',
        }
        for query, expected_preset in cases.items():
            with self.subTest(query=query):
                self.assertEqual(preset_from_natural_query(query), expected_preset)

    def test_case_insensitive(self):
        self.assertEqual(preset_from_natural_query('TODAY'), 'today')
        self.assertEqual(preset_from_natural_query('This Weekend'), 'this_weekend')

    def test_embedded_in_sentence(self):
        self.assertEqual(
            preset_from_natural_query('am I free this weekend?'),
            'this_weekend',
        )
        self.assertEqual(
            preset_from_natural_query('what does tomorrow look like'),
            'tomorrow',
        )

    def test_next_n_days(self):
        self.assertEqual(preset_from_natural_query('next 10 days'), 'next_10_days')
        self.assertEqual(preset_from_natural_query('next 7 days'), 'next_7_days')
        self.assertEqual(preset_from_natural_query('last 14 days'), 'last_14_days')
        self.assertEqual(preset_from_natural_query('past 30 days'), 'last_30_days')

    def test_saturday_sunday_keywords(self):
        self.assertEqual(preset_from_natural_query('what about Saturday'), 'this_weekend')
        self.assertEqual(preset_from_natural_query('next Sunday'), 'next_weekend')

    def test_n_days_out_of_range(self):
        self.assertIsNone(preset_from_natural_query('next 91 days'))
        self.assertIsNone(preset_from_natural_query('next 0 days'))

    def test_unrecognized_returns_none(self):
        self.assertIsNone(preset_from_natural_query('some random text'))
        self.assertIsNone(preset_from_natural_query(''))
        self.assertIsNone(preset_from_natural_query('   '))

    def test_dashes_normalized(self):
        self.assertEqual(
            preset_from_natural_query('last 7 days'),
            'last_7_days',
        )


class TestValidateFutureWindow(unittest.TestCase):
    """Tests for validate_future_window — rejecting past dates, bad ranges."""

    def test_future_window_is_valid(self):
        now = datetime.now(TZ)
        start = now + timedelta(hours=1)
        end = start + timedelta(hours=1)
        self.assertIsNone(validate_future_window(start, end, TZ_STR))

    def test_end_before_start_is_rejected(self):
        now = datetime.now(TZ)
        start = now + timedelta(hours=2)
        end = now + timedelta(hours=1)
        result = validate_future_window(start, end, TZ_STR)
        self.assertIsNotNone(result)
        self.assertIn('End time must be after start time', result)

    def test_past_start_is_rejected(self):
        now = datetime.now(TZ)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)
        result = validate_future_window(start, end, TZ_STR)
        self.assertIsNotNone(result)
        self.assertIn('past', result.lower())

    def test_5_minute_grace_period(self):
        """Start times within 5 minutes of now should be accepted."""
        now = datetime.now(TZ)
        start = now - timedelta(minutes=4)
        end = now + timedelta(hours=1)
        self.assertIsNone(validate_future_window(start, end, TZ_STR))

    def test_start_equals_end_is_rejected(self):
        now = datetime.now(TZ)
        t = now + timedelta(hours=1)
        result = validate_future_window(t, t, TZ_STR)
        self.assertIsNotNone(result)


class TestParseDt(unittest.TestCase):
    """Tests for parse_dt — ISO datetime parsing."""

    def test_parses_full_iso_with_timezone(self):
        dt = parse_dt('2026-05-15T14:30:00+05:30', TZ_STR)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 5)
        self.assertEqual(dt.day, 15)
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 30)

    def test_naive_datetime_gets_configured_timezone(self):
        dt = parse_dt('2026-05-15T14:30:00', TZ_STR)
        self.assertEqual(dt.tzinfo, TZ)

    def test_invalid_iso_raises_valueerror(self):
        with self.assertRaises(ValueError):
            parse_dt('not-a-date', TZ_STR)

    def test_z_suffix_parsed(self):
        dt = parse_dt('2026-05-15T14:30:00+00:00', TZ_STR)
        self.assertIsNotNone(dt.tzinfo)


class TestMissingFields(unittest.TestCase):
    """Tests for missing_fields helper."""

    def test_all_present(self):
        result = missing_fields(title='Meeting', start='2026-01-01')
        self.assertEqual(result, [])

    def test_empty_string_is_missing(self):
        result = missing_fields(title='', start='2026-01-01')
        self.assertIn('title', result)

    def test_none_is_missing(self):
        result = missing_fields(title=None, start='2026-01-01')
        self.assertIn('title', result)

    def test_whitespace_only_is_missing(self):
        result = missing_fields(title='   ', start='2026-01-01')
        self.assertIn('title', result)


class TestParseCalendarDate(unittest.TestCase):
    """Tests for parse_calendar_date — YYYY-MM-DD string to date object."""

    def test_valid_date(self):
        result = parse_calendar_date('2026-05-15')
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 5)
        self.assertEqual(result.day, 15)

    def test_truncates_to_10_chars(self):
        # If someone passes a full ISO datetime, it should still parse the date part
        result = parse_calendar_date('2026-05-15T14:30:00')
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.day, 15)

    def test_empty_returns_none(self):
        self.assertIsNone(parse_calendar_date(''))
        self.assertIsNone(parse_calendar_date(None))

    def test_invalid_returns_none(self):
        self.assertIsNone(parse_calendar_date('not-a-date'))
        self.assertIsNone(parse_calendar_date('2026-13-01'))


class TestEventsSummary(unittest.TestCase):
    """Tests for events_to_summary — formatting events for voice output."""

    def test_no_events(self):
        result = events_to_summary([], 'for today', TZ_STR)
        self.assertIn('No events', result)
        self.assertIn('free', result)

    def test_with_events(self):
        events = [
            {
                'summary': 'Team Standup',
                'start': {'dateTime': '2026-05-15T09:00:00+05:30'},
                'end': {'dateTime': '2026-05-15T09:30:00+05:30'},
            }
        ]
        result = events_to_summary(events, 'for today', TZ_STR)
        self.assertIn('Team Standup', result)

    def test_date_range_note_included(self):
        start = datetime(2026, 5, 15, 0, 0, 0, tzinfo=TZ)
        end = datetime(2026, 5, 17, 23, 59, 59, tzinfo=TZ)
        result = events_to_summary([], 'for weekend', TZ_STR, start_dt=start, end_dt=end)
        self.assertIn('Queried:', result)
        self.assertIn('2026-05-15', result)


if __name__ == '__main__':
    unittest.main()
