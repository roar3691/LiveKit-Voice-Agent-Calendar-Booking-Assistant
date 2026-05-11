from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from google.oauth2 import service_account
from googleapiclient.discovery import build

CALENDAR_SCOPE = ['https://www.googleapis.com/auth/calendar']


@dataclass
class TimeSlot:
    start: datetime
    end: datetime


class GoogleCalendarService:
    def __init__(self, service_account_file: str, calendar_id: str, timezone: str):
        self.calendar_id = calendar_id
        self.timezone = timezone
        creds = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=CALENDAR_SCOPE,
        )
        self.service_account_email = creds.service_account_email
        self.client = build('calendar', 'v3', credentials=creds)

    def _to_rfc3339(self, dt: datetime) -> str:
        return dt.isoformat()

    def list_events(self, window_start: datetime, window_end: datetime):
        return self.client.events().list(
            calendarId=self.calendar_id,
            timeMin=self._to_rfc3339(window_start),
            timeMax=self._to_rfc3339(window_end),
            singleEvents=True,
            orderBy='startTime',
            timeZone=self.timezone,
        ).execute()

    def get_event(self, event_id: str):
        return self.client.events().get(calendarId=self.calendar_id, eventId=event_id).execute()

    def update_event_summary(self, event_id: str, summary: str):
        return self.client.events().patch(
            calendarId=self.calendar_id,
            eventId=event_id,
            body={'summary': summary},
        ).execute()

    def delete_event(self, event_id: str):
        return self.client.events().delete(
            calendarId=self.calendar_id,
            eventId=event_id,
            sendUpdates='none',
        ).execute()

    def find_events(self, window_start: datetime, window_end: datetime, query: str | None = None):
        request = self.client.events().list(
            calendarId=self.calendar_id,
            timeMin=self._to_rfc3339(window_start),
            timeMax=self._to_rfc3339(window_end),
            singleEvents=True,
            orderBy='startTime',
            timeZone=self.timezone,
            q=query or None,
        )
        return request.execute().get('items', [])

    def is_free(self, start: datetime, end: datetime) -> bool:
        events = self.list_events(start, end)
        return len(events.get('items', [])) == 0

    def find_free_slots(
        self,
        window_start: datetime,
        window_end: datetime,
        meeting_minutes: int,
        limit: int = 5,
    ) -> List[TimeSlot]:
        events = self.list_events(window_start, window_end).get('items', [])
        busy_slots = []
        for e in events:
            start = e.get('start', {}).get('dateTime')
            end = e.get('end', {}).get('dateTime')
            if not start or not end:
                continue
            busy_slots.append((datetime.fromisoformat(start), datetime.fromisoformat(end)))

        busy_slots.sort(key=lambda x: x[0])
        free_slots: List[TimeSlot] = []
        cursor = window_start
        min_delta = timedelta(minutes=meeting_minutes)

        for busy_start, busy_end in busy_slots:
            if busy_start > cursor and (busy_start - cursor) >= min_delta:
                free_slots.append(TimeSlot(start=cursor, end=busy_start))
                if len(free_slots) >= limit:
                    return free_slots
            if busy_end > cursor:
                cursor = busy_end

        if window_end > cursor and (window_end - cursor) >= min_delta:
            free_slots.append(TimeSlot(start=cursor, end=window_end))

        return free_slots[:limit]

    def create_event(self, summary: str, description: str, start: datetime, end: datetime, attendee_email: str):
        body = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': self._to_rfc3339(start), 'timeZone': self.timezone},
            'end': {'dateTime': self._to_rfc3339(end), 'timeZone': self.timezone},
            'attendees': [{'email': attendee_email}] if attendee_email else [],
        }
        send_updates = 'all' if attendee_email else 'none'
        return self.client.events().insert(
            calendarId=self.calendar_id,
            body=body,
            sendUpdates=send_updates,
        ).execute()
