# LiveKit Voice Calendar Booking Assistant

Real-time voice agent built with LiveKit for multi-turn conversations, with Google Calendar tool-calling for availability checks and meeting booking.

## Features

- Real-time spoken interaction (`STT -> LLM -> TTS`)
- Multi-turn booking flow with missing-detail collection
- Calendar tool-calling loop:
  - `check_availability`
  - `suggest_time_slots`
  - `book_meeting`
- Booking confirmation with event details and link
- Provider switching for low-cost/local mode

## Project Structure

- `app/agent.py`: LiveKit voice agent + calendar tools + provider switching
- `app/calendar_service.py`: Google Calendar API wrapper
- `app/config.py`: environment configuration
- `streamlit_app/app.py`: setup/launch helper UI

## Zero-Credit Local Mode

This project supports a fully local stack:

- LLM: LM Studio (`LOCAL_LLM_BASE_URL`, model `LLM_MODEL`)
- STT: local OpenAI-compatible transcription server (`LOCAL_STT_BASE_URL`)
- TTS: local OpenAI-compatible speech server (`LOCAL_TTS_BASE_URL`)

Default `.env.example` is already set to:

- `LLM_PROVIDER=local`
- `STT_PROVIDER=local`
- `TTS_PROVIDER=local`

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure env:

```bash
cp .env.example .env
```

3. Start your local services:

- LM Studio server at `http://127.0.0.1:1234/v1`
- local STT server at `http://127.0.0.1:8001/v1` (OpenAI audio transcriptions compatible)
- local TTS server at `http://127.0.0.1:8002/v1` (OpenAI audio speech compatible)

4. Add Google service account JSON and set:

- `GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json`
- Share your Google Calendar with that service-account email.

5. Download LiveKit model assets and run:

```bash
uv run python app/agent.py download-files
uv run python app/agent.py console
```

6. Optional dashboard:

```bash
uv run streamlit run streamlit_app/app.py
```

## Cloud Providers (Optional)

If you want to switch back later:

- `LLM_PROVIDER=openai`
- `STT_PROVIDER=deepgram`
- `TTS_PROVIDER=openai`

Then set the relevant API keys.

## Voice Booking Flow

1. User asks to schedule a meeting.
2. Agent collects title/date/time/duration/attendee email.
3. Agent calls `check_availability`.
4. If busy, agent calls `suggest_time_slots`.
5. Agent asks for final confirmation.
6. Agent calls `book_meeting` and reads confirmation.
