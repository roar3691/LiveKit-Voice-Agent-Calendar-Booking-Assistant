# LiveKit Voice-to-Calendar Agent

A low-latency conversational agent built on the LiveKit Python SDK that facilitates multi-turn natural language interactions for Google Calendar management. The system coordinates speech-to-text (STT), large language models (LLM), and text-to-speech (TTS) into a unified, real-time pipeline, integrated natively with Google Workspace APIs via function calling.

## Architecture

The system supports a decoupled provider model, allowing runtime swapping between cloud infrastructure (OpenAI, Deepgram) and zero-cost local execution (Whisper.cpp, Piper TTS, LM Studio) without business logic modification.

### Voice Booking State Machine
The agent relies on a multi-turn confirmation loop before mutating state on the remote Google Calendar:
1. **Intent Recognition**: Extracts required fields (attendee email, duration, requested time) from the audio stream.
2. **Conflict Resolution (`check_availability`)**: Queries the user's primary calendar for overlapping events.
3. **Slot Suggestion (`suggest_time_slots`)**: If a conflict exists, dynamically scans for the next available windows within the requested bounds.
4. **Finalization (`book_meeting`)**: Submits the verified payload to the Google Calendar API and dispatches an invite.

## Core Components

- `app/agent.py`: Handles the LiveKit WebRTC session lifecycle, audio stream multiplexing, and tool-calling registration.
- `app/calendar_service.py`: Encapsulates Google Calendar API transactions and Service Account authorization.
- `app/config.py`: Environment management and configuration wrapper.
- `streamlit_app/app.py`: Diagnostics and testing interface.

## Prerequisites & Setup

### Dependency Management
This repository utilizes `uv` for deterministic dependency resolution.

```bash
uv sync
```

### Authorization
1. Generate a Service Account key in your Google Cloud Console with Google Calendar API permissions.
2. Save the artifact as `service_account.json` in the project root.
3. Grant calendar editor access to the service account's client email.

### Environment Configuration
```bash
cp .env.example .env
```

To run a fully local inference stack:
```ini
LLM_PROVIDER=local
STT_PROVIDER=local
TTS_PROVIDER=local
```
Ensure your local endpoints (e.g., LM Studio at `:1234`, Whisper at `:8001`, Piper at `:8002`) are compliant with the OpenAI API specification.

## Execution

Fetch the required STT/TTS model weights and initialize the worker:

```bash
uv run python app/agent.py download-files
uv run python app/agent.py console
```

To inspect telemetry or use the GUI harness:
```bash
uv run streamlit run streamlit_app/app.py
```
