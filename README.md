# LiveKit Voice-to-Calendar Agent

A low-latency conversational agent built on the LiveKit Python SDK that facilitates multi-turn natural language interactions for Google Calendar management. The system coordinates speech-to-text (STT), large language models (LLM), and text-to-speech (TTS) into a unified, real-time pipeline, integrated natively with Google Workspace APIs via function calling.

## Architecture & Data Flow

The system supports a decoupled provider model, allowing runtime swapping between cloud infrastructure (OpenAI, Deepgram) and zero-cost local execution (Whisper.cpp, Piper TTS, LM Studio) without business logic modification.

### Real-Time Voice Pipeline
1. **Audio Ingestion**: User audio is captured via WebRTC (browser) or local microphone (console) and routed to the STT provider.
2. **Transcription**: A persistent `whisper-server` process keeps the model hot in memory, eliminating the per-request model-load overhead of subprocess-based STT.
3. **LLM Synthesis & Tool Evaluation**: The transcribed text is sent to the LLM. The LLM evaluates if sufficient context exists to trigger a tool (e.g., `check_availability`, `reschedule_meeting`) or if a conversational response is required.
4. **Speech Synthesis**: Text output from the LLM is streamed to the TTS engine, rendered into PCM audio, and piped back to the user over WebRTC.

### Voice Booking State Machine
The agent relies on a multi-turn confirmation loop before mutating state on the remote Google Calendar:
1. **Intent Recognition**: Extracts required fields (attendee email, duration, requested time) from the audio stream.
2. **Conflict Resolution (`check_availability`)**: Queries the user's primary calendar for overlapping events.
3. **Slot Suggestion (`suggest_time_slots`)**: If a conflict exists, dynamically scans for the next available windows within the requested bounds.
4. **Finalization (`book_meeting`)**: Submits the verified payload to the Google Calendar API and dispatches the event.
5. **Rescheduling (`reschedule_meeting`)**: Finds an existing event by title, verifies the new time is free, and patches the event in place.

## Core Components

- `autopilot.py`: The primary system bootstrapper and process orchestrator. It manages virtual environment creation, dependency injection, and daemonizes local STT/TTS adapter web servers. It launches a persistent `whisper-server` process and conducts pre-flight health checks across all microservices before launching the agent.
- `app/agent.py`: Handles the LiveKit WebRTC session lifecycle, audio stream multiplexing, and tool-calling registration.
- `app/calendar_service.py`: Encapsulates Google Calendar API transactions and Service Account authorization.
- `app/config.py`: Environment management and configuration wrapper.
- `web/index.html`: Browser-based Web UI with live transcript and audio visualizer.
- `web/token_server.py`: Lightweight HTTP server that generates LiveKit JWTs and serves the Web UI.

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

### Console Mode (Headless)
The system is designed to be fully automated via the orchestrator.

```bash
# Starts whisper-server, local TTS/STT adapters, and boots the agent
python autopilot.py
```

### Web UI Mode (Browser)
The Web UI requires a running LiveKit server to broker WebRTC connections between the browser and the Python agent.

**Option A: Local LiveKit Server (Docker)**
```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
```

**Option B: LiveKit Cloud** — Sign up at [cloud.livekit.io](https://cloud.livekit.io) (free tier available).

Then set your `.env`:
```ini
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

Launch the agent in server mode (not console), then open `http://localhost:8100`:
```bash
python autopilot.py
```
