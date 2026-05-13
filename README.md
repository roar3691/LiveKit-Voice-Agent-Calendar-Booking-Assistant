# LiveKit Voice Agent — Calendar Booking Assistant

A production-grade, low-latency voice assistant built on the [LiveKit Agents SDK](https://docs.livekit.io/agents/) that manages Google Calendar through natural, multi-turn conversations. The system coordinates Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) into a unified real-time pipeline, with native Google Calendar integration via LLM function calling.

> **Run it entirely on your machine** — with local Whisper STT, Piper TTS, and any OpenAI-compatible LLM (e.g. LM Studio), or swap in cloud providers (OpenAI, Deepgram) with a single config change.

---

## ✨ Key Features

- **Natural language scheduling** — "Am I free this weekend?", "Book a standup at 10 AM tomorrow", "Block my Friday"
- **Smart date understanding** — handles relative expressions like "this weekend", "next 10 days", "last month" without asking for ISO dates
- **Multi-turn conversation** — confirms details, resolves conflicts, suggests alternatives before committing
- **All-day & multi-day events** — "Mark my weekend as busy", "Block next Monday through Wednesday"
- **Reschedule & cancel** — modify or remove events by title within the same session
- **Dual voice support** — primary (lessac-high) and secondary (amy-medium) Piper TTS voices
- **Optional neural TTS** — Qwen3-TTS backend via MLX for high-fidelity synthesis on Apple Silicon
- **Web UI** — browser-based interface with live transcript and audio visualizer over WebRTC
- **Console mode** — headless mode using local microphone/speaker, no LiveKit server required
- **Zero-cost local stack** — Whisper.cpp + Piper + LM Studio for fully offline operation

---

## 🏗️ Architecture

```
┌─────────────┐    WebRTC / Mic     ┌──────────────────────────────────────────────────┐
│  Browser UI │ ◄─────────────────► │              LiveKit Agent Server                │
│  (index.html│    or Console       │                                                  │
│   + Token   │                     │  ┌──────────┐  ┌───────────┐  ┌──────────────┐   │
│    Server)  │                     │  │  Silero   │  │  LLM      │  │    TTS       │   │
└─────────────┘                     │  │  VAD      │  │ (OpenAI / │  │ (Piper /     │   │
                                    │  └────┬─────┘  │ LM Studio)│  │  Qwen3 /     │   │
                                    │       │        └─────┬─────┘  │  OpenAI)     │   │
                                    │       ▼              │        └──────┬───────┘   │
                                    │  ┌──────────┐        │               │           │
                                    │  │  STT     │   Function Calls       │           │
                                    │  │(Whisper/ │   (tool results)       │           │
                                    │  │Deepgram) │        │               │           │
                                    │  └──────────┘        ▼               │           │
                                    │              ┌───────────────┐       │           │
                                    │              │ Calendar      │       │           │
                                    │              │ Booking       │◄──────┘           │
                                    │              │ Assistant     │                   │
                                    │              │ (agent.py)    │                   │
                                    │              └───────┬───────┘                   │
                                    └──────────────────────┼───────────────────────────┘
                                                           │
                                                    Google Calendar API
                                                    (Service Account)
```

### Voice Pipeline Flow

1. **Audio Ingestion** — User audio captured via WebRTC (browser) or local microphone (console), processed through Silero Voice Activity Detection (VAD)
2. **Transcription** — Audio routed to the STT provider (persistent `whisper-server` keeps the model hot in memory, eliminating per-request load overhead)
3. **LLM Reasoning & Tool Calling** — Transcribed text sent to the LLM, which decides whether to call a calendar tool or respond conversationally
4. **Speech Synthesis** — LLM text output is stripped of markdown/emojis and streamed to the TTS engine, rendered to PCM audio, and piped back to the user

### Booking State Machine

1. **Intent Recognition** — Extract fields (title, time, duration, attendee) from the conversation
2. **Conflict Resolution** — Query the calendar for overlapping events via `check_availability`
3. **Slot Suggestion** — If busy, scan for available windows via `suggest_time_slots`
4. **Confirmation** — Summarize proposed booking and wait for user approval
5. **Finalization** — Submit to Google Calendar API via `book_meeting` or `book_all_day_event`

---

## 📁 Project Structure

```
livekit-calendar-agent/
├── app/
│   ├── agent.py              # LiveKit session lifecycle, tool definitions, CalendarBookingAssistant
│   ├── calendar_service.py   # Google Calendar API wrapper (Service Account auth)
│   ├── config.py             # Environment variables and configuration
│   └── date_utils.py         # Date parsing, preset windows, natural language → date range
├── tests/
│   └── test_date_utils.py    # Comprehensive test suite for date utilities (430+ lines)
├── web/
│   ├── index.html            # Browser UI with live transcript and audio visualizer
│   └── token_server.py       # HTTP server for LiveKit JWT generation and static serving
├── models/                   # Local model files (whisper, piper) — gitignored
├── autopilot.py              # System orchestrator: venv, adapters, health checks, launch
├── pyproject.toml            # Project metadata and dependencies
├── .env.example              # Template environment configuration
└── README.md
```

---

## 🛠️ Tool Reference

The agent exposes the following function-calling tools to the LLM:

| Tool | Purpose | Key Arguments |
|------|---------|---------------|
| `smart_calendar_lookup` | Natural language schedule queries | `query` (e.g. "this weekend", "next 10 days") |
| `calendar_date_range_lookup` | Lookup events across specific date range | `start_date`, `end_date` (YYYY-MM-DD) |
| `check_availability` | Check if a specific time window is free | `requested_start_iso`, `requested_end_iso` |
| `suggest_time_slots` | Find free slots in a window | `window_start_iso`, `window_end_iso`, `meeting_minutes` |
| `book_meeting` | Create a timed calendar event | `title`, `start_iso`, `end_iso`, `attendee_email` (optional) |
| `book_all_day_event` | Create an all-day/multi-day event | `title`, `start_date`, `end_date` |
| `reschedule_meeting` | Find and move an existing event | `search_query`, `search_window_*`, `new_start_iso`, `new_end_iso` |
| `cancel_last_event` | Cancel the most recently booked event | (none) |
| `rename_last_event` | Rename the most recently booked event | `new_title` |

### Supported Natural Language Time References

`smart_calendar_lookup` understands the following phrases (and variations):

| Phrase | Resolved Window |
|--------|----------------|
| today, tomorrow | Single day |
| this weekend, next weekend | Saturday–Sunday |
| this week, this month, this year | Full period |
| last month, next month | Full calendar month |
| next 7/10/14/30 days | Rolling forward window |
| last 7/14/30 days | Rolling backward window |
| Saturday, Sunday | Maps to this/next weekend |

---

## 🚀 Prerequisites & Setup

### 1. System Requirements

- **Python 3.11** (required for `autopilot.py` venv creation)
- **whisper.cpp** — `brew install whisper-cpp` (provides the `whisper-server` binary)
- **Piper TTS** — `pip install piper-tts` or system binary
- **LM Studio** — Any OpenAI-compatible local LLM server at `http://127.0.0.1:1234`

### 2. Google Calendar Authorization

1. Create a **Service Account** in the [Google Cloud Console](https://console.cloud.google.com/) with the **Google Calendar API** enabled
2. Download the JSON key and save it as `service_account.json` in the project root
3. Share your calendar with the service account's email address (grant **Editor** access)

### 3. Install Dependencies

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
```

Or let the `autopilot.py` orchestrator handle everything automatically (it creates a venv, installs deps, and launches all services).

### 4. Download Models

Place the following model files in the `models/` directory:

| Model | File | Source |
|-------|------|--------|
| Whisper (base.en) | `ggml-base.en.bin` | [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp) |
| Piper (lessac-high) | `en_US-lessac-high.onnx` + `.onnx.json` | [Piper voices](https://github.com/rhasspy/piper/blob/master/VOICES.md) |
| Piper (amy-medium) | `en_US-amy-medium.onnx` + `.onnx.json` | Optional second voice |

### 5. Environment Configuration

```bash
cp .env.example .env
```

Key variables:

```ini
# Provider selection (local or cloud)
LLM_PROVIDER=local
STT_PROVIDER=local
TTS_PROVIDER=local

# LLM configuration
LLM_MODEL=mlx-qwen3.5-4b-claude-4.6-opus-reasoning-distilled
LLM_REQUEST_TIMEOUT_SECONDS=45
LLM_MAX_COMPLETION_TOKENS=1024

# Local endpoints
LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_STT_BASE_URL=http://127.0.0.1:8001/v1
LOCAL_TTS_BASE_URL=http://127.0.0.1:8002/v1

# TTS backend: 'piper' (lightweight) or 'qwen3' (neural, Apple Silicon)
TTS_BACKEND=piper

# Google Calendar
GOOGLE_CALENDAR_ID=primary
GOOGLE_TIMEZONE=Asia/Kolkata
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json

# LiveKit (required for Web UI mode)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

---

## ▶️ Running the Agent

### One-Command Launch (Recommended)

The orchestrator handles everything — venv setup, adapter startup, health checks, and agent launch:

```bash
python autopilot.py
```

What `autopilot.py` does:
1. Creates/reuses a Python 3.11 virtualenv
2. Installs all dependencies
3. Ensures `.env` is populated with defaults
4. Starts the persistent `whisper-server` (model stays hot in memory)
5. Starts the OpenAI-compatible STT adapter (port 8001)
6. Starts the TTS adapter — Piper or Qwen3 (port 8002)
7. Verifies LM Studio is reachable
8. Optionally starts the Web UI token server (port 8100)
9. Launches the LiveKit agent in **dev** mode (if LiveKit URL is configured) or **console** mode

### Console Mode (Headless)

Uses your local microphone and speakers directly — no LiveKit server needed:

```bash
# Ensure LIVEKIT_URL is empty or set to ws://localhost:7880 (default)
python autopilot.py
```

### Web UI Mode (Browser)

Requires a running LiveKit server to broker WebRTC connections:

**Option A — Local LiveKit Server (Docker):**
```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
```

**Option B — LiveKit Cloud:**
Sign up at [cloud.livekit.io](https://cloud.livekit.io) (free tier available) and set your credentials in `.env`.

Then configure `.env`:
```ini
LIVEKIT_URL=ws://localhost:7880   # or your cloud URL
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

Launch and open `http://localhost:8100`:
```bash
python autopilot.py
```

---

## 🧪 Testing

The project includes a comprehensive test suite for the date utility module:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test
uv run pytest tests/test_date_utils.py::TestPresetFromNaturalQuery -v
```

Test coverage includes:
- ISO datetime parsing (naive and timezone-aware)
- Date validation (future enforcement, end-after-start)
- All preset time windows (today, tomorrow, weekends, months, rolling windows)
- Natural language → preset token mapping
- Weekend boundary calculations (weekday, Saturday, Sunday)
- Event summary formatting
- Edge cases (year boundaries, December → January rollover)

---

## 🔌 Provider Swapping

The system supports runtime provider swapping without code changes:

| Component | Local (Zero-Cost) | Cloud |
|-----------|-------------------|-------|
| **STT** | Whisper.cpp via adapter (port 8001) | Deepgram Nova-2 |
| **LLM** | LM Studio / any OpenAI-compatible server | OpenAI GPT-4o |
| **TTS** | Piper (lightweight) or Qwen3-TTS (neural, MLX) | OpenAI TTS |
| **VAD** | Silero (always local) | Silero (always local) |

Switch providers by changing `LLM_PROVIDER`, `STT_PROVIDER`, and `TTS_PROVIDER` in `.env` to `local` or their cloud equivalents (`openai`, `deepgram`).

---

## 📝 License

This project is for educational and personal use.
