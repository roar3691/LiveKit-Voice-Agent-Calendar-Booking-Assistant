import os
from dotenv import load_dotenv

load_dotenv('.env')

LIVEKIT_URL = os.getenv('LIVEKIT_URL', '')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', '')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', '')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4.1-mini')
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv('LLM_REQUEST_TIMEOUT_SECONDS', '180'))
LLM_MAX_COMPLETION_TOKENS = int(os.getenv('LLM_MAX_COMPLETION_TOKENS', '1024'))
STT_PROVIDER = os.getenv('STT_PROVIDER', 'deepgram').lower()
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai').lower()

# OpenAI-compatible local endpoints (LM Studio / local gateways)
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')
OPENAI_TTS_VOICE = os.getenv('OPENAI_TTS_VOICE', 'alloy')

LOCAL_LLM_BASE_URL = os.getenv('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:1234/v1')
LOCAL_LLM_API_KEY = os.getenv('LOCAL_LLM_API_KEY', 'lm-studio')

LOCAL_STT_BASE_URL = os.getenv('LOCAL_STT_BASE_URL', 'http://127.0.0.1:8001/v1')
LOCAL_STT_API_KEY = os.getenv('LOCAL_STT_API_KEY', 'local-stt')
LOCAL_STT_MODEL = os.getenv('LOCAL_STT_MODEL', 'whisper-1')

LOCAL_TTS_BASE_URL = os.getenv('LOCAL_TTS_BASE_URL', 'http://127.0.0.1:8002/v1')
LOCAL_TTS_API_KEY = os.getenv('LOCAL_TTS_API_KEY', 'local-tts')
LOCAL_TTS_MODEL = os.getenv('LOCAL_TTS_MODEL', 'kokoro')
LOCAL_TTS_VOICE = os.getenv('LOCAL_TTS_VOICE', 'af_bella')

GOOGLE_CALENDAR_ID = os.getenv('GOOGLE_CALENDAR_ID', 'primary')
GOOGLE_TIMEZONE = os.getenv('GOOGLE_TIMEZONE', 'Asia/Kolkata')
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', 'service_account.json')

MIN_SLOT_MINUTES = int(os.getenv('MIN_SLOT_MINUTES', '30'))
