import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv('.env')

st.set_page_config(page_title='LiveKit Calendar Agent', layout='wide')
st.title('LiveKit Voice Calendar Booking Assistant')
st.caption('Setup helper for local no-credit stack and run commands')

llm_provider = os.getenv('LLM_PROVIDER', 'local')
stt_provider = os.getenv('STT_PROVIDER', 'local')
tts_provider = os.getenv('TTS_PROVIDER', 'local')

required = ['GOOGLE_SERVICE_ACCOUNT_FILE']
if llm_provider == 'local':
    required.append('LOCAL_LLM_BASE_URL')
else:
    required.append('OPENAI_API_KEY')

if stt_provider == 'local':
    required.append('LOCAL_STT_BASE_URL')
else:
    required.append('DEEPGRAM_API_KEY')

if tts_provider == 'local':
    required.append('LOCAL_TTS_BASE_URL')
else:
    required.append('OPENAI_API_KEY')

st.subheader('Environment Check')
for key in required:
    value = os.getenv(key, '')
    ok = bool(value)
    st.write(f"- {key}: {'configured' if ok else 'missing'}")

st.subheader('Run Commands')
st.code('uv sync', language='bash')
st.code('uv run python app/agent.py download-files', language='bash')
st.code('uv run python app/agent.py console', language='bash')

st.subheader('What This Agent Does')
st.markdown(
    """
- Handles multi-turn spoken scheduling conversations.
- Checks Google Calendar availability before booking.
- Suggests alternate free slots if a time is busy.
- Confirms details before creating the calendar event.
"""
)
