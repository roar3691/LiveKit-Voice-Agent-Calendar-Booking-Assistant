#!/usr/bin/env python3
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request as URLRequest, urlopen

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / '.venv'
PYTHON_BIN = VENV_DIR / 'bin' / 'python'
PIP_BIN = VENV_DIR / 'bin' / 'pip'

DEFAULTS: Dict[str, str] = {
    'LLM_PROVIDER': 'local',
    'STT_PROVIDER': 'local',
    'TTS_PROVIDER': 'local',
    'LLM_MODEL': 'mlx-qwen3.5-4b-claude-4.6-opus-reasoning-distilled',
    'LOCAL_LLM_BASE_URL': 'http://127.0.0.1:1234/v1',
    'LOCAL_LLM_API_KEY': 'lm-studio',
    'LLM_REQUEST_TIMEOUT_SECONDS': '45',
    'LLM_MAX_COMPLETION_TOKENS': '512',
    'LOCAL_STT_BASE_URL': 'http://127.0.0.1:8001/v1',
    'LOCAL_STT_API_KEY': 'local-stt',
    'LOCAL_STT_MODEL': 'whisper-1',
    'LOCAL_TTS_BASE_URL': 'http://127.0.0.1:8002/v1',
    'LOCAL_TTS_API_KEY': 'local-tts',
    'LOCAL_TTS_MODEL': 'piper',
    'LOCAL_TTS_VOICE': 'en_US-lessac-medium',
    'GOOGLE_SERVICE_ACCOUNT_FILE': 'service_account.json',
    'GOOGLE_CALENDAR_ID': 'primary',
    'GOOGLE_TIMEZONE': 'Asia/Kolkata',
    'MIN_SLOT_MINUTES': '30',
    # whisper.cpp
    'WHISPERCPP_HOST': '127.0.0.1',
    'WHISPERCPP_PORT': '8080',
    'WHISPERCPP_MODEL_PATH': str(ROOT / 'models' / 'ggml-base.en.bin'),
    # piper
    'PIPER_MODEL_PATH': str(ROOT / 'models' / 'en_US-lessac-medium.onnx'),
}

PIP_BOOTSTRAP = ['--upgrade', 'pip', 'setuptools', 'wheel']
APP_PACKAGES = [
    'livekit-agents[openai,silero]>=1.0.0',
    'google-api-python-client>=2.170.0',
    'google-auth>=2.40.0',
    'python-dotenv>=1.1.0',
    'aiohttp>=3.10.0',
]

CHILDREN: List[subprocess.Popen] = []


@dataclass
class HealthCheck:
    name: str
    url: str
    timeout_sec: int


def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True, retries: int = 1):
    attempt = 0
    while True:
        attempt += 1
        print(f"[run] {' '.join(cmd)}")
        try:
            return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            time.sleep(2)


def resolve_python311() -> str:
    py = shutil.which('python3.11')
    if py:
        return py
    raise RuntimeError('python3.11 not found. Install Python 3.11 (brew install python@3.11) and rerun.')


def ensure_venv():
    if not VENV_DIR.exists():
        py311 = resolve_python311()
        print(f'[setup] creating Python 3.11 virtualenv with {py311}...')
        run([py311, '-m', 'venv', str(VENV_DIR)], check=True)


def pip_install():
    print('[setup] upgrading packaging tools...')
    run([str(PIP_BIN), 'install', *PIP_BOOTSTRAP], check=True, retries=2)
    print('[setup] installing project dependencies...')
    run([str(PIP_BIN), 'install', *APP_PACKAGES], check=True, retries=2)


def ensure_env_file():
    env_path = ROOT / '.env'
    if env_path.exists():
        # Preserve existing .env content entirely; only append missing keys
        existing_text = env_path.read_text()
        existing_keys: set = set()
        for raw in existing_text.splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _ = line.split('=', 1)
            existing_keys.add(k.strip())

        missing_lines = []
        for k, v in DEFAULTS.items():
            if k not in existing_keys:
                missing_lines.append(f'{k}={v}')

        if missing_lines:
            with open(env_path, 'a') as f:
                f.write('\n# Auto-appended defaults\n')
                f.write('\n'.join(missing_lines) + '\n')
            print(f'[setup] appended {len(missing_lines)} missing keys to {env_path}')
        else:
            print(f'[setup] env config is up to date: {env_path}')
    else:
        # No .env exists; create one from defaults
        env_path.write_text('\n'.join(f'{k}={v}' for k, v in DEFAULTS.items()) + '\n')
        print(f'[setup] created env config: {env_path}')


def http_ok(url: str, timeout_sec: int = 2) -> bool:
    req = URLRequest(url, method='GET')
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def service_ready(name: str, url: str) -> bool:
    if http_ok(url):
        print(f'[ok] reusing running {name} at {url}')
        return True
    return False


def wait_health(check: HealthCheck):
    print(f'[wait] waiting for {check.name} at {check.url}...')
    start = time.time()
    while time.time() - start < check.timeout_sec:
        if http_ok(check.url):
            print(f'[ok] {check.name} is ready')
            return
        time.sleep(1)
    raise RuntimeError(f'{check.name} did not become ready: {check.url}')


def start_child(cmd: List[str], env: Optional[Dict[str, str]] = None):
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env or os.environ.copy())
    CHILDREN.append(proc)
    return proc


def cleanup_children(*_):
    for p in CHILDREN:
        if p.poll() is None:
            p.terminate()
    for p in CHILDREN:
        if p.poll() is None:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()


def check_lmstudio():
    if not http_ok('http://127.0.0.1:1234/v1/models'):
        raise RuntimeError(
            'LM Studio server is not reachable at http://127.0.0.1:1234/v1. '
            'Open LM Studio, load your model, and click Start Server.'
        )


def ensure_whispercpp_cli():
    if shutil.which('whisper-cli'):
        return 'whisper-cli'
    raise RuntimeError('whisper-cli not found. Install whisper.cpp (e.g., brew install whisper-cpp).')


def ensure_piper_binary():
    if shutil.which('piper'):
        return 'piper'
    local_piper = VENV_DIR / 'bin' / 'piper'
    if local_piper.exists():
        return str(local_piper)
    raise RuntimeError('piper binary not found. Install Piper (e.g., pip install piper-tts, or system piper binary).')


def ensure_file(path: str, purpose: str):
    if not Path(path).exists():
        raise RuntimeError(f'{purpose} missing at {path}')


def ensure_piper_model_files(model_path: str):
    ensure_file(model_path, 'Piper model')
    ensure_file(f'{model_path}.json', 'Piper model config')


def run_stt_adapter():
    # OpenAI-compatible STT adapter -> whisper.cpp CLI.
    from aiohttp import web

    whisper_cli = ensure_whispercpp_cli()
    model_path = os.getenv('WHISPERCPP_MODEL_PATH', DEFAULTS['WHISPERCPP_MODEL_PATH'])
    ensure_file(model_path, 'whisper.cpp model')

    async def health(_request: web.Request):
        return web.json_response({'ok': True, 'backend': 'whisper.cpp'})

    def clean_transcript(text: str) -> str:
        cleaned = ' '.join(text.strip().split())
        lowered = cleaned.lower().strip(' .!?')
        noise_phrases = {
            '',
            'blank audio',
            'music',
            'upbeat music',
            'background music',
            'crowd chattering',
            'applause',
            'silence',
            'thank you for watching',
            'thanks for watching',
            'i will see you in the next video',
            'see you in the next video',
        }
        if lowered in noise_phrases:
            return ''
        if (cleaned.startswith('[') and cleaned.endswith(']')) or (
            cleaned.startswith('(') and cleaned.endswith(')')
        ):
            return ''
        return cleaned

    async def transcriptions(request: web.Request):
        reader = await request.multipart()
        file_data = None
        language = None
        response_format = 'json'
        filename = 'audio.bin'

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == 'file':
                filename = part.filename or 'audio.bin'
                file_data = await part.read(decode=False)
            elif part.name == 'language':
                language = (await part.text()).strip() or None
            elif part.name == 'response_format':
                response_format = (await part.text()).strip() or 'json'

        if file_data is None:
            return web.json_response({'error': 'missing file'}, status=400)

        suffix = Path(filename).suffix or '.bin'
        data = file_data

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as norm_tmp:
            norm_wav_path = norm_tmp.name

        try:
            # If this is already a wav container, use it.
            # Otherwise treat payload as raw PCM16 mono and wrap into WAV.
            if data[:4] == b'RIFF':
                with open(norm_wav_path, 'wb') as out_f:
                    out_f.write(data)
            else:
                pcm = data
                if len(pcm) % 2 != 0:
                    pcm += b'\x00'
                with wave.open(norm_wav_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # PCM16
                    wf.setframerate(16000)
                    wf.writeframes(pcm)

            cmd = [
                whisper_cli,
                '--model', model_path,
                '--file', norm_wav_path,
                '--language', language or 'en',
                '--no-timestamps',
                '--no-prints',
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=180,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or 'whisper-cli failed').strip()
                return web.json_response({'error': err}, status=500)

            text = clean_transcript(proc.stdout)
            if response_format == 'text':
                return web.Response(text=text, content_type='text/plain')
            return web.json_response({'text': text})
        except Exception as exc:
            print(f'[stt-error] {exc}')
            return web.json_response({'error': str(exc)}, status=500)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            try:
                os.remove(norm_wav_path)
            except OSError:
                pass

    app = web.Application()
    app.add_routes([
        web.get('/health', health),
        web.post('/v1/audio/transcriptions', transcriptions),
    ])
    web.run_app(app, host='127.0.0.1', port=8001)


def run_tts_adapter():
    # OpenAI-compatible TTS adapter -> Piper CLI
    from aiohttp import web

    piper_bin = ensure_piper_binary()
    model_path = os.getenv('PIPER_MODEL_PATH', DEFAULTS['PIPER_MODEL_PATH'])
    ensure_piper_model_files(model_path)

    async def health(_request: web.Request):
        return web.json_response({'ok': True, 'backend': 'piper'})

    async def speech(request: web.Request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'invalid json'}, status=400)

        text = (body.get('input') or '').strip()
        if not text:
            return web.json_response({'error': 'missing input'}, status=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            out_path = tmp.name

        try:
            proc = subprocess.run(
                [piper_bin, '--model', model_path, '--output_file', out_path],
                input=text.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                return web.json_response({'error': proc.stderr.decode('utf-8', errors='ignore')}, status=500)

            audio = Path(out_path).read_bytes()
            return web.Response(body=audio, content_type='audio/wav')
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

    app = web.Application()
    app.add_routes([
        web.get('/health', health),
        web.post('/v1/audio/speech', speech),
    ])
    web.run_app(app, host='127.0.0.1', port=8002)


def run_main():
    ensure_venv()
    pip_install()
    ensure_env_file()

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    whisper_cli = ensure_whispercpp_cli()
    whisper_model = os.getenv('WHISPERCPP_MODEL_PATH', DEFAULTS['WHISPERCPP_MODEL_PATH'])
    piper_model = os.getenv('PIPER_MODEL_PATH', DEFAULTS['PIPER_MODEL_PATH'])
    ensure_file(whisper_model, 'whisper.cpp model')
    ensure_piper_model_files(piper_model)

    print(f'[ok] whisper.cpp CLI is ready: {whisper_cli}')

    # Start adapters, or reuse already-running copies from a previous run.
    if not service_ready('local STT adapter', 'http://127.0.0.1:8001/health'):
        start_child([str(PYTHON_BIN), str(ROOT / 'autopilot.py'), '--serve-stt-adapter'], env=env)
    if not service_ready('local TTS adapter', 'http://127.0.0.1:8002/health'):
        start_child([str(PYTHON_BIN), str(ROOT / 'autopilot.py'), '--serve-tts-adapter'], env=env)

    wait_health(HealthCheck('local STT adapter', 'http://127.0.0.1:8001/health', 120))
    wait_health(HealthCheck('local TTS adapter', 'http://127.0.0.1:8002/health', 120))

    check_lmstudio()

    run([str(PYTHON_BIN), '-m', 'app.agent', 'download-files'], cwd=ROOT, check=True, retries=2)
    print('[start] launching LiveKit calendar voice agent...')
    run([str(PYTHON_BIN), '-m', 'app.agent', 'console'], cwd=ROOT, check=True)


def main():
    atexit.register(cleanup_children)
    signal.signal(signal.SIGINT, cleanup_children)
    signal.signal(signal.SIGTERM, cleanup_children)

    if '--serve-stt-adapter' in sys.argv:
        run_stt_adapter()
        return

    if '--serve-tts-adapter' in sys.argv:
        run_tts_adapter()
        return

    try:
        run_main()
    except Exception as exc:
        print(f'[error] {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()
