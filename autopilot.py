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
    'LLM_MAX_COMPLETION_TOKENS': '1024',
    'LOCAL_STT_BASE_URL': 'http://127.0.0.1:8001/v1',
    'LOCAL_STT_API_KEY': 'local-stt',
    'LOCAL_STT_MODEL': 'whisper-1',
    'LOCAL_TTS_BASE_URL': 'http://127.0.0.1:8002/v1',
    'LOCAL_TTS_API_KEY': 'local-tts',
    'LOCAL_TTS_MODEL': 'piper',
    'LOCAL_TTS_VOICE': 'en_US-lessac-high',
    'GOOGLE_SERVICE_ACCOUNT_FILE': 'service_account.json',
    'GOOGLE_CALENDAR_ID': 'primary',
    'GOOGLE_TIMEZONE': 'Asia/Kolkata',
    'MIN_SLOT_MINUTES': '30',
    # whisper.cpp
    'WHISPERCPP_HOST': '127.0.0.1',
    'WHISPERCPP_PORT': '8080',
    'WHISPERCPP_MODEL_PATH': str(ROOT / 'models' / 'ggml-base.en.bin'),
    # piper
    'PIPER_MODEL_PATH': str(ROOT / 'models' / 'en_US-lessac-high.onnx'),
    # tts backend: 'qwen3' (neural, high quality) or 'piper' (lightweight)
    'TTS_BACKEND': 'qwen3',
    'QWEN3_TTS_MODEL': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit',
}

PIP_BOOTSTRAP = ['--upgrade', 'pip', 'setuptools', 'wheel']
APP_PACKAGES = [
    'livekit-agents[openai,silero]>=1.0.0',
    'google-api-python-client>=2.170.0',
    'google-auth>=2.40.0',
    'python-dotenv>=1.1.0',
    'aiohttp>=3.10.0',
    'livekit-api>=1.0.0',
]

WHISPER_SERVER_PORT = 8178  # Internal port for persistent whisper-server

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
            return resp.status < 500
    except Exception as exc:
        # Some servers return non-2xx pages but are still alive (e.g. whisper-server root)
        exc_str = str(exc)
        if '405' in exc_str or '404' in exc_str:
            return True
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


def ensure_whispercpp_server():
    if shutil.which('whisper-server'):
        return 'whisper-server'
    raise RuntimeError('whisper-server not found. Install whisper.cpp (e.g., brew install whisper-cpp).')


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
    # OpenAI-compatible STT adapter that proxies to a persistent whisper-server.
    import aiohttp as aiohttp_client
    from aiohttp import web

    whisper_inference_url = f'http://127.0.0.1:{WHISPER_SERVER_PORT}/inference'

    async def health(_request: web.Request):
        return web.json_response({'ok': True, 'backend': 'whisper-server (persistent)'})

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

        data = file_data

        # Normalize audio to WAV if it isn't already
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as norm_tmp:
            norm_wav_path = norm_tmp.name

        try:
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

            # Proxy to persistent whisper-server
            form = aiohttp_client.FormData()
            form.add_field('file', open(norm_wav_path, 'rb'),
                           filename='audio.wav', content_type='audio/wav')
            form.add_field('temperature', '0.0')
            form.add_field('response_format', 'json')
            if language:
                form.add_field('language', language)

            async with aiohttp_client.ClientSession() as session:
                async with session.post(whisper_inference_url, data=form, timeout=aiohttp_client.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        return web.json_response({'error': f'whisper-server error: {err}'}, status=500)
                    result = await resp.json()

            raw_text = result.get('text', '')
            text = clean_transcript(raw_text)

            if response_format == 'text':
                return web.Response(text=text, content_type='text/plain')
            return web.json_response({'text': text})
        except Exception as exc:
            print(f'[stt-error] {exc}')
            return web.json_response({'error': str(exc)}, status=500)
        finally:
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
    """OpenAI-compatible TTS adapter with Qwen3-TTS (MLX) or Piper backend."""
    from aiohttp import web

    tts_backend = os.getenv('TTS_BACKEND', 'qwen3').lower()

    if tts_backend == 'qwen3':
        return _run_qwen3_tts_adapter()
    else:
        return _run_piper_tts_adapter()


def _run_qwen3_tts_adapter():
    """Qwen3-TTS via mlx-audio — high quality neural TTS on Apple Silicon."""
    from aiohttp import web
    import io

    qwen3_model_id = os.getenv(
        'QWEN3_TTS_MODEL', 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit'
    )

    print(f'[tts] loading Qwen3-TTS model: {qwen3_model_id} ...')
    from mlx_audio.tts.utils import load_model
    model = load_model(qwen3_model_id)
    print(f'[tts] Qwen3-TTS model loaded successfully')

    async def health(_request: web.Request):
        return web.json_response({
            'ok': True, 'backend': 'qwen3-tts', 'model': qwen3_model_id
        })

    async def speech(request: web.Request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'invalid json'}, status=400)

        text = (body.get('input') or '').strip()
        if not text:
            return web.json_response({'error': 'missing input'}, status=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='qwen3tts_') as tmp:
            out_prefix = tmp.name.replace('.wav', '')

        try:
            from mlx_audio.tts.generate import generate_audio
            import numpy as np

            # Generate audio (saves to file)
            audio_result = generate_audio(
                model=model,
                text=text,
                file_prefix=out_prefix,
                verbose=False,
            )

            # Find the generated wav file
            out_path = out_prefix + '.wav'
            if not Path(out_path).exists():
                # Try finding any generated file
                import glob
                candidates = glob.glob(out_prefix + '*')
                if candidates:
                    out_path = candidates[0]
                else:
                    return web.json_response(
                        {'error': 'Qwen3-TTS did not produce audio output'}, status=500
                    )

            audio_bytes = Path(out_path).read_bytes()
            return web.Response(body=audio_bytes, content_type='audio/wav')
        except Exception as exc:
            print(f'[tts] Qwen3-TTS error: {exc}')
            return web.json_response({'error': str(exc)}, status=500)
        finally:
            # Clean up temp files
            import glob
            for f in glob.glob(out_prefix + '*'):
                try:
                    os.remove(f)
                except OSError:
                    pass

    app = web.Application()
    app.add_routes([
        web.get('/health', health),
        web.post('/v1/audio/speech', speech),
    ])
    web.run_app(app, host='127.0.0.1', port=8002)


def _run_piper_tts_adapter():
    """Piper TTS — lightweight local TTS with dual voice support."""
    from aiohttp import web

    piper_bin = ensure_piper_binary()

    # Primary voice: lessac-high (warm, professional)
    primary_model = os.getenv('PIPER_MODEL_PATH', DEFAULTS['PIPER_MODEL_PATH'])
    ensure_piper_model_files(primary_model)

    # Secondary voice: amy-medium (clear, friendly)
    secondary_model = str(ROOT / 'models' / 'en_US-amy-medium.onnx')
    if Path(secondary_model).exists() and Path(secondary_model + '.json').exists():
        print(f'[tts] dual voice enabled: lessac-high (primary) + amy-medium (secondary)')
    else:
        secondary_model = None
        print(f'[tts] single voice mode: {Path(primary_model).stem}')

    # Map voice names to model paths
    voice_map = {
        'en_US-lessac-high': primary_model,
        'lessac': primary_model,
        'default': primary_model,
    }
    if secondary_model:
        voice_map['en_US-amy-medium'] = secondary_model
        voice_map['amy'] = secondary_model

    async def health(_request: web.Request):
        voices = list(voice_map.keys())
        return web.json_response({'ok': True, 'backend': 'piper', 'voices': voices})

    async def speech(request: web.Request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({'error': 'invalid json'}, status=400)

        text = (body.get('input') or '').strip()
        if not text:
            return web.json_response({'error': 'missing input'}, status=400)

        # Select voice model based on request (OpenAI API sends 'voice' field)
        requested_voice = (body.get('voice') or 'default').strip()
        model = voice_map.get(requested_voice, primary_model)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            out_path = tmp.name

        try:
            proc = subprocess.run(
                [piper_bin, '--model', model, '--output_file', out_path],
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

    whisper_server_bin = ensure_whispercpp_server()
    whisper_model = os.getenv('WHISPERCPP_MODEL_PATH', DEFAULTS['WHISPERCPP_MODEL_PATH'])
    piper_model = os.getenv('PIPER_MODEL_PATH', DEFAULTS['PIPER_MODEL_PATH'])
    ensure_file(whisper_model, 'whisper.cpp model')
    ensure_piper_model_files(piper_model)

    # Launch persistent whisper-server (model stays hot in memory)
    whisper_health_url = f'http://127.0.0.1:{WHISPER_SERVER_PORT}/'
    if not service_ready('whisper-server', whisper_health_url):
        print(f'[start] launching persistent whisper-server on port {WHISPER_SERVER_PORT}...')
        start_child([
            whisper_server_bin,
            '--model', whisper_model,
            '--host', '127.0.0.1',
            '--port', str(WHISPER_SERVER_PORT),
            '--no-timestamps',
            '--language', 'en',
        ], env=env)
        wait_health(HealthCheck('whisper-server', whisper_health_url, 30))

    # Start adapters, or reuse already-running copies from a previous run.
    if not service_ready('local STT adapter', 'http://127.0.0.1:8001/health'):
        start_child([str(PYTHON_BIN), str(ROOT / 'autopilot.py'), '--serve-stt-adapter'], env=env)
    if not service_ready('local TTS adapter', 'http://127.0.0.1:8002/health'):
        start_child([str(PYTHON_BIN), str(ROOT / 'autopilot.py'), '--serve-tts-adapter'], env=env)

    wait_health(HealthCheck('local STT adapter', 'http://127.0.0.1:8001/health', 120))
    wait_health(HealthCheck('local TTS adapter', 'http://127.0.0.1:8002/health', 120))

    check_lmstudio()

    # Optionally launch the web UI token server
    if not service_ready('web UI', 'http://127.0.0.1:8100'):
        web_server_path = ROOT / 'web' / 'token_server.py'
        if web_server_path.exists():
            start_child([str(PYTHON_BIN), str(web_server_path)], env=env)
            wait_health(HealthCheck('web UI', 'http://127.0.0.1:8100', 15))

    run([str(PYTHON_BIN), '-m', 'app.agent', 'download-files'], cwd=ROOT, check=True, retries=2)

    # Determine launch mode: if LiveKit credentials exist, use 'dev' mode (connects to
    # LiveKit server for Web UI support). Otherwise fall back to 'console' mode.
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    livekit_url = os.getenv('LIVEKIT_URL', '').strip()
    if livekit_url and livekit_url != 'ws://localhost:7880':
        print(f'[start] launching agent in dev mode (connected to {livekit_url})...')
        print('[info]  open the LiveKit Agents Playground or http://127.0.0.1:8100 to interact via browser')
        run([str(PYTHON_BIN), '-m', 'app.agent', 'dev'], cwd=ROOT, check=True)
    else:
        print('[start] launching agent in console mode (local mic/speaker)...')
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
