import asyncio
import os
import logging
from livekit.plugins import openai
from livekit import rtc
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
load_dotenv(".env")

async def main():
    stt = openai.STT(
        model=os.environ["LOCAL_STT_MODEL"],
        api_key=os.environ["LOCAL_STT_API_KEY"],
        base_url=os.environ["LOCAL_STT_BASE_URL"]
    )
    
    # Generate some silence PCM frames to push
    # 1 second of silence at 16000Hz 1 channel
    samples = bytes([0] * 16000 * 2)
    frame = rtc.AudioFrame(sample_rate=16000, num_channels=1, samples_per_channel=16000)
    # copy samples (ctypes / whatever) - wait, rtc.AudioFrame doesn't have an easy constructor with raw bytes from python.
    # Actually, livekit.plugins.openai STT doesn't accept frames directly, it expects a stream.
    stream = stt.stream()
    
    # Let's just create a raw wav file and use recognize()
    with open("test.wav", "wb") as f:
        # minimal wav header
        f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')

    # rtc doesn't easily expose a way to load wav to frame in python, let's just test the STT adapter directly using httpx
    # wait, livekit.plugins.openai.STT uses client.audio.transcriptions.create
    # Let's just run it!
    import httpx
    with open("test.wav", "rb") as f:
        resp = httpx.post(os.environ["LOCAL_STT_BASE_URL"] + "/audio/transcriptions", files={"file": f}, data={"model": "whisper-1"})
        print(resp.status_code, resp.text)

if __name__ == "__main__":
    asyncio.run(main())
