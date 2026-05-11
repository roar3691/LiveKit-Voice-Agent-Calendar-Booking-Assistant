import asyncio
import os
import logging
from livekit.plugins import openai
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
load_dotenv(".env")

async def main():
    tts = openai.TTS(
        model="tts-1",
        voice=os.environ["LOCAL_TTS_VOICE"],
        api_key=os.environ["LOCAL_TTS_API_KEY"],
        base_url=os.environ["LOCAL_TTS_BASE_URL"],
        response_format="wav"
    )
    audio_stream = tts.synthesize("Hi, this is a test.")
    
    frames = []
    try:
        async for frame in audio_stream:
            frames.append(frame)
            print("got frame:", frame)
    except Exception as e:
        print("Exception:", e)

    print("Total frames:", len(frames))

if __name__ == "__main__":
    asyncio.run(main())
