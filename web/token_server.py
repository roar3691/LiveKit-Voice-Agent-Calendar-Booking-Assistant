#!/usr/bin/env python3
"""Lightweight token server that serves the Web UI and issues LiveKit JWTs."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', 'devkey')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', 'secret')
WEB_DIR = Path(__file__).resolve().parent
ROOM_PREFIX = 'calendar-agent'


async def index(_request: web.Request):
    return web.FileResponse(WEB_DIR / 'index.html')


async def get_config(_request: web.Request):
    """Return the LiveKit WebSocket URL so the frontend knows where to connect."""
    return web.json_response({'livekit_url': LIVEKIT_URL.replace('http://', 'ws://').replace('https://', 'wss://')})


async def get_token(request: web.Request):
    """Generate a signed JWT for the given identity to join the agent room."""
    from livekit import api

    identity = request.query.get('identity', 'web-user')
    requested_room = request.query.get('room', '').strip()
    room_name = requested_room or f'{ROOM_PREFIX}-{int(time.time())}'

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name='calendar-agent')],
            )
        )
        .to_jwt()
    )
    return web.json_response({'token': token, 'room': room_name})


async def health(_request: web.Request):
    return web.json_response({'ok': True, 'service': 'web-ui-token-server'})


app = web.Application()
app.add_routes([
    web.get('/', index),
    web.get('/api/config', get_config),
    web.get('/api/token', get_token),
    web.get('/health', health),
])

if __name__ == '__main__':
    print('[web] starting token server on http://127.0.0.1:8100')
    web.run_app(app, host='127.0.0.1', port=8100)
