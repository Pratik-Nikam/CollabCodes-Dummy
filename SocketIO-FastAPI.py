# File: app/main.py

import socketio
from fastapi import FastAPI
from app.socket_events import register_socket_events

# Initialize base FastAPI app for optional APIs
fastapi_app = FastAPI()

# Create Socket.IO Async Server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")

# Wrap with SocketIO ASGI App
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

# Register all Socket.IO event handlers
register_socket_events(sio)

# File: app/socket_events.py

def register_socket_events(sio):
    from app.handlers import FN_MAPPER
    from app.utils import extract_fn_name, extract_params

    @sio.event
    async def connect(sid, environ):
        print(f"Client connected: {sid}")

    @sio.event
    async def disconnect(sid):
        print(f"Client disconnected: {sid}")

    @sio.event
    async def user_uttered(sid, data):
        print(f"user_uttered received: {data}")

        message = data.get("message", "")
        fn_name = extract_fn_name(message)
        kwargs = extract_params(message)

        if fn_name == "session_init":
            # Step 1: Emit session_request
            await sio.emit("session_request", {"session_id": f"conversation_{sid}"}, to=sid)

            # Step 2: Emit session_confirm
            await sio.emit("session_confirm", {}, to=sid)

        fn = FN_MAPPER.get(fn_name)
        if fn:
            response = fn(**kwargs)
        else:
            response = "Sorry, I didn’t understand."

        await sio.emit("bot_uttered", {
            "text": response,
            "meta": {}
        }, to=sid)

# File: app/handlers.py

def session_init(**kwargs):
    return f"Hey {kwargs.get('common_authenticated_user_display_name', 'User')}, how can I help you?"

def demo(**kwargs):
    return f"Captured {kwargs.get('demo_entity', 'N/A')} value"

FN_MAPPER = {
    "session_init": session_init,
    "demo": demo
}

# File: app/utils.py

import json

def extract_params(message: str):
    try:
        json_part = message[message.find('{'):]
        return json.loads(json_part)
    except Exception:
        return {}

def extract_fn_name(message: str):
    try:
        fn_name = message[:message.find('{')].replace('/', '').strip()
        return fn_name
    except Exception:
        return ""

