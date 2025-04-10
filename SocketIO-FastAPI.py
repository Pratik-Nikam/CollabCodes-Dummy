from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import socketio
from app.core.config import load_config
from app.core.logger import setup_logger
from app.core.mongo import init_mongo
from app.socket_events import register_socket_events
from app.routes.tasks import router as task_router

load_config()
logger = setup_logger()

fastapi_app = FastAPI(
    title="Chatbot Platform API",
    version="1.0.0",
    description="Modular Socket.IO Chatbot Backend with Dynamic Intent Handling"
)

fastapi_app.include_router(task_router)

@fastapi_app.on_event("startup")
async def startup_event():
    await init_mongo()

def custom_openapi():
    if fastapi_app.openapi_schema:
        return fastapi_app.openapi_schema
    openapi_schema = get_openapi(
        title=fastapi_app.title,
        version=fastapi_app.version,
        description=fastapi_app.description,
        routes=fastapi_app.routes,
    )
    fastapi_app.openapi_schema = openapi_schema
    return fastapi_app.openapi_schema

fastapi_app.openapi = custom_openapi

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

register_socket_events(sio)

@fastapi_app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "ok"}

@fastapi_app.get("/docs", tags=["Documentation"])
def docs_redirect():
    return fastapi_app.openapi()


from app.core.registry import get_task
from app.core.intent_classifier import classify_intent
from app.utils import extract_fn_name, extract_params, log_conversation

def register_socket_events(sio):

    @sio.event
    async def connect(sid, environ):
        print(f"Client connected: {sid}")

    @sio.event
    async def disconnect(sid):
        print(f"Client disconnected: {sid}")

    @sio.event
    async def user_uttered(sid, data):
        message = data.get("message", "")
        fn_name = extract_fn_name(message)
        kwargs = extract_params(message)

        await log_conversation(sid, "user", message)

        if fn_name == "session_init":
            await sio.emit("session_request", {"session_id": f"conversation_{sid}"}, to=sid)
            await sio.emit("session_confirm", {}, to=sid)

        intent = classify_intent(message)
        task = get_task(intent)

        if task:
            response = await task.process(kwargs)
        else:
            response = {"text": "Sorry, I didn't understand that."}

        await sio.emit("bot_uttered", {
            "text": response.get("text", ""),
            "meta": {},
            "payload": response
        }, to=sid)

        await log_conversation(sid, "bot", response.get("text", ""), intent)


import json
from datetime import datetime
from app.core.mongo import get_collection

def extract_fn_name(message: str):
    try:
        return message[:message.find("{")].replace("/", "").strip()
    except:
        return ""

def extract_params(message: str):
    try:
        json_part = message[message.find("{"):]
        return json.loads(json_part)
    except:
        return {}

async def log_conversation(sid: str, role: str, message: str, intent: str = None):
    log = {
        "session_id": sid,
        "role": role,
        "message": message,
        "intent": intent,
        "timestamp": datetime.utcnow()
    }
    await get_collection("conversations").insert_one(log)


import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "chatbot.log")

    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger



import os
from dotenv import load_dotenv
import yaml

CONFIG = {}

def load_config():
    env = os.getenv("ENV", "development")
    env_file = f".env.{env}" if os.path.exists(f".env.{env}") else ".env"
    load_dotenv(dotenv_path=env_file)

    config_path = f"config/{env}.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            global CONFIG
            CONFIG = yaml.safe_load(f)

    return CONFIG


from motor.motor_asyncio import AsyncIOMotorClient
import os

mongo_client = None
mongo_db = None

async def init_mongo():
    global mongo_client, mongo_db
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_client = AsyncIOMotorClient(mongo_uri)
    mongo_db = mongo_client["chatbot"]

def get_collection(name: str):
    return mongo_db[name]


from abc import ABC, abstractmethod

class BaseTask(ABC):
    intent: str
    patterns: list
    description: str = ""

    @abstractmethod
    async def process(self, params: dict):
        pass


import os, sys, importlib
from app.core.task_base import BaseTask

TASKS = {}

def load_tasks():
    sys.path.insert(0, os.path.abspath("app/tasks"))
    for file in os.listdir("app/tasks"):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]
            module = importlib.import_module(module_name)
            if hasattr(module, "TaskHandler"):
                task: BaseTask = module.TaskHandler()
                TASKS[task.intent] = task

def get_task(intent_name: str):
    return TASKS.get(intent_name)

def list_tasks():
    return [
        {
            "intent": task.intent,
            "patterns": task.patterns,
            "description": getattr(task, "description", "")
        }
        for task in TASKS.values()
    ]

load_tasks()


import re
from app.core.registry import TASKS

def classify_intent(message: str) -> str:
    for intent, task in TASKS.items():
        for pattern in task.patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return intent
    return "unknown"

from fastapi import APIRouter
from app.core.registry import list_tasks

router = APIRouter()

@router.get("/tasks", tags=["Tasks"])
async def get_task_metadata():
    return list_tasks()


from app.core.task_base import BaseTask

class TaskHandler(BaseTask):
    intent = "create_ticket"
    patterns = ["create.*ticket", "open.*ticket"]
    description = "Guides user through the process of creating a support ticket."

    async def process(self, params):
        if "gci" not in params:
            return {
                "text": "Please enter your GCI to continue.",
                "template": "enter_gci"
            }

        cases = self.fetch_cases(params["gci"])
        return {
            "text": "Select a case to continue:",
            "template": "list_cases",
            "cases": cases
        }

    def fetch_cases(self, gci):
        return [
            {"id": "C101", "description": "Login issue"},
            {"id": "C102", "description": "Error loading dashboard"}
        ]


from app.core.task_base import BaseTask

class TaskHandler(BaseTask):
    intent = "get_case_details"
    patterns = ["get.*case.*details", "show.*case"]
    description = "Retrieve case details for a given GCI."

    async def process(self, params):
        gci = params.get("gci")
        if not gci:
            return {
                "text": "Please provide a GCI to fetch case details.",
                "template": "enter_gci"
            }

        details = self.fetch_case_details(gci)
        return {
            "text": f"Here are the case details for {gci}:",
            "details": details
        }

    def fetch_case_details(self, gci):
        return [
            {"case_id": "C123", "status": "open", "summary": "Login issue"},
            {"case_id": "C124", "status": "in-progress", "summary": "System crash"}
        ]


PORT=5005
HOST=0.0.0.0
ENV=development
LOG_DIR=logs
MONGO_URI=mongodb://localhost:27017

# File: run.py

import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )


# app/core/config.py

import os
import yaml

CONFIG = {}

def load_config():
    env = os.getenv("ENV", "development")

    # Only try .env file locally
    if os.getenv("USE_ENV_FILE", "true").lower() == "true":
        from dotenv import load_dotenv
        env_file = f".env.{env}" if os.path.exists(f".env.{env}") else ".env"
        load_dotenv(dotenv_path=env_file)

    # Load YAML config optionally
    config_path = f"config/{env}.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            global CONFIG
            CONFIG = yaml.safe_load(f)

    return CONFIG


# File: Makefile

# Default environment variables (override as needed)
ENV ?= development
PORT ?= 5005
HOST ?= 0.0.0.0

.PHONY: install run lint format test

install:
	pip install -r requirements.txt

run:
	ENV=$(ENV) HOST=$(HOST) PORT=$(PORT) python run.py

lint:
	flake8 app

format:
	black app

test:
	pytest -v

<!-- File: public/socket-playground.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Socket.IO Chatbot Playground</title>
  <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
</head>
<body>
  <h2>Chatbot Socket.IO Playground</h2>
  <div>
    <input id="input" placeholder="Type message here...">
    <button onclick="sendMessage()">Send</button>
  </div>
  <pre id="log" style="background:#f0f0f0; padding:1em;"></pre>

  <script>
    const socket = io("http://localhost:5005");
    const log = document.getElementById("log");

    socket.on("connect", () => logMessage("Connected: " + socket.id));
    socket.on("bot_uttered", data => logMessage("Bot: " + JSON.stringify(data)));
    socket.on("session_request", data => logMessage("Session Request: " + JSON.stringify(data)));
    socket.on("session_confirm", data => logMessage("Session Confirmed"));

    function sendMessage() {
      const msg = document.getElementById("input").value;
      socket.emit("user_uttered", { message: msg });
      logMessage("You: " + msg);
    }

    function logMessage(text) {
      log.textContent += text + "\n";
    }
  </script>
</body>
</html>


from fastapi import APIRouter

router = APIRouter()

@router.get("/socket-events", tags=["Documentation"])
async def socket_events_doc():
    return {
        "description": "List of supported Socket.IO events and their formats.",
        "events": {
            "connect": "Triggered automatically when a user connects.",
            "disconnect": "Triggered when the user disconnects.",
            "user_uttered": {
                "description": "User sends a message to the bot.",
                "format": {
                    "message": "/intent_name{\"key\": \"value\"}"
                }
            },
            "bot_uttered": {
                "description": "Bot responds with a structured message.",
                "format": {
                    "text": "Hey there!",
                    "meta": {},
                    "payload": {}
                }
            },
            "session_request": "Sent by the server to initiate a session.",
            "session_confirm": "Sent by the server to confirm the session is ready."
        }
    }



    fastapi_app.include_router(socket_doc_router)

    








