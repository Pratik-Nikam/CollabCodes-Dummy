from abc import ABC, abstractmethod
from datetime import datetime

class ConversationLogger(ABC):
    @abstractmethod
    async def log(self, user_id: str, user_name: str, message: str, direction: str):
        pass


from core.db_logger import ConversationLogger
from core.database import get_db
from datetime import datetime

class OracleConversationLogger(ConversationLogger):
    async def log(self, user_id: str, user_name: str, message: str, direction: str):
        conn_gen = get_db()
        conn = next(conn_gen)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO CONVERSATIONS (USER_ID, USER_NAME, MESSAGE, DIRECTION, LOG_TIME)
                VALUES (:user_id, :user_name, :message, :direction, :log_time)
            """, {
                "user_id": user_id,
                "user_name": user_name,
                "message": message,
                "direction": direction,
                "log_time": datetime.now()
            })
            conn.commit()
        finally:
            cursor.close()
            next(conn_gen, None)


from core.db_logger import ConversationLogger
from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
collection = client.chatbot.conversations

class MongoConversationLogger(ConversationLogger):
    async def log(self, user_id: str, user_name: str, message: str, direction: str):
        collection.insert_one({
            "user_id": user_id,
            "user_name": user_name,
            "message": message,
            "direction": direction,
            "timestamp": datetime.now()
        })


from core.oracle_logger import OracleConversationLogger
from core.mongo_logger import MongoConversationLogger

# Choose dynamically
use_mongo = False  # Toggle this

conversation_logger = MongoConversationLogger() if use_mongo else OracleConversationLogger()

from core.oracle_logger import OracleConversationLogger
from core.mongo_logger import MongoConversationLogger

# Choose dynamically
use_mongo = False  # Toggle this

conversation_logger = MongoConversationLogger() if use_mongo else OracleConversationLogger()


import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

file_handler = RotatingFileHandler(
    filename=f"{LOG_DIR}/app.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger("chatbot")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

from core.logger import logger

logger.info("Bot started")
logger.error("Failed to connect to DB")





