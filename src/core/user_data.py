import sys
import os
import tempfile
import json
from datetime import datetime
from threading import Lock
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import secrets
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import generate_uuid
from utils.logging_utils import logging

user_data_lock = Lock()
user_data_store = {}


class UserData:
    def __init__(self, username):
        self.username = username
        self.user_id = generate_uuid()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.file_data = b""  # Store encrypted data
        self.report_data = b""
        self.rebuttal_data = b""
        self.lock = Lock()
        self.encryption_key, self.salt = self._generate_key()
        logging.info(
            f"\n{'=' * 50}\nNEW USER SESSION CREATED\nUsername: {self.username}\nUser ID: {self.user_id}\n{'=' * 50}\n")

    def _generate_key(self):
        """Generate a secure AES-256 key using PBKDF2."""
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        password = secrets.token_bytes(32)  # Random password for session
        key = kdf.derive(password)
        return key, salt

    def encrypt_data(self, data: str) -> bytes:
        """Encrypt data using AES-256 in CBC mode."""
        if not data:
            return b""
        iv = secrets.token_bytes(16)  # Random IV
        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padded_data = data.encode() + b" " * (16 - len(data.encode()) % 16)
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext)

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt data using AES-256 in CBC mode."""
        if not encrypted_data:
            return ""
        try:
            data = base64.b64decode(encrypted_data)
            iv = data[:16]
            ciphertext = data[16:]
            cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
            return decrypted_padded.rstrip(b" ").decode()
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            return ""

    def log_current_state(self):
        log_message = (
            f"\n{'=' * 50}\nUSER DATA SNAPSHOT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Username: {self.username}\n"
            f"User ID: {self.user_id}\n"
            f"File Data (encrypted, first 50 chars): {self.file_data[:50].hex() if self.file_data else 'None'}\n"
            f"Report Data (encrypted, first 50 chars): {self.report_data[:50].hex() if self.report_data else 'None'}\n"
            f"Rebuttal Data (encrypted, first 50 chars): {self.rebuttal_data[:50].hex() if self.rebuttal_data else 'None'}\n"
            f"{'=' * 50}\n"
        )
        logging.info(log_message)


def get_user_data(username):
    with user_data_lock:
        if username not in user_data_store:
            user_data_store[username] = UserData(username)
            logging.info(f"Created new user data for: {username}")
        else:
            logging.info(f"Retrieved existing user data for: {username}")
        return user_data_store[username]


def save_chat_history(chat_history, username):
    user_data = get_user_data(username)
    chat_history_dict = {}
    for idx, message in enumerate(chat_history):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_msg = message[0] if isinstance(message, (list, tuple)) else message
        bot_response = message[1] if isinstance(message, (list, tuple)) else ""
        chat_history_dict[f"message_{idx + 1}"] = {
            "user_msg": user_msg,
            "user_timestamp": timestamp,
            "bot_response": user_data.encrypt_data(bot_response).hex() if bot_response else "",
            "bot_timestamp": timestamp
        }
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir,
                                  f"chat_history_{user_data.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(temp_file_path, "w") as temp_file:
        json.dump(chat_history_dict, temp_file, indent=4)
    return temp_file_path


def download_chat_history(username):
    user_data = get_user_data(username)
    temp_dir = tempfile.gettempdir()
    temp_files = [f for f in os.listdir(temp_dir) if f.startswith(f"chat_history_{user_data.user_id}_")]
    if temp_files:
        latest_file = max(temp_files, key=lambda f: os.path.getctime(os.path.join(temp_dir, f)))
        return os.path.join(temp_dir, latest_file)
    return None
