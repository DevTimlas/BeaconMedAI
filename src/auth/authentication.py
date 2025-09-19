import sys
import os
import time

from utils.logging_utils import logging

AUTH_CREDENTIALS = {
    "michael rice": "4321",
    "Tim": "1038",
    "Tester": "Test012"
}

ACTIVE_SESSIONS = {}
SESSION_TIMEOUT = 60 * 60 * 30  # 30 hours


def validate_login(username, password):
    if username not in AUTH_CREDENTIALS or AUTH_CREDENTIALS[username] != password:
        return False, "Invalid credentials"
    if username in ACTIVE_SESSIONS:
        session = ACTIVE_SESSIONS[username]
        if time.time() - session["last_activity"] > SESSION_TIMEOUT:
            del ACTIVE_SESSIONS[username]
        else:
            return False, "This account is already in use"
    ACTIVE_SESSIONS[username] = {
        "last_activity": time.time(),
        "session_id": str(hash(f"{username}{time.time()}"))
    }
    logging.info(f"Successful login for user: {username}")
    return True, "Login successful"


def cleanup_expired_sessions():
    while True:
        current_time = time.time()
        expired_users = [
            username for username, session in ACTIVE_SESSIONS.items()
            if current_time - session["last_activity"] > SESSION_TIMEOUT
        ]
        for username in expired_users:
            if username in ACTIVE_SESSIONS:
                del ACTIVE_SESSIONS[username]
            logging.info(f"Cleaned up expired session for {username}")
        time.sleep(60)
