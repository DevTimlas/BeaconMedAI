import sys
import os
import threading
from auth.authentication import validate_login, cleanup_expired_sessions
from ui.gradio_interface import gradio_ui
from utils.logging_utils import logging
from core.vectorstore import initialize_vectorstore_for_edition

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    logging.info("Starting BeaconMedicalAi AMA application")
    try:
        # Attempt to initialize vectorstore, but don't crash if it fails
        try:
            initialize_vectorstore_for_edition("4th Edition")
        except Exception as e:
            logging.error(f"Failed to initialize vectorstore on startup: {e}")
            logging.info("Proceeding with application launch, vectorstore can be initialized later via UI")

        port = int(os.environ.get('PORT', 7860))
        cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
        cleanup_thread.start()

        demo = gradio_ui()
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            favicon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png'),
            auth=validate_login,
            share=False
        )
    except Exception as e:
        logging.error(f"Error launching application: {e}")
        print(f"Error launching application: {e}")


if __name__ == "__main__":
    main()
