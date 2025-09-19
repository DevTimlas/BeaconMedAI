import sys
import os
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add console handler
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
logger.add(os.path.join(log_dir, "beacon_medical_ai_{time}.log"), rotation="1 MB", level="DEBUG")

logging = logger
