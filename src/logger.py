# description of logging module: 
# This module provides a flexible framework for emitting log messages from Python programs.
# It is used to track events that happen when some software runs, which can be useful for
# debugging or understanding the flow of a program.

import logging
import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Generate log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] - %(lineno)d - %(name)s - %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Logger initialized successfully.")
        raise CustomException(e, sys)
