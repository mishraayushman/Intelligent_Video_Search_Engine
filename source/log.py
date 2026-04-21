import logging as logg
import os
from datetime import datetime

# 1. Setup paths
LOG_DIR = os.path.join(os.getcwd(), "Logs")
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H-%M-%S')}.log"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 2. Configure with HANDLERS only
logg.basicConfig(
    level=logg.INFO,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logg.FileHandler(LOG_FILE_PATH), # This replaces the 'filename' argument
        logg.StreamHandler()             # This allows you to see logs in the terminal
    ]
)