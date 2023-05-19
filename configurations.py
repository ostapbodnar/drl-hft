import logging
import os

import pytz as tz


# Data Directory
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data_recorder', 'database', 'data_exports')

LOGS_PATH = os.path.join(ROOT_PATH, "logs", "hft.log")
print("LOGS_PATH: ", LOGS_PATH)
# singleton for logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler(LOGS_PATH)])
LOGGER = logging.getLogger('crypto_rl_log')

# ./recorder.py
SNAPSHOT_RATE = 1.0  # For example, 0.25 = 4x per second

MAX_BOOK_ROWS = 15

TIMEZONE = tz.utc

# ./gym_trading/utils/broker.py
MARKET_ORDER_FEE = 0.0020
LIMIT_ORDER_FEE = 0.0
SLIPPAGE = 0.0005

# ./indicators/*
INDICATOR_WINDOW = [60 * i for i in [5, 15]]  # Convert minutes to seconds
INDICATOR_WINDOW_MAX = max(INDICATOR_WINDOW)
INDICATOR_WINDOW_FEATURES = [f'_{i}' for i in [5, 15]]  # Create labels
EMA_ALPHA = 0.99  # [0.9, 0.99, 0.999, 0.9999]

# agent penalty configs
ENCOURAGEMENT = 0.000000000001

