import os
from dotenv import load_dotenv

load_dotenv()

PATAPI = os.getenv("PATAPI")
APP_ID = os.getenv("APP_ID")
CLIENTid = os.getenv("CLIENTid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

TICKS_FILE = os.path.join(DATA_DIR, "ticks_data.csv")

FEATURE_FILE = os.path.join(DATA_DIR, "feature_data.csv")

FEATURE_V2_FILE = os.path.join(
    DATA_DIR,
    "feature_data_v2.csv"
)

FEATURE_V3_FILE = os.path.join(
    DATA_DIR,
    "feature_data_v3.csv"
)

FEATURE_V4_FILE = os.path.join(
    DATA_DIR,
    "feature_data_v4.csv"
)