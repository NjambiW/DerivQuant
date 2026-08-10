import sys
import os
import json
import time
import pandas as pd
import websocket


# --------------------------------------------------
# FIND SRC DIRECTORY
# --------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

from authenticator import get_websocket_url


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SYMBOL = "R_100"

TARGET_TICKS = 20000

TICKS_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)


# --------------------------------------------------
# CONNECT
# --------------------------------------------------

def connect():
    """Connects to Deriv WebSocket."""

    url = get_websocket_url()

    ws = websocket.WebSocket()

    ws.connect(url)

    print("Connected to Deriv.")

    return ws


# --------------------------------------------------
# SAVE TICK
# --------------------------------------------------

def save_tick(tick):
    """Saves one tick to the forward CSV."""

    price = tick["quote"]
    epoch = tick["epoch"]
    symbol = tick["symbol"]

    last_digit = int(f"{price:.2f}"[-1])

    tick_data = {
        "time": epoch,
        "symbol": symbol,
        "price": price,
        "Last digit": last_digit
    }

    df = pd.DataFrame([tick_data])

    file_exists = os.path.isfile(TICKS_FILE)

    df.to_csv(
        TICKS_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


# --------------------------------------------------
# COLLECT
# --------------------------------------------------

def collect():

    ws = None

    collected = 0

    while collected < TARGET_TICKS:

        try:

            ws = connect()

            request = {
                "ticks": SYMBOL,
                "subscribe": 1
            }

            ws.send(json.dumps(request))

            print(
                f"Subscribed to {SYMBOL}."
            )

            print(
                f"Target ticks: {TARGET_TICKS}"
            )

            print()

            while collected < TARGET_TICKS:

                response = json.loads(
                    ws.recv()
                )

                if response.get("msg_type") != "tick":
                    continue

                tick = response["tick"]

                save_tick(tick)

                collected += 1

                if collected % 100 == 0:

                    print(
                        f"Collected "
                        f"{collected}/{TARGET_TICKS}"
                    )

            ws.close()

            break

        except KeyboardInterrupt:

            print(
                "\nCollector stopped by user."
            )

            if ws:
                ws.close()

            break

        except Exception as error:

            print()
            print("Connection lost.")
            print(f"Error: {error}")
            print("Reconnecting in 5 seconds...")

            if ws:

                try:
                    ws.close()

                except:
                    pass

            time.sleep(5)

    print()
    print("FORWARD COLLECTION COMPLETE.")
    print(f"Ticks collected: {collected}")
    print(f"Saved to: {TICKS_FILE}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    collect()