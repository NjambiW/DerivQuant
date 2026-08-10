import json
import os
import time
import pandas as pd
import websocket

from authenticator import get_websocket_url
from config import TICKS_FILE


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "R_100"

# How many historical ticks to request per batch
HISTORY_BATCH_SIZE = 1000

# How often to print progress while collecting
PROGRESS_EVERY = 10


# ============================================================
# CONNECT TO DERIV
# ============================================================

def connect():
    """Connect to Deriv WebSocket."""

    print("\n========================================")
    print("CONNECTING TO DERIV")
    print("========================================")

    url = get_websocket_url()

    print("WebSocket URL obtained.")
    print("Connecting...")

    ws = websocket.WebSocket()
    ws.connect(url)

    print("CONNECTED SUCCESSFULLY")

    return ws


# ============================================================
# GET LAST SAVED TIMESTAMP
# ============================================================

def get_last_timestamp():

    if not os.path.isfile(TICKS_FILE):

        print("\nNo existing tick file found.")
        print("Starting from live data.")

        return None

    try:

        df = pd.read_csv(TICKS_FILE)

    except Exception as error:

        print("\nCould not read tick file.")
        print("Error:", error)

        return None

    if df.empty:

        print("\nTick file exists but is empty.")

        return None

    if "time" not in df.columns:

        print("\nERROR: 'time' column does not exist.")

        return None

    last_timestamp = int(df["time"].max())

    print("\n========================================")
    print("LAST SAVED TICK")
    print("========================================")

    print("Last timestamp:", last_timestamp)
    print("Rows currently in file:", len(df))

    return last_timestamp


# ============================================================
# SAVE ONE TICK
# ============================================================

def save_tick(tick):

    price = float(tick["quote"])
    epoch = int(tick["epoch"])
    symbol = tick["symbol"]

    # R_100 has 2 decimal places
    last_digit = int(f"{price:.2f}"[-1])

    tick_data = {
        "time": epoch,
        "symbol": symbol,
        "price": price,
        "Last digit": last_digit
    }

    df = pd.DataFrame([tick_data])

    file_exists = os.path.isfile(TICKS_FILE)

    # Make sure directory exists
    parent = os.path.dirname(TICKS_FILE)

    if parent:
        os.makedirs(parent, exist_ok=True)

    df.to_csv(
        TICKS_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


# ============================================================
# RECOVER HISTORICAL DATA
# ============================================================

def recover_missing_ticks(ws, last_timestamp):
    """Recover all missing ticks in batches until reaching latest data."""

    if last_timestamp is None:
        print("No previous timestamp found.")
        print("Skipping historical recovery.")
        return

    print("\n========================================")
    print("HISTORICAL DATA RECOVERY")
    print("========================================")

    print(f"Last saved timestamp: {last_timestamp}")

    current_timestamp = last_timestamp
    total_recovered = 0
    batch_number = 0

    while True:

        batch_number += 1

        print("\n----------------------------------------")
        print(f"REQUESTING HISTORY BATCH {batch_number}")
        print("----------------------------------------")

        request = {
            "ticks_history": SYMBOL,
            "start": current_timestamp,
            "end": "latest",
            "count": 1000,
            "style": "ticks"
        }

        ws.send(json.dumps(request))

        response = json.loads(ws.recv())

        # Check for Deriv error
        if "error" in response:
            print("\nDERIV ERROR:")
            print(response["error"])
            break

        if response.get("msg_type") != "history":

            print("\nUnexpected response:")
            print(response)

            break

        history = response["history"]

        prices = history.get("prices", [])
        times = history.get("times", [])

        print(f"Ticks returned by Deriv: {len(times)}")

        if not times:

            print("No more historical ticks available.")
            break

        batch_recovered = 0

        newest_timestamp = current_timestamp

        for price, epoch in zip(prices, times):

            epoch = int(epoch)

            # Don't save ticks already in our CSV
            if epoch <= last_timestamp:
                continue

            tick = {
                "quote": price,
                "epoch": epoch,
                "symbol": SYMBOL
            }

            save_tick(tick)

            batch_recovered += 1
            total_recovered += 1

            if epoch > newest_timestamp:
                newest_timestamp = epoch

        print(f"New ticks recovered: {batch_recovered}")
        print(f"Newest timestamp: {newest_timestamp}")

        # Prevent infinite loop
        if newest_timestamp <= current_timestamp:

            print("\nTimestamp did not advance.")
            print("Stopping recovery.")

            break

        # Move forward
        current_timestamp = newest_timestamp

        # If Deriv returned less than 1000,
        # we have probably reached the latest available data.
        if len(times) < 1000:

            print("\nLess than 1000 ticks returned.")
            print("Historical recovery has reached the latest data.")

            break

    print("\n========================================")
    print("HISTORICAL RECOVERY COMPLETE")
    print("========================================")

    print(f"Total new ticks recovered: {total_recovered}")

# ============================================================
# SUBSCRIBE TO LIVE TICKS
# ============================================================

def subscribe_live_ticks(ws):

    print("\n========================================")
    print("STARTING LIVE COLLECTION")
    print("========================================")

    request = {
        "ticks": SYMBOL,
        "subscribe": 1
    }

    ws.send(json.dumps(request))

    print("Subscription request sent.")
    print("Waiting for live ticks...")


# ============================================================
# LIVE COLLECTION
# ============================================================

def stream_live_ticks():

    print("\n")
    print("========================================")
    print("R_100 TICK COLLECTOR")
    print("========================================")

    print("Symbol:", SYMBOL)
    print("Saving to:", TICKS_FILE)

    ws = None

    while True:

        try:

            # ------------------------------------------------
            # CONNECT
            # ------------------------------------------------

            ws = connect()

            # ------------------------------------------------
            # FIND LAST SAVED DATA
            # ------------------------------------------------

            last_timestamp = get_last_timestamp()

            # ------------------------------------------------
            # RECOVER MISSED DATA
            # ------------------------------------------------

            recover_missing_ticks(
                ws,
                last_timestamp
            )

            # ------------------------------------------------
            # START LIVE SUBSCRIPTION
            # ------------------------------------------------

            subscribe_live_ticks(ws)

            tick_count = 0

            # ------------------------------------------------
            # RECEIVE LIVE TICKS FOREVER
            # ------------------------------------------------

            while True:

                raw_response = ws.recv()

                response = json.loads(raw_response)

                msg_type = response.get("msg_type")

                # Ignore non-tick messages
                if msg_type != "tick":

                    print("\nReceived message:", response)

                    continue

                tick = response["tick"]

                save_tick(tick)

                tick_count += 1

                print(
                    f"Saved tick #{tick_count} | "
                    f"{tick['symbol']} | "
                    f"{tick['quote']} | "
                    f"{tick['epoch']}"
                )

        # ----------------------------------------------------
        # USER STOPS PROGRAM
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print("\n\n========================================")
            print("COLLECTOR STOPPED BY USER")
            print("========================================")

            if ws:

                try:
                    ws.close()

                except:
                    pass

            break

        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except Exception as error:

            print("\n\n========================================")
            print("CONNECTION ERROR")
            print("========================================")

            print(error)

            if ws:

                try:
                    ws.close()

                except:
                    pass

            print("\nReconnecting in 5 seconds...")

            time.sleep(5)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("\nPROGRAM STARTED")

    stream_live_ticks()