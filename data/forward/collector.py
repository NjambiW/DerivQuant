import sys
import os
import json
import time
import pandas as pd
import websocket


# ============================================================
# FIND PROJECT ROOT AND SRC
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from authenticator import get_websocket_url


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "R_100"

TARGET_TICKS = 60000

BATCH_SIZE = 1000

TICKS_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)


# ============================================================
# CONNECT
# ============================================================

def connect():

    url = get_websocket_url()

    print()
    print("Connecting to Deriv...")

    ws = websocket.WebSocket()

    ws.connect(url)

    print("Connected to Deriv.")

    return ws


# ============================================================
# GET SAVED COUNT
# ============================================================

def get_saved_count():

    if not os.path.isfile(TICKS_FILE):
        return 0

    df = pd.read_csv(TICKS_FILE)

    if df.empty:
        return 0

    return len(df)


# ============================================================
# GET LAST SAVED TIMESTAMP
# ============================================================

def get_last_timestamp():

    if not os.path.isfile(TICKS_FILE):
        return None

    df = pd.read_csv(TICKS_FILE)

    if df.empty:
        return None

    return int(df["time"].max())


# ============================================================
# SAVE ONE TICK
# ============================================================

def save_tick(tick):

    price = float(tick["quote"])
    epoch = int(tick["epoch"])
    symbol = tick["symbol"]

    last_digit = int(
        f"{price:.2f}"[-1]
    )

    tick_data = {
        "time": epoch,
        "symbol": symbol,
        "price": price,
        "Last digit": last_digit
    }

    df = pd.DataFrame(
        [tick_data]
    )

    file_exists = os.path.isfile(
        TICKS_FILE
    )

    df.to_csv(
        TICKS_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


# ============================================================
# RECOVER MISSING TICKS
# ============================================================

def recover_missing_ticks(ws):

    last_timestamp = get_last_timestamp()

    if last_timestamp is None:

        print(
            "No previous timestamp found."
        )

        return


    print()
    print(
        "======================================"
    )

    print(
        "FORWARD DATA RECOVERY"
    )

    print(
        "======================================"
    )

    print(
        f"Latest saved timestamp: "
        f"{last_timestamp}"
    )


    current_end = int(
        time.time()
    )

    total_recovered = 0


    while last_timestamp < current_end:

        request = {

            "ticks_history": SYMBOL,

            "start": last_timestamp + 1,

            "end": current_end,

            "count": BATCH_SIZE,

            "style": "ticks"
        }


        ws.send(
            json.dumps(request)
        )


        response = json.loads(
            ws.recv()
        )


        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        if response.get("error"):

            print()
            print(
                "DERIV ERROR:"
            )

            print(
                response
            )

            break


        # ----------------------------------------------------
        # CHECK HISTORY
        # ----------------------------------------------------

        if response.get(
            "msg_type"
        ) != "history":

            print(
                "Unexpected response:"
            )

            print(
                response
            )

            break


        history = response.get(
            "history",
            {}
        )


        prices = history.get(
            "prices",
            []
        )

        times = history.get(
            "times",
            []
        )


        if not times:

            break


        # ----------------------------------------------------
        # SAVE ONLY NEW TICKS
        # ----------------------------------------------------

        new_ticks = 0

        newest_timestamp = last_timestamp


        for price, epoch in zip(
            prices,
            times
        ):

            epoch = int(epoch)


            if epoch <= last_timestamp:

                continue


            tick = {

                "quote": price,

                "epoch": epoch,

                "symbol": SYMBOL
            }


            save_tick(tick)

            new_ticks += 1

            total_recovered += 1


            if epoch > newest_timestamp:

                newest_timestamp = epoch


        # ----------------------------------------------------
        # SHOW PROGRESS
        # ----------------------------------------------------

        print(
            f"Recovered batch: "
            f"{new_ticks} ticks"
        )


        print(
            f"Newest timestamp: "
            f"{newest_timestamp}"
        )


        # ----------------------------------------------------
        # PREVENT INFINITE LOOP
        # ----------------------------------------------------

        if newest_timestamp <= last_timestamp:

            print(
                "Timestamp did not advance."
            )

            break


        last_timestamp = newest_timestamp


        # ----------------------------------------------------
        # REACHED CURRENT TIME
        # ----------------------------------------------------

        if last_timestamp >= current_end:

            break


        time.sleep(
            0.2
        )


    print()
    print(
        f"Total new ticks recovered: "
        f"{total_recovered}"
    )

    print(
        f"Dataset now contains: "
        f"{get_saved_count()} ticks"
    )

    print(
        "======================================"
    )

    print()


# ============================================================
# SUBSCRIBE TO LIVE TICKS
# ============================================================

def subscribe_live_ticks(ws):

    request = {

        "ticks": SYMBOL,

        "subscribe": 1
    }


    ws.send(
        json.dumps(request)
    )


    print(
        f"Subscribed to live "
        f"{SYMBOL} ticks."
    )


# ============================================================
# LIVE COLLECTION
# ============================================================

def collect_live_ticks():

    ws = None


    while True:

        try:

            # ------------------------------------------------
            # CONNECT
            # ------------------------------------------------

            ws = connect()


            saved_count = (
                get_saved_count()
            )


            print(
                f"Existing forward ticks: "
                f"{saved_count}"
            )


            # ------------------------------------------------
            # RECOVER GAP
            # ------------------------------------------------

            recover_missing_ticks(
                ws
            )


            # ------------------------------------------------
            # CHECK TARGET
            # ------------------------------------------------

            saved_count = (
                get_saved_count()
            )


            if saved_count >= TARGET_TICKS:

                print()
                print(
                    "======================================"
                )

                print(
                    "TARGET REACHED"
                )

                print(
                    f"Total ticks: "
                    f"{saved_count}"
                )

                print(
                    "======================================"
                )

                ws.close()

                return


            # ------------------------------------------------
            # SUBSCRIBE
            # ------------------------------------------------

            subscribe_live_ticks(
                ws
            )


            # ------------------------------------------------
            # LIVE STREAM
            # ------------------------------------------------

            while True:

                response = json.loads(
                    ws.recv()
                )


                # Ignore non-tick messages

                if response.get(
                    "msg_type"
                ) != "tick":

                    continue


                tick = response[
                    "tick"
                ]


                save_tick(
                    tick
                )


                saved_count += 1


                print(
                    f"Saved live tick "
                    f"{saved_count}/{TARGET_TICKS} | "
                    f"{tick['symbol']} | "
                    f"{tick['quote']} | "
                    f"{tick['epoch']}"
                )


                # --------------------------------------------
                # TARGET
                # --------------------------------------------

                if saved_count >= TARGET_TICKS:

                    print()
                    print(
                        "======================================"
                    )

                    print(
                        "FORWARD COLLECTION COMPLETE"
                    )

                    print(
                        f"Total ticks: "
                        f"{saved_count}"
                    )

                    print(
                        f"Saved to: "
                        f"{TICKS_FILE}"
                    )

                    print(
                        "======================================"
                    )

                    ws.close()

                    return


        # ----------------------------------------------------
        # USER STOPPED
        # ----------------------------------------------------

        except KeyboardInterrupt:

            print()
            print(
                "Collector stopped by user."
            )


            if ws:

                try:
                    ws.close()

                except:
                    pass


            break


        # ----------------------------------------------------
        # CONNECTION LOST
        # ----------------------------------------------------

        except Exception as error:

            print()
            print(
                "Connection lost."
            )

            print(
                f"Error: {error}"
            )


            if ws:

                try:
                    ws.close()

                except:
                    pass


            print(
                "Reconnecting in 5 seconds..."
            )

            time.sleep(
                5
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "======================================"
    )

    print(
        "V4 FORWARD DATA COLLECTOR"
    )

    print(
        "======================================"
    )

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Target ticks: {TARGET_TICKS}"
    )

    print(
        f"Existing ticks: "
        f"{get_saved_count()}"
    )

    print(
        f"Saving to: "
        f"{TICKS_FILE}"
    )

    print(
        "======================================"
    )

    collect_live_ticks()