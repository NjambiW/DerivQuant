import json
import os
import time
import pandas as pd
import websocket
from authenticator import get_websocket_url


CSV_FILE = "ticks_data.csv"
SYMBOL = "R_100"


def connect():
    """Connects to Deriv WebSocket and returns the connection."""

    url = get_websocket_url()

    ws = websocket.WebSocket()
    ws.connect(url)

    print("Connected to Deriv.")

    return ws


def get_last_timestamp():
    """Gets the timestamp of the last tick saved in the CSV."""

    if not os.path.isfile(CSV_FILE):
        return None

    df = pd.read_csv(CSV_FILE)

    if df.empty:
        return None

    return int(df["time"].max())


def save_tick(tick):
    """Saves one tick to the CSV file."""

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

    file_exists = os.path.isfile(CSV_FILE)

    df.to_csv(
        CSV_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


def recover_missing_ticks(ws, last_timestamp):
    """Recovers ticks missed while the connection was down."""

    if last_timestamp is None:
        return

    print(
        f"Recovering ticks from timestamp "
        f"{last_timestamp}..."
    )

    request = {
        "ticks_history": SYMBOL,
        "start": last_timestamp,
        "end": "latest",
        "style": "ticks"
    }

    ws.send(json.dumps(request))

    response = json.loads(ws.recv())

    if response.get("msg_type") != "history":
        print("Could not recover historical ticks.")
        print(response)
        return

    history = response["history"]

    prices = history["prices"]
    times = history["times"]

    recovered = 0

    for price, epoch in zip(prices, times):

        # Prevent duplicate ticks
        if epoch <= last_timestamp:
            continue

        tick = {
            "quote": price,
            "epoch": epoch,
            "symbol": SYMBOL
        }

        save_tick(tick)

        recovered += 1

    print(f"Recovered {recovered} missing ticks.")


def subscribe_live_ticks(ws):
    """Subscribes to live ticks."""

    request = {
        "ticks": SYMBOL,
        "subscribe": 1
    }

    ws.send(json.dumps(request))

    print(f"Subscribed to live {SYMBOL} ticks.")


def stream_live_ticks():
    """Continuously collects live ticks and recovers lost data."""

    ws = None

    while True:

        try:

            # Connect
            ws = connect()

            # Find the last tick already saved
            last_timestamp = get_last_timestamp()

            # Recover anything missed before going live
            recover_missing_ticks(
                ws,
                last_timestamp
            )

            # Subscribe to live ticks
            subscribe_live_ticks(ws)

            # Receive live ticks
            while True:

                response = json.loads(ws.recv())

                if response.get("msg_type") != "tick":
                    continue

                tick = response["tick"]

                save_tick(tick)

                print(
                    f"Saved tick | "
                    f"{tick['symbol']} | "
                    f"{tick['quote']} | "
                    f"{tick['epoch']}"
                )

        except KeyboardInterrupt:

            print("\nCollector stopped by user.")

            if ws:
                ws.close()

            break

        except Exception as error:

            print("\nConnection lost.")
            print(f"Error: {error}")

            # Save whatever has already been written
            if ws:
                try:
                    ws.close()
                except:
                    pass

            print("Reconnecting in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    stream_live_ticks()