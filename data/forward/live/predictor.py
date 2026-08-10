import os
import sys
import json
import requests
import websocket
import pandas as pd
from datetime import datetime, timezone



# ============================================================
# H3 LIVE PAPER PREDICTOR
# ============================================================
#
# FROZEN HYPOTHESIS:
#
# Large Negative Delta
# + Low Volatility
# + Previous Digit 4
# -> Predict Next Digit 3
#
# PAPER TESTING ONLY
# NO TRADES ARE PLACED
# ============================================================


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

LIVE_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(LIVE_DIR)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add src to Python's import path
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import project configuration
from config import APP_ID, PATAPI, CLIENTid
# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "R_100"

DELTA_THRESHOLD = -0.100
VOLATILITY_THRESHOLD = 0.145358

PREVIOUS_DIGIT = 4
TARGET_DIGIT = 3

BASELINE = 0.10

VOLATILITY_WINDOW = 20



RESULTS_FILE = os.path.join(
    LIVE_DIR,
    "prediction_results.csv"
)


# ============================================================
# GET OTP
# ============================================================

def get_websocket_url():
    """
    Obtain an authenticated Deriv WebSocket URL.
    Uses the same credentials as the working collector.
    """

    url = (
        f"https://api.derivws.com/trading/v1/options/"
        f"accounts/{CLIENTid}/otp"
    )

    headers = {
        "Deriv-App-ID": str(APP_ID),
        "Authorization": f"Bearer {PATAPI}",
        "Content-Type": "application/json"
    }

    print("Getting WebSocket URL...")
    print("APP_ID:", APP_ID)
    print("CLIENTid:", CLIENTid)
    print("PATAPI loaded:", bool(PATAPI))
    response = requests.post(
        url,
        headers=headers,
        timeout=30
    )

    print(response.status_code)
    print(response.text)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to obtain Deriv OTP: "
            f"{response.status_code} {response.text}"
        )

    data = response.json()

    if "data" not in data or "url" not in data["data"]:
        raise RuntimeError(
            f"Invalid OTP response: {data}"
        )

    ws_url = data["data"]["url"]

    print("URL obtained.")

    return ws_url


# ============================================================
# DIGIT EXTRACTION
# ============================================================

def get_last_digit(price):

    formatted = f"{float(price):.2f}"

    return int(formatted[-1])


# ============================================================
# VOLATILITY
# ============================================================

def calculate_volatility(prices):

    if len(prices) < VOLATILITY_WINDOW:

        return None

    series = pd.Series(
        prices[-VOLATILITY_WINDOW:]
    )

    return float(
        series.diff().std()
    )


# ============================================================
# SAVE RESULT
# ============================================================

def save_prediction(record):

    df = pd.DataFrame([record])

    file_exists = os.path.isfile(
        RESULTS_FILE
    )

    df.to_csv(
        RESULTS_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


# ============================================================
# EXISTING RESULTS
# ============================================================

def load_existing_stats():

    if not os.path.isfile(
        RESULTS_FILE
    ):

        return 0, 0

    try:

        df = pd.read_csv(
            RESULTS_FILE
        )

        if "correct" not in df.columns:

            return 0, 0

        valid = df["correct"].dropna()

        total = len(valid)

        correct = int(
            valid.sum()
        )

        return total, correct

    except Exception:

        return 0, 0


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("H3 LIVE PAPER PREDICTOR")
    print("=" * 70)

    print()
    print("Frozen hypothesis:")
    print()
    print("Large Negative Delta")
    print("+ Low Volatility")
    print("+ Previous Digit 4")
    print("-> Predict Next Digit 3")

    print()
    print(f"Symbol: {SYMBOL}")

    print(
        f"Delta threshold: "
        f"<= {DELTA_THRESHOLD}"
    )

    print(
        f"Volatility threshold: "
        f"<= {VOLATILITY_THRESHOLD}"
    )

    print(
        f"Previous digit: "
        f"{PREVIOUS_DIGIT}"
    )

    print(
        f"Target digit: "
        f"{TARGET_DIGIT}"
    )

    print(
        f"Baseline: "
        f"{BASELINE:.2%}"
    )

    print()
    print(
        f"Results file:\n"
        f"{RESULTS_FILE}"
    )


    # ========================================================
    # EXISTING RESULTS
    # ========================================================

    total_predictions, total_correct = (
        load_existing_stats()
    )

    print()
    print("Existing paper results:")
    print(
        f"Predictions: "
        f"{total_predictions}"
    )

    print(
        f"Correct: "
        f"{total_correct}"
    )

    if total_predictions > 0:

        accuracy = (
            total_correct /
            total_predictions
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"Difference from baseline: "
            f"{accuracy - BASELINE:+.2%}"
        )


    # ========================================================
    # GET WEBSOCKET URL
    # ========================================================

    ws_url = get_websocket_url()


    # ========================================================
    # CONNECT
    # ========================================================

    print()
    print("Connecting...")

    try:

        ws = websocket.create_connection(
            ws_url,
            timeout=30
        )

    except Exception as e:

        raise RuntimeError(
            f"WebSocket connection failed: {e}"
        )

    print("Connected to Deriv.")


    # ========================================================
    # SUBSCRIBE
    # ========================================================

    request = {
        "ticks": SYMBOL,
        "subscribe": 1
    }

    print()
    print("Request:")
    print(request)

    ws.send(
        json.dumps(request)
    )

    response = json.loads(
        ws.recv()
    )

    print()
    print("Response:")
    print(response)


    if response.get("msg_type") != "tick":

        print()
        print(
            "Warning: subscription response "
            "was not a tick."
        )


    # ========================================================
    # STATE
    # ========================================================

    prices = []

    previous_price = None
    previous_digit = None

    pending_prediction = None

    tick_count = 0
    signal_count = 0


    # ========================================================
    # LIVE LOOP
    # ========================================================

    print()
    print("=" * 70)
    print("LISTENING FOR LIVE TICKS")
    print("PAPER TESTING ONLY")
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    print()


    try:

        while True:

            message = ws.recv()

            if not message:

                continue

            data = json.loads(
                message
            )

            if data.get(
                "msg_type"
            ) != "tick":

                continue

            tick = data.get(
                "tick"
            )

            if not tick:

                continue


            # ------------------------------------------------
            # CURRENT TICK
            # ------------------------------------------------

            price = float(
                tick["quote"]
            )

            epoch = int(
                tick["epoch"]
            )

            symbol = tick[
                "symbol"
            ]

            current_digit = (
                get_last_digit(price)
            )

            timestamp = (
                datetime.fromtimestamp(
                    epoch,
                    tz=timezone.utc
                ).astimezone()
            )

            tick_count += 1

            prices.append(
                price
            )


            # ------------------------------------------------
            # EVALUATE PREVIOUS PREDICTION
            # ------------------------------------------------

            if pending_prediction is not None:

                predicted_digit = (
                    pending_prediction[
                        "predicted_digit"
                    ]
                )

                actual_digit = (
                    current_digit
                )

                correct = (
                    actual_digit ==
                    predicted_digit
                )

                record = {

                    "signal_timestamp":
                        pending_prediction[
                            "signal_timestamp"
                        ],

                    "result_timestamp":
                        timestamp.isoformat(),

                    "symbol":
                        symbol,

                    "signal_price":
                        pending_prediction[
                            "signal_price"
                        ],

                    "result_price":
                        price,

                    "previous_digit":
                        pending_prediction[
                            "previous_digit"
                        ],

                    "predicted_digit":
                        predicted_digit,

                    "actual_digit":
                        actual_digit,

                    "delta":
                        pending_prediction[
                            "delta"
                        ],

                    "volatility":
                        pending_prediction[
                            "volatility"
                        ],

                    "correct":
                        int(correct)
                }


                save_prediction(
                    record
                )


                total_predictions += 1

                if correct:

                    total_correct += 1


                accuracy = (
                    total_correct /
                    total_predictions
                )


                result = (
                    "WIN"
                    if correct
                    else "LOSS"
                )


                print()
                print("-" * 70)

                print(
                    f"{result}"
                )

                print(
                    f"Predicted digit: "
                    f"{predicted_digit}"
                )

                print(
                    f"Actual digit: "
                    f"{actual_digit}"
                )

                print(
                    f"Signal price: "
                    f"{pending_prediction['signal_price']:.2f}"
                )

                print(
                    f"Result price: "
                    f"{price:.2f}"
                )

                print(
                    f"Paper predictions: "
                    f"{total_predictions}"
                )

                print(
                    f"Correct: "
                    f"{total_correct}"
                )

                print(
                    f"Paper accuracy: "
                    f"{accuracy:.2%}"
                )

                print(
                    f"Baseline: "
                    f"{BASELINE:.2%}"
                )

                print(
                    f"Difference: "
                    f"{accuracy - BASELINE:+.2%}"
                )

                print("-" * 70)


                pending_prediction = None


            # ------------------------------------------------
            # FIRST TICK
            # ------------------------------------------------

            if previous_price is None:

                previous_price = price

                previous_digit = (
                    current_digit
                )

                continue


            # ------------------------------------------------
            # DELTA
            # ------------------------------------------------

            delta = (
                price -
                previous_price
            )


            # ------------------------------------------------
            # VOLATILITY
            # ------------------------------------------------

            volatility = (
                calculate_volatility(
                    prices
                )
            )


            # ------------------------------------------------
            # H3
            # ------------------------------------------------

            h3 = False

            if volatility is not None:

                h3 = (
                    delta <=
                    DELTA_THRESHOLD
                    and
                    volatility <=
                    VOLATILITY_THRESHOLD
                )


            # ------------------------------------------------
            # CONDITIONAL H3
            # ------------------------------------------------

            conditional_h3 = (

                h3

                and

                previous_digit ==
                PREVIOUS_DIGIT

            )


            # ------------------------------------------------
            # CREATE PAPER PREDICTION
            # ------------------------------------------------

            if conditional_h3:

                signal_count += 1

                pending_prediction = {

                    "predicted_digit":
                        TARGET_DIGIT,

                    "signal_timestamp":
                        timestamp.isoformat(),

                    "signal_price":
                        price,

                    "previous_digit":
                        previous_digit,

                    "delta":
                        delta,

                    "volatility":
                        volatility
                }


                print()
                print()
                print(
                    ">>> CONDITIONAL H3 SIGNAL <<<"
                )

                print(
                    f"Time: {timestamp}"
                )

                print(
                    f"Price: {price:.2f}"
                )

                print(
                    f"Previous digit: "
                    f"{previous_digit}"
                )

                print(
                    f"Delta: "
                    f"{delta:.6f}"
                )

                print(
                    f"Volatility: "
                    f"{volatility:.6f}"
                )

                print(
                    f"PREDICTION -> "
                    f"NEXT DIGIT = "
                    f"{TARGET_DIGIT}"
                )

                print(
                    f"Conditional signals: "
                    f"{signal_count}"
                )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if tick_count % 100 == 0:

                if total_predictions > 0:

                    accuracy = (
                        total_correct /
                        total_predictions
                    )

                    print(
                        f"[STATUS] "
                        f"Ticks={tick_count} | "
                        f"Signals={signal_count} | "
                        f"Predictions={total_predictions} | "
                        f"Accuracy={accuracy:.2%}"
                    )

                else:

                    print(
                        f"[STATUS] "
                        f"Ticks={tick_count} | "
                        f"Signals={signal_count} | "
                        f"Predictions=0"
                    )


            # ------------------------------------------------
            # UPDATE STATE
            # ------------------------------------------------

            previous_price = price

            previous_digit = (
                current_digit
            )


    except KeyboardInterrupt:

        print()
        print()
        print("=" * 70)
        print("PREDICTOR STOPPED")
        print("=" * 70)

        print(
            f"Ticks processed: "
            f"{tick_count}"
        )

        print(
            f"Conditional H3 signals: "
            f"{signal_count}"
        )

        print(
            f"Paper predictions: "
            f"{total_predictions}"
        )

        print(
            f"Correct predictions: "
            f"{total_correct}"
        )

        if total_predictions > 0:

            accuracy = (
                total_correct /
                total_predictions
            )

            print(
                f"Paper accuracy: "
                f"{accuracy:.2%}"
            )

            print(
                f"Baseline: "
                f"{BASELINE:.2%}"
            )

            print(
                f"Difference: "
                f"{accuracy - BASELINE:+.2%}"
            )

        print()
        print(
            f"Results saved to:\n"
            f"{RESULTS_FILE}"
        )

        print("=" * 70)


    finally:

        try:

            ws.close()

        except Exception:

            pass


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()