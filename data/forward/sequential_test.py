import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# V4 SEQUENTIAL TEST
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

# ============================================================
# SETTINGS
# ============================================================

TARGET_DIGIT = 3
BASELINE = 0.10

LARGE_NEGATIVE_DELTA = -0.10
LOW_VOLATILITY = 0.145358

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

# Print progress every N H3 observations
PROGRESS_INTERVAL = 500

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("Loading forward dataset...")
    print()

    print(
        f"Forward file: {FORWARD_FILE}"
    )

    if not os.path.isfile(FORWARD_FILE):

        print()
        print("ERROR: Forward dataset not found.")
        print(FORWARD_FILE)
        sys.exit(1)

    df = pd.read_csv(FORWARD_FILE)

    print(
        f"Raw rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Convert required columns
    # --------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    df["Last digit"] = pd.to_numeric(
        df["Last digit"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "price",
            "time",
            "Last digit"
        ]
    ).copy()

    df["Last digit"] = (
        df["Last digit"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    print(
        f"Usable rows: {len(df)}"
    )

    return df


# ============================================================
# CREATE V4 FEATURES
# ============================================================

def create_features(df):

    # --------------------------------------------------------
    # Price Delta
    # --------------------------------------------------------

    df["Price Delta"] = (
        df["price"].diff()
    )

    # --------------------------------------------------------
    # Round Distance
    # --------------------------------------------------------

    df["Round Distance"] = (
        (df["price"] - df["price"].round())
        .abs()
    )

    # --------------------------------------------------------
    # Step Volatility 10
    # --------------------------------------------------------

    df["Step Volatility 10"] = (
        df["Price Delta"]
        .rolling(10)
        .std()
    )

    return df


# ============================================================
# SEQUENTIAL TEST
# ============================================================

def run_sequential_test(df):

    print()
    print("============================================================")
    print("V4 SEQUENTIAL H3 TEST")
    print("============================================================")
    print()

    print(
        "Hypothesis:"
    )

    print(
        "Large Negative Delta + Low Volatility"
    )

    print(
        "→ Predict Next Digit 3"
    )

    print()

    print(
        f"Large negative delta <= "
        f"{LARGE_NEGATIVE_DELTA:.6f}"
    )

    print(
        f"Low volatility <= "
        f"{LOW_VOLATILITY:.6f}"
    )

    print(
        f"Target digit: {TARGET_DIGIT}"
    )

    print(
        f"Baseline: {BASELINE:.2%}"
    )

    print()
    print("Processing data chronologically...")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    h3_count = 0
    correct_predictions = 0

    longest_win_streak = 0
    longest_loss_streak = 0

    current_win_streak = 0
    current_loss_streak = 0

    # --------------------------------------------------------
    # Store sequential results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We stop at len(df)-1 because we need the NEXT tick
    # to determine whether the prediction was correct.
    # --------------------------------------------------------

    for i in range(1, len(df) - 1):

        current_delta = df.loc[
            i,
            "Price Delta"
        ]

        current_volatility = df.loc[
            i,
            "Step Volatility 10"
        ]

        # ----------------------------------------------------
        # Skip unavailable feature values
        # ----------------------------------------------------

        if pd.isna(current_delta):
            continue

        if pd.isna(current_volatility):
            continue

        # ----------------------------------------------------
        # H3 CONDITION
        # ----------------------------------------------------

        h3 = (
            current_delta <= LARGE_NEGATIVE_DELTA
            and
            current_volatility <= LOW_VOLATILITY
        )

        if not h3:
            continue

        # ----------------------------------------------------
        # H3 occurred
        # ----------------------------------------------------

        h3_count += 1

        # The NEXT tick is the outcome
        next_digit = int(
            df.loc[
                i + 1,
                "Last digit"
            ]
        )

        prediction_correct = (
            next_digit == TARGET_DIGIT
        )

        if prediction_correct:

            correct_predictions += 1

            current_win_streak += 1
            current_loss_streak = 0

            if current_win_streak > longest_win_streak:
                longest_win_streak = current_win_streak

        else:

            current_loss_streak += 1
            current_win_streak = 0

            if current_loss_streak > longest_loss_streak:
                longest_loss_streak = current_loss_streak

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "trigger_index": i,
                "trigger_time": df.loc[i, "time"],
                "price": df.loc[i, "price"],
                "price_delta": current_delta,
                "volatility": current_volatility,
                "prediction": TARGET_DIGIT,
                "actual_digit": next_digit,
                "correct": prediction_correct
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if h3_count % PROGRESS_INTERVAL == 0:

            accuracy = (
                correct_predictions / h3_count
            )

            print(
                f"H3 observations: {h3_count} | "
                f"Correct: {correct_predictions} | "
                f"Accuracy: {accuracy:.2%}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("============================================================")
    print("SEQUENTIAL TEST RESULTS")
    print("============================================================")
    print()

    if h3_count == 0:

        print("No H3 conditions were found.")
        return

    accuracy = (
        correct_predictions / h3_count
    )

    difference = (
        accuracy - BASELINE
    )

    # --------------------------------------------------------
    # Binomial test
    # --------------------------------------------------------

    test = binomtest(
        correct_predictions,
        h3_count,
        BASELINE,
        alternative="greater"
    )

    p_value = test.pvalue

    expected_correct = (
        h3_count * BASELINE
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        f"H3 observations: {h3_count}"
    )

    print(
        f"Correct predictions: "
        f"{correct_predictions}"
    )

    print(
        f"Incorrect predictions: "
        f"{h3_count - correct_predictions}"
    )

    print()

    print(
        f"Observed accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Baseline accuracy: "
        f"{BASELINE:.2%}"
    )

    print(
        f"Difference: "
        f"{difference:+.2%}"
    )

    print()

    print(
        f"Expected correct at baseline: "
        f"{expected_correct:.1f}"
    )

    print(
        f"Observed advantage: "
        f"{correct_predictions - expected_correct:+.1f} predictions"
    )

    print()

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    print()

    print(
        f"Longest winning streak: "
        f"{longest_win_streak}"
    )

    print(
        f"Longest losing streak: "
        f"{longest_loss_streak}"
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print()
    print("## CONCLUSION")
    print()

    if accuracy > BASELINE and p_value < 0.05:

        print(
            "H3 shows statistically significant "
            "sequential predictive evidence."
        )

        print(
            "The next digit 3 occurs above the "
            "10% baseline when H3 is triggered."
        )

    elif accuracy > BASELINE:

        print(
            "H3 performs above the 10% baseline, "
            "but the evidence is not statistically significant."
        )

        print(
            "More forward data is required."
        )

    else:

        print(
            "H3 does not currently outperform "
            "the 10% baseline sequentially."
        )

    # ========================================================
    # SAVE SEQUENTIAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    output_file = os.path.join(
        FORWARD_DIR,
        "h3_sequential_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Sequential results saved to:"
    )

    print(
        output_file
    )

    print()
    print("Sequential test complete.")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = create_features(df)

    run_sequential_test(df)


if __name__ == "__main__":
    main()