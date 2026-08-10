import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# V4 H3 ROLLING STABILITY TEST
# ============================================================
#
# Hypothesis:
# Large Negative Delta + Low Volatility -> Next Digit 3
#
# This test evaluates H3 chronologically in rolling blocks
# to determine whether the observed effect remains stable
# throughout the forward dataset.
# ============================================================


# ============================================================
# FIND PROJECT ROOT
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

ROLLING_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(ROLLING_DIR)
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

# This is the threshold discovered during V4 analysis.
LOW_VOLATILITY = 0.145358

# Number of H3 observations per rolling block.
BLOCK_SIZE = 500

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    ROLLING_DIR,
    "results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "h3_rolling_results.csv"
)


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
        print("ERROR:")
        print("Forward dataset not found.")
        print(FORWARD_FILE)

        sys.exit(1)

    df = pd.read_csv(FORWARD_FILE)

    print(
        f"Raw rows: {len(df)}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required_columns = [
        "time",
        "price",
        "Last digit"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print("ERROR:")
        print(
            f"Missing columns: {missing}"
        )

        sys.exit(1)

    df = df.copy()

    # Make sure data is chronological.
    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # Convert numeric columns safely.
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df["Last digit"] = pd.to_numeric(
        df["Last digit"],
        errors="coerce"
    )

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    # Price change from previous tick.
    df["price_delta"] = (
        df["price"].diff()
    )

    # Step volatility over the same V4 window.
    #
    # We use the standard deviation of recent price deltas.
    df["step_volatility"] = (
        df["price_delta"]
        .rolling(10)
        .std()
    )

    # The target is the NEXT digit.
    df["next_digit"] = (
        df["Last digit"].shift(-1)
    )

    # Remove rows where the required information
    # cannot be calculated.
    df = df.dropna(
        subset=[
            "price_delta",
            "step_volatility",
            "next_digit"
        ]
    ).copy()

    df["next_digit"] = (
        df["next_digit"].astype(int)
    )

    return df


# ============================================================
# APPLY H3 CONDITION
# ============================================================

def get_h3_rows(df):

    h3 = df[
        (df["price_delta"] <= LARGE_NEGATIVE_DELTA)
        &
        (df["step_volatility"] <= LOW_VOLATILITY)
    ].copy()

    return h3


# ============================================================
# RUN ROLLING TEST
# ============================================================

def run_rolling_test(h3):

    print()
    print("============================================================")
    print("V4 H3 ROLLING STABILITY TEST")
    print("============================================================")

    print()
    print("Hypothesis:")
    print(
        "Large Negative Delta + Low Volatility"
    )
    print(
        "-> Predict Next Digit 3"
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

    print(
        f"Rolling block size: "
        f"{BLOCK_SIZE} H3 observations"
    )

    print()
    print(
        f"Total H3 observations: "
        f"{len(h3)}"
    )

    if len(h3) < BLOCK_SIZE:

        print()
        print(
            "Not enough H3 observations "
            "for a rolling test."
        )

        return

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results = []

    block_number = 0

    # --------------------------------------------------------
    # Process H3 observations chronologically
    # --------------------------------------------------------

    for start in range(
        0,
        len(h3),
        BLOCK_SIZE
    ):

        block = h3.iloc[
            start:start + BLOCK_SIZE
        ]

        # Ignore incomplete final block.
        if len(block) < BLOCK_SIZE:
            break

        block_number += 1

        observations = len(block)

        correct = int(
            (
                block["next_digit"]
                == TARGET_DIGIT
            ).sum()
        )

        probability = (
            correct /
            observations
        )

        difference = (
            probability -
            BASELINE
        )

        test = binomtest(
            correct,
            observations,
            BASELINE,
            alternative="greater"
        )

        p_value = test.pvalue

        start_time = pd.to_datetime(
            block["time"].iloc[0],
            unit="s",
            utc=True
        ).tz_convert(
            "Africa/Nairobi"
        )

        end_time = pd.to_datetime(
            block["time"].iloc[-1],
            unit="s",
            utc=True
        ).tz_convert(
            "Africa/Nairobi"
        )

        result = {

            "block": block_number,

            "start_time": start_time,

            "end_time": end_time,

            "observations": observations,

            "correct": correct,

            "probability": probability,

            "difference": difference,

            "p_value": p_value

        }

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print(
        "## ROLLING BLOCK RESULTS"
    )

    print()
    print(
        "Block | Observations | Correct | "
        "Probability | Difference | P-value"
    )

    print(
        "------------------------------------------------------------"
    )

    for _, row in results_df.iterrows():

        print(
            f"{int(row['block']):5d} | "
            f"{int(row['observations']):12d} | "
            f"{int(row['correct']):7d} | "
            f"{row['probability'] * 100:10.2f}% | "
            f"{row['difference'] * 100:+10.2f}% | "
            f"{row['p_value']:.6f}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    probabilities = (
        results_df["probability"]
    )

    significant = (
        results_df["p_value"] < 0.05
    )

    above_baseline = (
        results_df["probability"]
        > BASELINE
    )

    print()
    print(
        "============================================================"
    )

    print(
        "## ROLLING SUMMARY"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"Rolling blocks tested: "
        f"{len(results_df)}"
    )

    print(
        f"Blocks above 10%: "
        f"{above_baseline.sum()}/"
        f"{len(results_df)}"
    )

    print(
        f"Statistically significant blocks: "
        f"{significant.sum()}/"
        f"{len(results_df)}"
    )

    print(
        f"Mean probability: "
        f"{probabilities.mean() * 100:.2f}%"
    )

    print(
        f"Minimum probability: "
        f"{probabilities.min() * 100:.2f}%"
    )

    print(
        f"Maximum probability: "
        f"{probabilities.max() * 100:.2f}%"
    )

    print(
        f"Median probability: "
        f"{probabilities.median() * 100:.2f}%"
    )

    # ========================================================
    # FIRST VS LAST BLOCK
    # ========================================================

    first_probability = (
        results_df["probability"].iloc[0]
    )

    last_probability = (
        results_df["probability"].iloc[-1]
    )

    change = (
        last_probability -
        first_probability
    )

    print()

    print(
        f"First block probability: "
        f"{first_probability * 100:.2f}%"
    )

    print(
        f"Last block probability: "
        f"{last_probability * 100:.2f}%"
    )

    print(
        f"First-to-last change: "
        f"{change * 100:+.2f}%"
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    print()
    print(
        "## CONCLUSION"
    )

    print()

    significant_count = int(
        significant.sum()
    )

    above_count = int(
        above_baseline.sum()
    )

    total_blocks = len(
        results_df
    )

    if (
        significant_count >= 2
        and
        above_count >= total_blocks * 0.60
        and
        probabilities.mean() > BASELINE
    ):

        print(
            "H3 shows strong rolling stability."
        )

        print(
            "The effect remains above baseline "
            "across most rolling blocks."
        )

    elif (
        above_count >= total_blocks * 0.50
        and
        probabilities.mean() > BASELINE
    ):

        print(
            "H3 shows partial rolling stability."
        )

        print(
            "The effect remains above baseline "
            "in multiple rolling periods, "
            "but the strength is inconsistent."
        )

    else:

        print(
            "H3 does NOT show strong rolling stability."
        )

        print(
            "The observed advantage may be "
            "concentrated in particular periods."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print(
        "Rolling results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()
    print(
        "Rolling stability test complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_data(
        df
    )

    print(
        f"Usable rows: {len(df)}"
    )

    h3 = get_h3_rows(
        df
    )

    print()
    print(
        f"H3 observations found: "
        f"{len(h3)}"
    )

    run_rolling_test(
        h3
    )


if __name__ == "__main__":

    main()