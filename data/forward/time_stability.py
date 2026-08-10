import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# V4 TIME STABILITY TEST
# ============================================================
CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)




FORWARD_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "forward",
    "r100_forward.csv"
)

# ============================================================
# FROZEN V4 THRESHOLDS
# ============================================================

LARGE_NEGATIVE_DELTA = -0.100000
LOW_VOLATILITY = 0.145358

TARGET_DIGIT = 3

BASELINE = 0.10

# Number of chronological blocks
BLOCK_SIZE = 5000


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading forward dataset...")
    print()

    if not os.path.isfile(FORWARD_FILE):

        print("ERROR:")
        print("Forward dataset not found:")
        print(FORWARD_FILE)

        return None

    df = pd.read_csv(FORWARD_FILE)

    print(
        f"Forward file: {FORWARD_FILE}"
    )

    print(
        f"Raw rows: {len(df)}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df["Last digit"] = pd.to_numeric(
        df["Last digit"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Calculate price delta
    # --------------------------------------------------------

    df["Price Delta"] = (
        df["price"].shift(-1)
        - df["price"]
    )

    # --------------------------------------------------------
    # Absolute delta
    # --------------------------------------------------------

    df["Absolute Delta"] = (
        df["Price Delta"].abs()
    )

    # --------------------------------------------------------
    # Step volatility 10
    # --------------------------------------------------------

    df["Step Volatility 10"] = (
        df["Price Delta"]
        .rolling(10)
        .std()
    )

    # --------------------------------------------------------
    # Target = NEXT DIGIT
    # --------------------------------------------------------

    df["Next Digit"] = (
        df["Last digit"].shift(-1)
    )

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Price Delta",
            "Step Volatility 10",
            "Next Digit"
        ]
    ).copy()

    df["Next Digit"] = (
        df["Next Digit"]
        .astype(int)
    )

    return df


# ============================================================
# TEST ONE BLOCK
# ============================================================

def test_block(df, block_number):

    observations = len(df)

    if observations == 0:
        return None

    # --------------------------------------------------------
    # H3 CONDITION
    #
    # Large Negative Delta
    # +
    # Low Volatility
    # --------------------------------------------------------

    condition = (
        (df["Price Delta"] <= LARGE_NEGATIVE_DELTA)
        &
        (df["Step Volatility 10"] <= LOW_VOLATILITY)
    )

    subset = df[condition]

    sample_size = len(subset)

    if sample_size == 0:

        return {
            "block": block_number,
            "observations": 0,
            "digit_3": 0,
            "probability": 0,
            "difference": 0,
            "p_value": None
        }

    # --------------------------------------------------------
    # Count digit 3
    # --------------------------------------------------------

    digit_3_count = (
        subset["Next Digit"]
        == TARGET_DIGIT
    ).sum()

    probability = (
        digit_3_count
        /
        sample_size
    )

    difference = (
        probability
        -
        BASELINE
    )

    # --------------------------------------------------------
    # One-sided binomial test
    #
    # H0: probability <= 10%
    # H1: probability > 10%
    # --------------------------------------------------------

    result = binomtest(
        digit_3_count,
        sample_size,
        BASELINE,
        alternative="greater"
    )

    p_value = result.pvalue

    return {
        "block": block_number,
        "observations": sample_size,
        "digit_3": digit_3_count,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V4 TIME STABILITY TEST")
    print("=" * 60)
    print()

    print(
        "Hypothesis:"
    )

    print(
        "Large Negative Delta + Low Volatility"
        " -> Next Digit 3"
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
        f"Block size: {BLOCK_SIZE} rows"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    if df is None:
        return

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(df)

    print(
        f"Usable rows: {len(df)}"
    )

    print()

    # --------------------------------------------------------
    # CREATE CHRONOLOGICAL BLOCKS
    # --------------------------------------------------------

    results = []

    total_rows = len(df)

    block_number = 1

    for start in range(
        0,
        total_rows,
        BLOCK_SIZE
    ):

        end = min(
            start + BLOCK_SIZE,
            total_rows
        )

        block = df.iloc[
            start:end
        ].copy()

        result = test_block(
            block,
            block_number
        )

        if result is not None:

            result["start_row"] = start
            result["end_row"] = end - 1

            results.append(result)

        block_number += 1

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TIME STABILITY RESULTS")
    print("=" * 60)
    print()

    print(
        "Block | Rows | Condition | Digit 3 | Probability | "
        "Difference | P-value"
    )

    print("-" * 90)

    for result in results:

        if result["observations"] == 0:

            print(
                f"{result['block']:5d} | "
                f"{result['observations']:4d} | "
                f"{'NONE':9s} | "
                f"{result['digit_3']:7d} | "
                f"{'N/A':11s} | "
                f"{'N/A':10s} | "
                f"{'N/A'}"
            )

            continue

        print(
            f"{result['block']:5d} | "
            f"{result['observations']:4d} | "
            f"{'H3':9s} | "
            f"{result['digit_3']:7d} | "
            f"{result['probability']:10.2%} | "
            f"{result['difference']:+9.2%} | "
            f"{result['p_value']:.6f}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    valid_results = [
        r for r in results
        if r["observations"] > 0
    ]

    print()
    print("=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)
    print()

    if not valid_results:

        print(
            "No usable H3 observations were found."
        )

        return

    probabilities = [
        r["probability"]
        for r in valid_results
    ]

    significant_blocks = [
        r for r in valid_results
        if r["p_value"] < 0.05
        and r["probability"] > BASELINE
    ]

    above_baseline = [
        r for r in valid_results
        if r["probability"] > BASELINE
    ]

    mean_probability = (
        sum(probabilities)
        /
        len(probabilities)
    )

    minimum_probability = min(
        probabilities
    )

    maximum_probability = max(
        probabilities
    )

    print(
        f"Blocks tested: "
        f"{len(valid_results)}"
    )

    print(
        f"Blocks above 10%: "
        f"{len(above_baseline)}/"
        f"{len(valid_results)}"
    )

    print(
        f"Statistically significant blocks: "
        f"{len(significant_blocks)}/"
        f"{len(valid_results)}"
    )

    print(
        f"Mean probability: "
        f"{mean_probability:.2%}"
    )

    print(
        f"Minimum probability: "
        f"{minimum_probability:.2%}"
    )

    print(
        f"Maximum probability: "
        f"{maximum_probability:.2%}"
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()

    proportion_above = (
        len(above_baseline)
        /
        len(valid_results)
    )

    if (
        proportion_above >= 0.70
        and len(significant_blocks) >= 2
    ):

        print(
            "H3 shows promising time stability."
        )

        print(
            f"{len(above_baseline)}/"
            f"{len(valid_results)} blocks were above "
            "the 10% baseline."
        )

        print(
            f"{len(significant_blocks)} blocks were "
            "individually statistically significant."
        )

    elif proportion_above >= 0.50:

        print(
            "H3 shows partial time stability."
        )

        print(
            "The effect appears in multiple periods, "
            "but is not consistently strong."
        )

    else:

        print(
            "H3 does NOT show strong time stability."
        )

        print(
            "The aggregate forward result may be "
            "driven by particular periods."
        )

    print()
    print(
        "Time stability test complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()