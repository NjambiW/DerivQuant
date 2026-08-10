import sys
import os
import pandas as pd
from scipy.stats import binomtest


# ============================================================
# V4 FORWARD VALIDATION
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
# SETTINGS
# ============================================================

BASELINE = 0.10

# These thresholds match the V4 discovery analysis.
# We keep them frozen so we don't move the goalposts.
DELTA_QUANTILE = 0.75
VOLATILITY_QUANTILE = 0.50

ROUND_DISTANCE_THRESHOLD = 0.10


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.isfile(FORWARD_FILE):

        raise FileNotFoundError(
            f"\nForward dataset not found:\n"
            f"{FORWARD_FILE}"
        )

    df = pd.read_csv(FORWARD_FILE)

    print(f"Forward file: {FORWARD_FILE}")
    print(f"Raw rows: {len(df)}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Make sure price is numeric
    # --------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Calculate price delta
    # --------------------------------------------------------

    df["Price Delta"] = (
        df["price"].diff()
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
    # Round distance
    #
    # Distance from current price to nearest whole number
    # --------------------------------------------------------

    df["Round Distance"] = (
        df["price"] - df["price"].round()
    ).abs()

    # --------------------------------------------------------
    # Next digit
    #
    # IMPORTANT:
    # We predict the NEXT tick's digit.
    # --------------------------------------------------------

    df["Next Digit"] = (
        df["Last digit"].shift(-1)
    )

    # --------------------------------------------------------
    # Remove rows that cannot be evaluated
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "price",
            "Price Delta",
            "Step Volatility 10",
            "Round Distance",
            "Next Digit"
        ]
    )

    df["Next Digit"] = (
        df["Next Digit"].astype(int)
    )

    return df


# ============================================================
# CREATE FROZEN CONDITIONS
# ============================================================

def create_conditions(df):

    df = df.copy()

    # --------------------------------------------------------
    # Determine thresholds from THIS forward dataset
    #
    # These are descriptive thresholds, not learned targets.
    # --------------------------------------------------------

    large_positive_threshold = (
        df["Price Delta"]
        .quantile(DELTA_QUANTILE)
    )

    large_negative_threshold = (
        df["Price Delta"]
        .quantile(1 - DELTA_QUANTILE)
    )

    low_volatility_threshold = (
        df["Step Volatility 10"]
        .quantile(VOLATILITY_QUANTILE)
    )

    print()
    print("## FORWARD FEATURE THRESHOLDS")
    print()
    print(
        f"Large positive delta >= "
        f"{large_positive_threshold:.6f}"
    )

    print(
        f"Large negative delta <= "
        f"{large_negative_threshold:.6f}"
    )

    print(
        f"Low volatility <= "
        f"{low_volatility_threshold:.6f}"
    )

    # --------------------------------------------------------
    # HYPOTHESIS 1
    #
    # Near whole number -> digit 4
    # --------------------------------------------------------

    df["H1_Near_Whole"] = (
        df["Round Distance"]
        <= ROUND_DISTANCE_THRESHOLD
    )

    # --------------------------------------------------------
    # HYPOTHESIS 2
    #
    # Large positive delta + low volatility -> digit 9
    # --------------------------------------------------------

    df["H2_Positive_LowVol"] = (
        (df["Price Delta"] >= large_positive_threshold)
        &
        (df["Step Volatility 10"] <= low_volatility_threshold)
    )

    # --------------------------------------------------------
    # HYPOTHESIS 3
    #
    # Large negative delta + low volatility -> digit 3
    # --------------------------------------------------------

    df["H3_Negative_LowVol"] = (
        (df["Price Delta"] <= large_negative_threshold)
        &
        (df["Step Volatility 10"] <= low_volatility_threshold)
    )

    return df


# ============================================================
# TEST ONE HYPOTHESIS
# ============================================================

def test_hypothesis(
    df,
    condition_column,
    target_digit,
    hypothesis_name
):

    subset = df[
        df[condition_column]
    ]

    observations = len(subset)

    if observations == 0:

        print()
        print(hypothesis_name)
        print("No observations found.")

        return None

    occurrences = (
        subset["Next Digit"] == target_digit
    ).sum()

    probability = (
        occurrences / observations
    )

    difference = (
        probability - BASELINE
    )

    # --------------------------------------------------------
    # One-sided binomial test
    #
    # H0: p = 10%
    # H1: p > 10%
    # --------------------------------------------------------

    result = binomtest(
        occurrences,
        observations,
        BASELINE,
        alternative="greater"
    )

    p_value = result.pvalue

    print()
    print("=" * 60)
    print(hypothesis_name)
    print("=" * 60)

    print(
        f"Target digit: {target_digit}"
    )

    print(
        f"Observations: {observations}"
    )

    print(
        f"Occurrences: {occurrences}"
    )

    print(
        f"Observed probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Baseline: "
        f"{BASELINE * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference * 100:.2f}%"
    )

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    if p_value < 0.05:

        print()
        print(
            "RESULT: STATISTICALLY SIGNIFICANT"
        )

    else:

        print()
        print(
            "RESULT: NOT STATISTICALLY SIGNIFICANT"
        )

    return {
        "hypothesis": hypothesis_name,
        "observations": observations,
        "occurrences": occurrences,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }


# ============================================================
# FULL DIGIT DISTRIBUTION
# ============================================================

def digit_distribution(df):

    print()
    print("=" * 60)
    print("FORWARD DATA OVERALL DIGIT DISTRIBUTION")
    print("=" * 60)

    counts = (
        df["Next Digit"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    for digit in range(10):

        count = counts.get(
            digit,
            0
        )

        percentage = (
            count / total * 100
        )

        print(
            f"Digit {digit}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V4 FORWARD VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_features(df)

    print(
        f"Usable rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Create conditions
    # --------------------------------------------------------

    df = create_conditions(df)

    # --------------------------------------------------------
    # Overall distribution
    # --------------------------------------------------------

    digit_distribution(df)

    # --------------------------------------------------------
    # Test hypotheses
    # --------------------------------------------------------

    results = []

    # H1
    result = test_hypothesis(
        df,
        "H1_Near_Whole",
        4,
        "HYPOTHESIS 1: Near Whole Number -> Digit 4"
    )

    if result:
        results.append(result)

    # H2
    result = test_hypothesis(
        df,
        "H2_Positive_LowVol",
        9,
        "HYPOTHESIS 2: Large Positive Delta + Low Volatility -> Digit 9"
    )

    if result:
        results.append(result)

    # H3
    result = test_hypothesis(
        df,
        "H3_Negative_LowVol",
        3,
        "HYPOTHESIS 3: Large Negative Delta + Low Volatility -> Digit 3"
    )

    if result:
        results.append(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print()
    print("=" * 60)
    print("V4 FORWARD VALIDATION SUMMARY")
    print("=" * 60)

    print()

    for result in results:

        print(
            f"{result['hypothesis']}"
        )

        print(
            f"Observations: "
            f"{result['observations']}"
        )

        print(
            f"Probability: "
            f"{result['probability'] * 100:.2f}%"
        )

        print(
            f"Difference: "
            f"{result['difference'] * 100:.2f}%"
        )

        print(
            f"P-value: "
            f"{result['p_value']:.6f}"
        )

        print()

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    significant = [
        r for r in results
        if r["p_value"] < 0.05
    ]

    print("=" * 60)
    print("FINAL INTERPRETATION")
    print("=" * 60)

    print()

    if not significant:

        print(
            "None of the frozen V4 hypotheses "
            "show statistically significant evidence "
            "of an edge in the forward dataset."
        )

        print()
        print(
            "The V4 relationships did not "
            "survive forward validation."
        )

    else:

        print(
            f"{len(significant)} hypothesis(es) "
            "show statistically significant evidence "
            "above the 10% baseline:"
        )

        print()

        for result in significant:

            print(
                f"- {result['hypothesis']}"
            )

            print(
                f"  Probability: "
                f"{result['probability'] * 100:.2f}%"
            )

            print(
                f"  P-value: "
                f"{result['p_value']:.6f}"
            )

    print()
    print("=" * 60)
    print("FORWARD VALIDATION COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()