import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

V4_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(V4_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pandas as pd
from scipy.stats import binomtest

from config import FEATURE_V4_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(
        FEATURE_V4_FILE
    )


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    required = [
        "Price Delta",
        "Round Distance",
        "Step Volatility 10",
        "Next Digit"
    ]

    df = df.dropna(
        subset=required
    ).copy()

    df["Next Digit"] = (
        df["Next Digit"]
        .astype(int)
    )

    return df


# --------------------------------------------------
# TEST HYPOTHESIS
# --------------------------------------------------

def test_hypothesis(
    name,
    df,
    target_digit
):

    total = len(df)

    if total == 0:

        print(
            f"\n{name}"
        )

        print(
            "No observations."
        )

        return

    correct = (
        df["Next Digit"] ==
        target_digit
    ).sum()

    accuracy = (
        correct / total
    )

    baseline = 0.10

    difference = (
        accuracy - baseline
    )

    test = binomtest(
        k=int(correct),
        n=total,
        p=baseline,
        alternative="greater"
    )

    print(
        f"\n{name}"
    )

    print("\n---")

    print(
        f"Target digit: {target_digit}"
    )

    print(
        f"Observations: {total}"
    )

    print(
        f"Digit {target_digit} occurrences: "
        f"{correct}"
    )

    print(
        f"Observed probability: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Baseline: "
        f"{baseline * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference * 100:.2f}%"
    )

    print(
        f"P-value: "
        f"{test.pvalue:.6f}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(
        f"Total rows: {len(df)}"
    )

    df = prepare_data(df)

    print(
        f"Usable rows: {len(df)}"
    )

    # --------------------------------------------------
    # DISCOVERY / CONFIRMATION SPLIT
    # --------------------------------------------------

    split_index = int(
        len(df) * 0.70
    )

    discovery = df.iloc[
        :split_index
    ].copy()

    confirmation = df.iloc[
        split_index:
    ].copy()

    print(
        f"\nDiscovery rows: "
        f"{len(discovery)}"
    )

    print(
        f"Confirmation rows: "
        f"{len(confirmation)}"
    )

    print(
        "\n## V4 CONFIRMATION VALIDATION"
    )

    print("\n---")

    print(
        "\nThe following hypotheses were "
        "selected BEFORE confirmation testing."
    )

    # --------------------------------------------------
    # HYPOTHESIS 1
    #
    # Near whole number -> Digit 4
    # --------------------------------------------------

    test_hypothesis(
        "HYPOTHESIS 1: Near Whole Number -> Digit 4",

        confirmation[
            confirmation["Round Distance"] <= 0.10
        ],

        target_digit=4
    )

    # --------------------------------------------------
    # HYPOTHESIS 2
    #
    # Large positive delta + low volatility
    # -> Digit 9
    # --------------------------------------------------

    # Thresholds are calculated ONLY from
    # discovery data.

    positive_delta_threshold = (
        discovery["Price Delta"]
        .quantile(0.75)
    )

    volatility_low_threshold = (
        discovery["Step Volatility 10"]
        .quantile(0.25)
    )

    test_hypothesis(
        "HYPOTHESIS 2: Large Positive Delta + Low Volatility -> Digit 9",

        confirmation[
            (confirmation["Price Delta"] >
             positive_delta_threshold)
            &
            (confirmation["Step Volatility 10"] <=
             volatility_low_threshold)
        ],

        target_digit=9
    )

    # --------------------------------------------------
    # HYPOTHESIS 3
    #
    # Large negative delta + low volatility
    # -> Digit 3
    # --------------------------------------------------

    negative_delta_threshold = (
        discovery["Price Delta"]
        .quantile(0.25)
    )

    test_hypothesis(
        "HYPOTHESIS 3: Large Negative Delta + Low Volatility -> Digit 3",

        confirmation[
            (confirmation["Price Delta"] <
             negative_delta_threshold)
            &
            (confirmation["Step Volatility 10"] <=
             volatility_low_threshold)
        ],

        target_digit=3
    )

    # --------------------------------------------------
    # BASELINE DISTRIBUTION
    # --------------------------------------------------

    print(
        "\n## CONFIRMATION DIGIT DISTRIBUTION"
    )

    print("\n---")

    counts = (
        confirmation["Next Digit"]
        .value_counts()
        .reindex(
            range(10),
            fill_value=0
        )
    )

    total = len(
        confirmation
    )

    for digit in range(10):

        percentage = (
            counts[digit] /
            total *
            100
        )

        print(
            f"Digit {digit}: "
            f"{counts[digit]} "
            f"({percentage:.2f}%)"
        )

    print(
        "\n## CONFIRMATION COMPLETE"
    )

    print("\n---")

    print(
        "The hypotheses were frozen "
        "before testing the confirmation period."
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()