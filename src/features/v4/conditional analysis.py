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
import numpy as np

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

    features = [
        "Price Delta",
        "Round Distance",
        "Step Volatility 10"
    ]

    df = df.dropna(
        subset=features + ["Next Digit"]
    ).copy()

    df["Next Digit"] = (
        df["Next Digit"]
        .astype(int)
    )

    return df


# --------------------------------------------------
# DIGIT DISTRIBUTION
# --------------------------------------------------

def digit_distribution(df):

    counts = (
        df["Next Digit"]
        .value_counts()
        .reindex(range(10), fill_value=0)
    )

    total = counts.sum()

    percentages = (
        counts / total * 100
    )

    result = pd.DataFrame({
        "Digit": range(10),
        "Count": counts.values,
        "Percentage": percentages.values
    })

    return result


# --------------------------------------------------
# TEST CONDITION
# --------------------------------------------------

def analyze_condition(
    name,
    condition_df
):

    total = len(condition_df)

    if total < 100:

        return

    counts = (
        condition_df["Next Digit"]
        .value_counts()
        .reindex(
            range(10),
            fill_value=0
        )
    )

    percentages = (
        counts / total * 100
    )

    highest_digit = (
        percentages
        .idxmax()
    )

    highest_percentage = (
        percentages.max()
    )

    lowest_digit = (
        percentages
        .idxmin()
    )

    lowest_percentage = (
        percentages.min()
    )

    # Binomial test for highest digit
    # against 10% baseline
    highest_count = (
        counts.loc[highest_digit]
    )

    test = binomtest(
        k=int(highest_count),
        n=total,
        p=0.10,
        alternative="greater"
    )

    p_value = test.pvalue

    print(
        f"\n{name}"
    )

    print(
        f"Observations: {total}"
    )

    print(
        f"Most common next digit: "
        f"{highest_digit}"
    )

    print(
        f"Highest probability: "
        f"{highest_percentage:.2f}%"
    )

    print(
        f"Lowest probability: "
        f"{lowest_percentage:.2f}%"
    )

    print(
        f"Highest digit count: "
        f"{highest_count}"
    )

    print(
        f"P-value vs 10%: "
        f"{p_value:.6f}"
    )

    print(
        "Distribution:"
    )

    for digit in range(10):

        print(
            f"  {digit}: "
            f"{counts.loc[digit]} "
            f"({percentages.loc[digit]:.2f}%)"
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

    print(
        "\n## V4 CONDITIONAL DIGIT ANALYSIS"
    )

    print("\n---")

    # --------------------------------------------------
    # OVERALL DISTRIBUTION
    # --------------------------------------------------

    print(
        "\n## OVERALL DISTRIBUTION"
    )

    print("\n---")

    overall = digit_distribution(df)

    print(
        overall.to_string(
            index=False,
            formatters={
                "Percentage":
                    lambda x:
                    f"{x:.2f}%"
            }
        )
    )

    # --------------------------------------------------
    # PRICE DELTA CONDITIONS
    # --------------------------------------------------

    print(
        "\n## PRICE DELTA CONDITIONS"
    )

    print("\n---")

    analyze_condition(
        "Large Positive Delta",
        df[
            df["Price Delta"] >
            df["Price Delta"].quantile(0.75)
        ]
    )

    analyze_condition(
        "Large Negative Delta",
        df[
            df["Price Delta"] <
            df["Price Delta"].quantile(0.25)
        ]
    )

    analyze_condition(
        "Small Delta",
        df[
            df["Price Delta"].abs() <
            df["Price Delta"].abs().quantile(0.25)
        ]
    )

    # --------------------------------------------------
    # ROUND DISTANCE CONDITIONS
    # --------------------------------------------------

    print(
        "\n## ROUND DISTANCE CONDITIONS"
    )

    print("\n---")

    analyze_condition(
        "Very Close To Whole Number",
        df[
            df["Round Distance"] <= 0.10
        ]
    )

    analyze_condition(
        "Far From Whole Number",
        df[
            df["Round Distance"] >= 0.40
        ]
    )

    # --------------------------------------------------
    # STEP VOLATILITY CONDITIONS
    # --------------------------------------------------

    print(
        "\n## STEP VOLATILITY CONDITIONS"
    )

    print("\n---")

    volatility_25 = (
        df["Step Volatility 10"]
        .quantile(0.25)
    )

    volatility_75 = (
        df["Step Volatility 10"]
        .quantile(0.75)
    )

    analyze_condition(
        "Low Step Volatility",
        df[
            df["Step Volatility 10"]
            <= volatility_25
        ]
    )

    analyze_condition(
        "High Step Volatility",
        df[
            df["Step Volatility 10"]
            >= volatility_75
        ]
    )

    # --------------------------------------------------
    # COMBINED CONDITIONS
    # --------------------------------------------------

    print(
        "\n## COMBINED CONDITIONS"
    )

    print("\n---")

    analyze_condition(
        "Large Positive Delta + Low Volatility",
        df[
            (df["Price Delta"] >
             df["Price Delta"].quantile(0.75))
            &
            (df["Step Volatility 10"] <=
             volatility_25)
        ]
    )

    analyze_condition(
        "Large Negative Delta + Low Volatility",
        df[
            (df["Price Delta"] <
             df["Price Delta"].quantile(0.25))
            &
            (df["Step Volatility 10"] <=
             volatility_25)
        ]
    )

    analyze_condition(
        "Near Whole Number + Low Volatility",
        df[
            (df["Round Distance"] <= 0.10)
            &
            (df["Step Volatility 10"] <=
             volatility_25)
        ]
    )

    analyze_condition(
        "Near Whole Number + High Volatility",
        df[
            (df["Round Distance"] <= 0.10)
            &
            (df["Step Volatility 10"] >=
             volatility_75)
        ]
    )

    print(
        "\n## ANALYSIS COMPLETE"
    )

    print("\n---")

    print(
        "Look for conditions where:"
    )

    print(
        "1. A digit is substantially above 10%"
    )

    print(
        "2. The sample size is reasonably large"
    )

    print(
        "3. The p-value is small"
    )

    print(
        "4. The result makes structural sense"
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()