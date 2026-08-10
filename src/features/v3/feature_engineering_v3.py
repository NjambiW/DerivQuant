import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

V3_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(V3_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pandas as pd
import numpy as np

from config import TICKS_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(TICKS_FILE)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def create_features(df):

    df = df.copy()

    digits = df["Last digit"]

    # --------------------------------------------------
    # BASIC DIGIT CHANGE
    # --------------------------------------------------

    df["Digit Change"] = (
        digits.diff().abs()
    )

    # --------------------------------------------------
    # REPEATED DIGIT
    # --------------------------------------------------

    df["Digit Repeated"] = (
        digits == digits.shift(1)
    ).astype(int)

    # --------------------------------------------------
    # REPEAT STREAK
    # --------------------------------------------------

    df["Repeat Streak"] = (
        digits.groupby(
            (digits != digits.shift()).cumsum()
        ).cumcount() + 1
    )

    # --------------------------------------------------
    # ROLLING MOST COMMON DIGIT
    # --------------------------------------------------

    def most_common(window):

        counts = window.value_counts()

        return counts.index[0]

    df["Most Common Digit 10"] = (
        digits
        .rolling(10)
        .apply(
            lambda x: most_common(
                pd.Series(x)
            ),
            raw=False
        )
    )

    # --------------------------------------------------
    # ROLLING LEAST COMMON DIGIT
    # --------------------------------------------------

    def least_common(window):

        counts = window.value_counts()

        return counts.index[-1]

    df["Least Common Digit 10"] = (
        digits
        .rolling(10)
        .apply(
            lambda x: least_common(
                pd.Series(x)
            ),
            raw=False
        )
    )

    # --------------------------------------------------
    # DIGIT CONCENTRATION
    # --------------------------------------------------

    def concentration(window):

        counts = pd.Series(window).value_counts(
            normalize=True
        )

        return counts.max()

    df["Digit Concentration 10"] = (
        digits
        .rolling(10)
        .apply(
            concentration,
            raw=False
        )
    )

    # --------------------------------------------------
    # DIGIT ENTROPY
    # --------------------------------------------------

    def entropy(window):

        counts = pd.Series(window).value_counts(
            normalize=True
        )

        return -np.sum(
            counts * np.log2(counts)
        )

    df["Digit Entropy 10"] = (
        digits
        .rolling(10)
        .apply(
            entropy,
            raw=False
        )
    )

    # --------------------------------------------------
    # LONGER-TERM ENTROPY
    # --------------------------------------------------

    df["Digit Entropy 25"] = (
        digits
        .rolling(25)
        .apply(
            entropy,
            raw=False
        )
    )

    # --------------------------------------------------
    # DIGIT DISTRIBUTION SHIFT
    # --------------------------------------------------

    def distribution_shift(window):

        series = pd.Series(window)

        recent = series.iloc[-5:]
        previous = series.iloc[:5]

        recent_dist = (
            recent.value_counts(
                normalize=True
            )
            .reindex(range(10), fill_value=0)
        )

        previous_dist = (
            previous.value_counts(
                normalize=True
            )
            .reindex(range(10), fill_value=0)
        )

        return np.abs(
            recent_dist - previous_dist
        ).sum()

    df["Distribution Shift 10"] = (
        digits
        .rolling(10)
        .apply(
            distribution_shift,
            raw=False
        )
    )

    # --------------------------------------------------
    # EVEN / ODD CONCENTRATION
    # --------------------------------------------------

    df["Even"] = (
        digits % 2 == 0
    ).astype(int)

    df["Even Percentage 10"] = (
        df["Even"]
        .rolling(10)
        .mean()
    )

    df["Even Percentage 25"] = (
        df["Even"]
        .rolling(25)
        .mean()
    )

    # --------------------------------------------------
    # CLEAN UP
    # --------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(f"Raw rows: {len(df)}")

    df = create_features(df)

    print(
        f"Feature rows: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print("\n## V3 NEW FEATURES")
    print("\n---")

    new_features = [
        "Digit Change",
        "Digit Repeated",
        "Repeat Streak",
        "Most Common Digit 10",
        "Least Common Digit 10",
        "Digit Concentration 10",
        "Digit Entropy 10",
        "Digit Entropy 25",
        "Distribution Shift 10",
        "Even Percentage 10",
        "Even Percentage 25"
    ]

    print(
        df[new_features].tail(20)
    )

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "feature_data_v3.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved V3 features to:"
    )

    print(output_file)


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()