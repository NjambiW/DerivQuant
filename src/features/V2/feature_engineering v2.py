import sys
import os

# Allow Python to find config.py
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import pandas as pd

from config import TICKS_FILE


def load_data():
    """Load raw tick data."""
    return pd.read_csv(TICKS_FILE)


def create_features(df):
    """Create second-generation features."""

    # --------------------------------------------------
    # BASIC DIGIT HISTORY
    # --------------------------------------------------

    df["Previous Digit"] = df["Last digit"].shift(1)
    df["Digit 2 Ago"] = df["Last digit"].shift(2)
    df["Digit 3 Ago"] = df["Last digit"].shift(3)
    df["Digit 5 Ago"] = df["Last digit"].shift(5)
    df["Digit 10 Ago"] = df["Last digit"].shift(10)

    # --------------------------------------------------
    # TARGET
    # --------------------------------------------------

    df["Next Digit"] = df["Last digit"].shift(-1)

    # --------------------------------------------------
    # DIGIT CHANGE
    # --------------------------------------------------

    df["Digit Change"] = (
        df["Last digit"] -
        df["Previous Digit"]
    )

    # --------------------------------------------------
    # PARITY
    # --------------------------------------------------

    df["Parity"] = df["Last digit"] % 2

    df["Previous Parity"] = (
        df["Previous Digit"] % 2
    )

    df["Next Parity"] = (
        df["Next Digit"] % 2
    )

    # --------------------------------------------------
    # REPETITION
    # --------------------------------------------------

    df["Digit Repeated"] = (
        df["Last digit"] ==
        df["Previous Digit"]
    ).astype(int)

    df["Digit Repeated 2 Ago"] = (
        df["Last digit"] ==
        df["Digit 2 Ago"]
    ).astype(int)

    # --------------------------------------------------
    # RECENT DIGIT FREQUENCIES
    # --------------------------------------------------

    for digit in range(10):

        df[f"Digit {digit} Count 10"] = (
            df["Last digit"]
            .rolling(10)
            .apply(
                lambda x, d=digit:
                (x == d).sum(),
                raw=True
            )
        )

        df[f"Digit {digit} Count 25"] = (
            df["Last digit"]
            .rolling(25)
            .apply(
                lambda x, d=digit:
                (x == d).sum(),
                raw=True
            )
        )

    # --------------------------------------------------
    # PARITY COUNTS
    # --------------------------------------------------

    df["Even Count 10"] = (
        (df["Last digit"] % 2 == 0)
        .rolling(10)
        .sum()
    )

    df["Odd Count 10"] = (
        (df["Last digit"] % 2 == 1)
        .rolling(10)
        .sum()
    )

    df["Even Percentage 10"] = (
        df["Even Count 10"] / 10 * 100
    )

    df["Even Count 25"] = (
        (df["Last digit"] % 2 == 0)
        .rolling(25)
        .sum()
    )

    df["Odd Count 25"] = (
        (df["Last digit"] % 2 == 1)
        .rolling(25)
        .sum()
    )

    df["Even Percentage 25"] = (
        df["Even Count 25"] / 25 * 100
    )

    # --------------------------------------------------
    # DIGIT STREAK
    # --------------------------------------------------

    streak = []

    current_digit = None
    current_streak = 0

    for digit in df["Last digit"]:

        if digit == current_digit:
            current_streak += 1
        else:
            current_digit = digit
            current_streak = 1

        streak.append(current_streak)

    df["Digit Streak"] = streak

    # --------------------------------------------------
    # PARITY STREAK
    # --------------------------------------------------

    parity_streak = []

    current_parity = None
    current_parity_streak = 0

    for parity in df["Parity"]:

        if parity == current_parity:
            current_parity_streak += 1
        else:
            current_parity = parity
            current_parity_streak = 1

        parity_streak.append(current_parity_streak)

    df["Parity Streak"] = parity_streak

    return df


def main():

    df = load_data()

    print(f"Raw rows: {len(df)}")

    df = create_features(df)

    print(f"Feature rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\n## NEW FEATURES")
    print("----------------")

    new_features = [
        "Digit Repeated",
        "Digit Repeated 2 Ago",
        "Digit Streak",
        "Parity Streak",
        "Digit 0 Count 10",
        "Digit 1 Count 10",
        "Digit 2 Count 10",
        "Digit 3 Count 10",
        "Digit 4 Count 10",
        "Digit 5 Count 10",
        "Digit 6 Count 10",
        "Digit 7 Count 10",
        "Digit 8 Count 10",
        "Digit 9 Count 10",
    ]

    print(df[new_features].tail(20))

    # Save v2 dataset
    output_file = os.path.join(
        os.path.dirname(TICKS_FILE),
        "feature_data_v2.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved v2 features to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()