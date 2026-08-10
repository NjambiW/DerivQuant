import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import pandas as pd
from config import FEATURE_FILE


def load_data():
    """Load feature data."""
    return pd.read_csv(FEATURE_FILE)


def split_data(df):
    """Split data chronologically into training and testing sets."""

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test


def evaluate_combination(train, test, features):
    """
    Learn the most common Next Digit for each combination
    of feature values using training data.
    """

    # Find the most common next digit for each combination
    predictions = (
        train.groupby(features)["Next Digit"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index(name="Prediction")
    )

    # Match those learned combinations against test data
    test = test.merge(
        predictions,
        on=features,
        how="left"
    )

    test = test.dropna(
        subset=["Prediction", "Next Digit"]
    )

    correct = (
        test["Prediction"] == test["Next Digit"]
    ).sum()

    total = len(test)

    if total == 0:
        return 0, 0, 0

    accuracy = (correct / total) * 100

    return correct, total, accuracy


def main():

    df = load_data()

    # Remove rows where target doesn't exist
    df = df.dropna(subset=["Next Digit"])

    print(f"Total rows: {len(df)}")

    train, test = split_data(df)

    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")

    combinations = [

        ["Previous Digit", "Digit Change"],

        ["Previous Digit", "Even Percentage 25"],

        ["Digit Change", "Even Percentage 25"],

        ["Digit Change", "Even Percentage 50"],

        ["Digit Change", "Even Percentage 100"],

        [
            "Previous Digit",
            "Digit Change",
            "Even Percentage 25"
        ],

        [
            "Previous Digit",
            "Digit Change",
            "Even Percentage 50"
        ],

        [
            "Digit Change",
            "Even Percentage 25",
            "Even Percentage 50"
        ]
    ]

    print("\n## FEATURE COMBINATION VALIDATION")
    print("--------------------------------")

    for features in combinations:

        correct, total, accuracy = evaluate_combination(
            train,
            test,
            features
        )

        name = " + ".join(features)

        print(f"\n{name}")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Baseline: 10.00%")
        print(f"Difference: {accuracy - 10:.2f}%")


if __name__ == "__main__":
    main()