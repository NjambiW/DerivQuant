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


def evaluate_feature(train, test, feature):
    """
    Learn the most common Next Digit for each feature value
    using training data, then test it on unseen data.
    """

    predictions = (
        train.groupby(feature)["Next Digit"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    test = test.copy()

    test["Prediction"] = test[feature].map(predictions)

    test = test.dropna(subset=["Prediction", "Next Digit"])

    correct = (
        test["Prediction"] == test["Next Digit"]
    ).sum()

    total = len(test)

    accuracy = (correct / total) * 100

    return correct, total, accuracy


def main():

    df = load_data()

    # Remove rows where the target doesn't exist
    df = df.dropna(subset=["Next Digit"])

    print(f"Total rows: {len(df)}")

    train, test = split_data(df)

    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")

    features = [
        "Previous Digit",
        "Digit Change",
        "Even Percentage 10",
        "Even Percentage 25",
        "Even Percentage 50",
        "Even Percentage 100"
    ]

    print("\n## OUT-OF-SAMPLE FEATURE VALIDATION")
    print("-----------------------------------")

    for feature in features:

        correct, total, accuracy = evaluate_feature(
            train,
            test,
            feature
        )

        print(f"\n{feature}")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Baseline: 10.00%")
        print(f"Difference: {accuracy - 10:.2f}%")


if __name__ == "__main__":
    main()