import sys
import os

# Allow Python to find config.py
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import FEATURE_FILE


def load_data():
    """Load feature data."""
    return pd.read_csv(FEATURE_FILE)


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    features = [
        "Previous Digit",
        "Digit 2 Ago",
        "Digit 3 Ago",
        "Digit 5 Ago",
        "Digit 10 Ago",
        "Digit Change",
        "Even Percentage 10",
        "Even Percentage 25",
        "Even Percentage 50",
        "Even Percentage 100"
    ]

    target = "Next Digit"

    # Remove incomplete rows
    df = df.dropna(
        subset=features + [target]
    )

    print(f"Usable rows: {len(df)}")

    X = df[features]
    y = df[target]

    # Chronological 80/20 split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get probabilities for every digit
    probabilities = model.predict_proba(X_test)

    classes = model.classes_

    # Highest probability digit
    predictions = model.predict(X_test)

    # Probability of the model's chosen digit
    max_probabilities = probabilities.max(axis=1)

    # Check whether prediction was correct
    correct = predictions == y_test.values

    results = pd.DataFrame({
        "Actual": y_test.values,
        "Prediction": predictions,
        "Probability": max_probabilities,
        "Correct": correct
    })

    print("\n## PROBABILITY ANALYSIS")
    print("----------------------")

    print("\nAverage predicted probability:")
    print(
        f"{results['Probability'].mean() * 100:.2f}%"
    )

    print(
        f"Highest predicted probability: "
        f"{results['Probability'].max() * 100:.2f}%"
    )

    print(
        f"Lowest predicted probability: "
        f"{results['Probability'].min() * 100:.2f}%"
    )

    # Confidence groups
    print("\n## CONFIDENCE LEVELS")
    print("--------------------")

    confidence_levels = [
        ("10%-11%", 0.10, 0.11),
        ("11%-12%", 0.11, 0.12),
        ("12%-13%", 0.12, 0.13),
        ("13%-15%", 0.13, 0.15),
        ("15%+", 0.15, 1.00)
    ]

    for name, lower, upper in confidence_levels:

        group = results[
            (results["Probability"] >= lower) &
            (results["Probability"] < upper)
        ]

        if len(group) == 0:
            continue

        accuracy = group["Correct"].mean() * 100

        print(
            f"{name}: "
            f"{len(group)} predictions | "
            f"Accuracy: {accuracy:.2f}%"
        )

    # Top individual predictions
    print("\n## HIGHEST CONFIDENCE PREDICTIONS")
    print("----------------------------------")

    top_predictions = results.sort_values(
        "Probability",
        ascending=False
    ).head(20)

    print(
        top_predictions[
            [
                "Actual",
                "Prediction",
                "Probability",
                "Correct"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
