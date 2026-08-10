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
from sklearn.metrics import accuracy_score

from config import FEATURE_FILE


def load_data():
    """Load feature data."""
    df = pd.read_csv(FEATURE_FILE)
    return df


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    # Features used by the model
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

    # Remove rows where required data is missing
    df = df.dropna(
        subset=features + [target]
    )

    print(f"Usable rows: {len(df)}")

    # Separate features and target
    X = df[features]
    y = df[target]

    # Chronological split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Train model
    print("\n## LOGISTIC REGRESSION")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    correct = (predictions == y_test).sum()
    total = len(y_test)

    baseline = 10.0
    difference = (accuracy * 100) - baseline

    print("\n---")
    print(
        f"Correct predictions: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Random baseline: "
        f"{baseline:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference:.2f}%"
    )


if __name__ == "__main__":
    main()