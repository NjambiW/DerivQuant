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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import FEATURE_FILE


def load_data():
    """Load feature data."""
    return pd.read_csv(FEATURE_FILE)


def prepare_data(df):
    """Prepare features and target."""

    features = [
        "Previous Digit",
        "Digit 2 Ago",
        "Digit 3 Ago",
        "Digit 5 Ago",
        "Digit 10 Ago",
        "Parity",
        "Previous Parity",
        "Digit Change",
        "Even Percentage 10",
        "Even Percentage 25",
        "Even Percentage 50",
        "Even Percentage 100"
    ]

    target = "Next Digit"

    # Keep only rows where all required values exist
    df = df.dropna(
        subset=features + [target]
    )

    X = df[features]
    y = df[target].astype(int)

    return X, y


def split_data(X, y):
    """Split data chronologically."""

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the baseline Random Forest model."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    X, y = prepare_data(df)

    print(f"Usable rows: {len(X)}")

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    model = train_model(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    print("\n## RANDOM FOREST BASELINE")
    print("-------------------------")

    print(f"Correct predictions: {(predictions == y_test).sum()}/{len(y_test)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Random baseline: 10.00%")
    print(f"Difference: {accuracy - 10:.2f}%")


if __name__ == "__main__":
    main()