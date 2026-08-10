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
from config import FEATURE_FILE


def load_data():
    """Load feature data."""
    df = pd.read_csv(FEATURE_FILE)
    return df


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    # Features we want the model to use
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
    df = df.dropna(subset=features + [target])

    print(f"Usable rows: {len(df)}")

    X = df[features]
    y = df[target]

    # Chronological split
    split = int(len(df) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Create Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    correct = (predictions == y_test).sum()
    total = len(y_test)

    accuracy = correct / total * 100

    print("\n## RANDOM FOREST BASELINE")
    print("-------------------------")

    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Random baseline: 10.00%")
    print(f"Difference: {accuracy - 10:.2f}%")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\n## FEATURE IMPORTANCE")
    print("----------------------")

    for feature, value in importance.items():
        print(f"{feature}: {value:.4f}")


if __name__ == "__main__":
    main()