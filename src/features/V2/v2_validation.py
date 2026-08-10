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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import FEATURE_V2_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    """Load V2 feature data."""
    df = pd.read_csv(FEATURE_V2_FILE)
    return df


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):
    """Prepare features and target."""

    # Target = next digit
    df["Next Digit"] = df["Last digit"].shift(-1)

    # Remove rows where target is unavailable
    df = df.dropna(subset=["Next Digit"])

    # New V2 features
    features = [
        "Digit Repeated",
        "Digit Repeated 2 Ago",
        "Digit Repeated 3 Ago",

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

    # Keep only features that actually exist
    features = [
        feature
        for feature in features
        if feature in df.columns
    ]

    X = df[features]
    y = df["Next Digit"].astype(int)

    # Remove any remaining missing values
    valid = X.notna().all(axis=1) & y.notna()

    X = X[valid]
    y = y[valid]

    return X, y, features


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

def split_data(X, y):

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

def train_model(X_train, y_train):

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate(model, X_test, y_test):

    predictions = model.predict(X_test)

    correct = (predictions == y_test).sum()

    total = len(y_test)

    accuracy = correct / total * 100

    baseline = 10.00

    difference = accuracy - baseline

    print("\n## V2 OUT-OF-SAMPLE VALIDATION")
    print("\n---")

    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Random baseline: {baseline:.2f}%")
    print(f"Difference: {difference:.2f}%")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    X, y, features = prepare_data(df)

    print(f"Usable rows: {len(X)}")

    print("\n## V2 FEATURES USED")
    print("\n---")

    for feature in features:
        print(feature)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n## DATA SPLIT")
    print("\n---")

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    model = train_model(
        X_train,
        y_train
    )

    validate(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()