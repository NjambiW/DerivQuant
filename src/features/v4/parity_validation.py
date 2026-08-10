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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import FEATURE_V4_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    return pd.read_csv(FEATURE_V4_FILE)


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    features = [
        "Price Delta",
        "Round Distance",
        "Step Volatility 10"
    ]

    # Remove rows where Next Digit is missing
    # before calculating parity
    df = df.dropna(
        subset=features + ["Next Digit"]
    ).copy()

    # Create Even/Odd target from Next Digit
    # 0 = Even
    # 1 = Odd
    df["Next Parity"] = (
        df["Next Digit"] % 2
    ).astype(int)

    X = df[features]

    y = df["Next Parity"]

    return X, y


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(
        f"Total rows: {len(df)}"
    )

    X, y = prepare_data(df)

    print(
        f"Usable rows: {len(X)}"
    )

    # --------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    # --------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    correct = (
        predictions ==
        y_test.to_numpy()
    ).sum()

    total = len(y_test)

    accuracy = (
        correct / total
    )

    baseline = 0.50

    difference = (
        accuracy - baseline
    )

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    print(
        "\n## V4 PARITY VALIDATION"
    )

    print("\n---")

    print(
        "\nFeatures used:"
    )

    print(
        "Price Delta"
    )

    print(
        "Round Distance"
    )

    print(
        "Step Volatility 10"
    )

    print(
        "\nTarget:"
    )

    print(
        "Next Digit → Even / Odd"
    )

    print(
        f"\nCorrect predictions: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Random baseline: "
        f"{baseline * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{difference * 100:.2f}%"
    )

    # --------------------------------------------------
    # PREDICTION BREAKDOWN
    # --------------------------------------------------

    even_predictions = (
        predictions == 0
    ).sum()

    odd_predictions = (
        predictions == 1
    ).sum()

    print(
        "\n## PREDICTION DISTRIBUTION"
    )

    print("\n---")

    print(
        f"Even predictions: "
        f"{even_predictions}"
    )

    print(
        f"Odd predictions: "
        f"{odd_predictions}"
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()