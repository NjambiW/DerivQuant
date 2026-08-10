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

    return pd.read_csv(
        FEATURE_V4_FILE
    )


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    features = [
        "Price Delta",
        "Absolute Delta",
        "Step Volatility 5",
        "Step Volatility 10",
        "Round Distance",
        "Half Distance",
        "Previous Digit"
    ]

    df = df.dropna(
        subset=features + ["Next Digit"]
    )

    X = df[features]

    y = df["Next Digit"].astype(int)

    return X, y, features


# --------------------------------------------------
# VALIDATE FEATURE
# --------------------------------------------------

def validate_feature(
    X,
    y,
    feature
):

    X_feature = X[
        [feature]
    ]

    split_index = int(
        len(X_feature) * 0.80
    )

    X_train = X_feature.iloc[
        :split_index
    ]

    X_test = X_feature.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    # Standardize
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

    # Model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

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

    difference = (
        accuracy - 0.10
    )

    return (
        correct,
        total,
        accuracy,
        difference
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(
        f"Total rows: {len(df)}"
    )

    X, y, features = prepare_data(
        df
    )

    print(
        f"Usable rows: {len(X)}"
    )

    print(
        "\n## V4 FEATURE VALIDATION"
    )

    print("\n---")

    for feature in features:

        (
            correct,
            total,
            accuracy,
            difference
        ) = validate_feature(
            X,
            y,
            feature
        )

        print(
            f"\n{feature}"
        )

        print(
            f"Correct: "
            f"{correct}/{total}"
        )

        print(
            f"Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Baseline: 10.00%"
        )

        print(
            f"Difference: "
            f"{difference * 100:.2f}%"
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()