import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

V3_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(V3_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from config import FEATURE_V3_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(FEATURE_V3_FILE)


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    df = df.copy()

    df["Next Digit"] = df["Last digit"].shift(-1)

    df = df.dropna(
        subset=["Next Digit"]
    )

    return df


# --------------------------------------------------
# TEST FEATURE SET
# --------------------------------------------------

def test_feature_set(
    df,
    feature_set
):

    available_features = [
        feature
        for feature in feature_set
        if feature in df.columns
    ]

    data = df.dropna(
        subset=available_features
    )

    X = data[available_features]

    y = data["Next Digit"].astype(int)

    # Chronological 80/20 split

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Standardize

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
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
    ) * 100

    difference = accuracy - 10.0

    return correct, total, accuracy, difference


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(
        f"Total rows: {len(df)}"
    )

    df = prepare_data(df)

    print(
        f"Usable rows: {len(df)}"
    )

    # --------------------------------------------------
    # FEATURE SETS
    # --------------------------------------------------

    feature_sets = {

        "Repeat Streak": [
            "Repeat Streak"
        ],

        "Digit Repeated": [
            "Digit Repeated"
        ],

        "Repeat Streak + Digit Repeated": [
            "Repeat Streak",
            "Digit Repeated"
        ],

        "Repeat Streak + Digit Repeated + Entropy": [
            "Repeat Streak",
            "Digit Repeated",
            "Digit Entropy 10"
        ],

        "Repeat Streak + Digit Repeated + Even %": [
            "Repeat Streak",
            "Digit Repeated",
            "Even Percentage 10"
        ]
    }

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print(
        "\n## V3 FEATURE SUBSET VALIDATION"
    )

    print("\n---")

    for name, features in feature_sets.items():

        correct, total, accuracy, difference = (
            test_feature_set(
                df,
                features
            )
        )

        print(
            f"\n{name}"
        )

        print(
            f"Features: {features}"
        )

        print(
            f"Correct: "
            f"{correct}/{total}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Baseline: 10.00%"
        )

        print(
            f"Difference: "
            f"{difference:.2f}%"
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()