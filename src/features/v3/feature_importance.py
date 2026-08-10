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

    features = [
        "Digit Change",
        "Digit Repeated",
        "Repeat Streak",
        "Most Common Digit 10",
        "Least Common Digit 10",
        "Digit Concentration 10",
        "Digit Entropy 10",
        "Digit Entropy 25",
        "Distribution Shift 10",
        "Even Percentage 10",
        "Even Percentage 25"
    ]

    features = [
        feature
        for feature in features
        if feature in df.columns
    ]

    df = df.dropna(
        subset=features + ["Next Digit"]
    )

    X = df[features]

    y = df["Next Digit"].astype(int)

    return X, y, features


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    X, y, features = prepare_data(df)

    print(f"Usable rows: {len(X)}")

    # --------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

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

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # TRAIN LOGISTIC REGRESSION
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
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    coefficients = model.coef_

    importance = (
        abs(coefficients)
        .mean(axis=0)
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n## V3 FEATURE IMPORTANCE")
    print("\n---")

    for _, row in importance_df.iterrows():

        print(
            f"{row['Feature']}: "
            f"{row['Importance']:.4f}"
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()