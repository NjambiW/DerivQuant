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

from scipy.stats import binomtest

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

    # Next digit is the target
    df["Next Digit"] = df["Last digit"].shift(-1)

    features = [
        "Repeat Streak",
        "Digit Repeated",
        "Digit Entropy 10"
    ]

    df = df.dropna(
        subset=features + ["Next Digit"]
    )

    X = df[features]

    y = df["Next Digit"].astype(int)

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
    # CHRONOLOGICAL 80/20 SPLIT
    # --------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

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

    predictions = model.predict(
        X_test_scaled
    )

    # --------------------------------------------------
    # ACCURACY
    # --------------------------------------------------

    correct = (
        predictions ==
        y_test.to_numpy()
    ).sum()

    total = len(y_test)

    accuracy = (
        correct / total
    )

    baseline = 0.10

    expected_correct = (
        total * baseline
    )

    advantage = (
        accuracy - baseline
    )

    # --------------------------------------------------
    # BINOMIAL TEST
    # --------------------------------------------------

    test_result = binomtest(
        correct,
        total,
        baseline,
        alternative="greater"
    )

    p_value = test_result.pvalue

    # --------------------------------------------------
    # 95% CONFIDENCE INTERVAL
    # --------------------------------------------------

    confidence_interval = test_result.proportion_ci(
        confidence_level=0.95
    )

    lower = confidence_interval.low
    upper = confidence_interval.high

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print(
        "\n## V3 SUBSET STATISTICAL VALIDATION"
    )

    print("\n---")

    print(
        "Features tested:"
    )

    print(
        "Repeat Streak"
    )

    print(
        "Digit Repeated"
    )

    print(
        "Digit Entropy 10"
    )

    print(
        f"\nCorrect predictions: "
        f"{correct}/{total}"
    )

    print(
        f"Observed accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Random baseline: "
        f"{baseline * 100:.2f}%"
    )

    print(
        f"Expected correct at baseline: "
        f"{expected_correct:.1f}"
    )

    print(
        f"Observed advantage: "
        f"{advantage * 100:.2f} percentage points"
    )

    print(
        f"95% confidence interval: "
        f"{lower * 100:.2f}% - "
        f"{upper * 100:.2f}%"
    )

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    # --------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------

    print(
        "\n## CONCLUSION"
    )

    print("\n---")

    if p_value < 0.05:

        print(
            "The V3 subset shows statistically "
            "significant evidence of accuracy "
            "above the 10% baseline."
        )

    else:

        print(
            "The V3 subset does NOT show "
            "statistically significant evidence "
            "of accuracy above the 10% baseline."
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()