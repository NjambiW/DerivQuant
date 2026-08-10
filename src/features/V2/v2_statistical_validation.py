import sys
import os

CURRENT_FILE = os.path.abspath(__file__)

V2_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(V2_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import binomtest

from config import FEATURE_V2_FILE


def load_data():
    return pd.read_csv(FEATURE_V2_FILE)


def prepare_data(df):

    df["Next Digit"] = df["Last digit"].shift(-1)

    features = [
        "Digit Repeated",
        "Digit Repeated 2 Ago",
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

    return X, y


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    X, y = prepare_data(df)

    print(f"Usable rows: {len(X)}")

    # Chronological 80/20 split
    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # --------------------------------------------------
    # TRAIN DECISION TREE
    # --------------------------------------------------

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    correct = (
        predictions == y_test.to_numpy()
    ).sum()

    total = len(y_test)

    accuracy = correct / total

    # --------------------------------------------------
    # STATISTICAL TEST
    # --------------------------------------------------

    baseline = 0.10

    expected_correct = total * baseline

    advantage = (
        accuracy - baseline
    ) * 100

    result = binomtest(
        correct,
        total,
        baseline,
        alternative="greater"
    )

    p_value = result.pvalue

    confidence_interval = result.proportion_ci(
        confidence_level=0.95
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n## V2 STATISTICAL VALIDATION")
    print("\n---")

    print(
        f"Correct predictions: "
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
        f"{expected_correct:.0f}"
    )

    print(
        f"Observed advantage: "
        f"{advantage:.2f} percentage points"
    )

    print(
        f"95% confidence interval: "
        f"{confidence_interval.low * 100:.2f}% - "
        f"{confidence_interval.high * 100:.2f}%"
    )

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    # --------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------

    print("\n## CONCLUSION")
    print("\n---")

    if p_value < 0.05:

        print(
            "The V2 model shows statistically "
            "significant evidence of accuracy "
            "above the 10% baseline."
        )

    else:

        print(
            "The V2 model does NOT show statistically "
            "significant evidence of accuracy above "
            "the 10% baseline."
        )


if __name__ == "__main__":
    main()