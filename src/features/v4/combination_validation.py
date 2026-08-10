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

    return X, y


# --------------------------------------------------
# TEST COMBINATION
# --------------------------------------------------

def test_combination(
    X,
    y,
    features
):

    X_subset = X[features]

    split_index = int(
        len(X_subset) * 0.80
    )

    X_train = X_subset.iloc[
        :split_index
    ]

    X_test = X_subset.iloc[
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

    # Logistic Regression
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

    accuracy = correct / total

    difference = accuracy - 0.10

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

    X, y = prepare_data(df)

    print(
        f"Usable rows: {len(X)}"
    )

    print(
        "\n## V4 FEATURE COMBINATION VALIDATION"
    )

    print("\n---")

    combinations = [

        (
            "Price Delta + Round Distance",
            [
                "Price Delta",
                "Round Distance"
            ]
        ),

        (
            "Price Delta + Half Distance",
            [
                "Price Delta",
                "Half Distance"
            ]
        ),

        (
            "Price Delta + Step Volatility 5",
            [
                "Price Delta",
                "Step Volatility 5"
            ]
        ),

        (
            "Price Delta + Step Volatility 10",
            [
                "Price Delta",
                "Step Volatility 10"
            ]
        ),

        (
            "Round Distance + Price Delta + Step Volatility 5",
            [
                "Round Distance",
                "Price Delta",
                "Step Volatility 5"
            ]
        ),

        (
            "Round Distance + Price Delta + Step Volatility 10",
            [
                "Round Distance",
                "Price Delta",
                "Step Volatility 10"
            ]
        ),

        (
            "All V4 Features",
            [
                "Price Delta",
                "Absolute Delta",
                "Step Volatility 5",
                "Step Volatility 10",
                "Round Distance",
                "Half Distance",
                "Previous Digit"
            ]
        )
    ]

    results = []

    # --------------------------------------------------
    # RUN TESTS
    # --------------------------------------------------

    for name, features in combinations:

        (
            correct,
            total,
            accuracy,
            difference
        ) = test_combination(
            X,
            y,
            features
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
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Baseline: 10.00%"
        )

        print(
            f"Difference: "
            f"{difference * 100:.2f}%"
        )

        results.append({
            "Combination": name,
            "Correct": correct,
            "Total": total,
            "Accuracy": accuracy,
            "Difference": difference
        })

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print(
        "\n## V4 COMBINATION SUMMARY"
    )

    print("\n---")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy":
                    lambda x:
                    f"{x * 100:.2f}%",
                "Difference":
                    lambda x:
                    f"{x * 100:.2f}%"
            }
        )
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()