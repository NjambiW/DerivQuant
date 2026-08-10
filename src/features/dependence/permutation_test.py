import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

DEPENDENCE_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(DEPENDENCE_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import FEATURE_V3_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(
        FEATURE_V3_FILE
    )


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    df = df.copy()

    df["Next Digit"] = (
        df["Last digit"].shift(-1)
    )

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
# TRAIN / TEST
# --------------------------------------------------

def evaluate_model(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

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

    accuracy = (
        correct / len(y_test)
    )

    return accuracy


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
    # REAL MODEL
    # --------------------------------------------------

    real_accuracy = evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        "\n## REAL MODEL"
    )

    print("\n---")

    print(
        f"Accuracy: "
        f"{real_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------
    # PERMUTATION TEST
    # --------------------------------------------------

    permutations = 500

    permutation_accuracies = []

    print(
        "\n## PERMUTATION TEST"
    )

    print("\n---")

    print(
        f"Running {permutations} "
        f"random permutations..."
    )

    rng = np.random.default_rng(
        42
    )

    for i in range(permutations):

        # Shuffle training targets
        shuffled_y_train = (
            rng.permutation(
                y_train.to_numpy()
            )
        )

        accuracy = evaluate_model(
            X_train,
            X_test,
            shuffled_y_train,
            y_test
        )

        permutation_accuracies.append(
            accuracy
        )

        if (i + 1) % 100 == 0:

            print(
                f"Completed "
                f"{i + 1}/{permutations}"
            )

    permutation_accuracies = np.array(
        permutation_accuracies
    )

    # --------------------------------------------------
    # PERMUTATION STATISTICS
    # --------------------------------------------------

    mean_accuracy = (
        permutation_accuracies.mean()
    )

    std_accuracy = (
        permutation_accuracies.std()
    )

    percentile_95 = np.percentile(
        permutation_accuracies,
        95
    )

    percentile_99 = np.percentile(
        permutation_accuracies,
        99
    )

    # --------------------------------------------------
    # EMPIRICAL P-VALUE
    # --------------------------------------------------

    better_or_equal = (
        permutation_accuracies
        >= real_accuracy
    ).sum()

    empirical_p_value = (
        better_or_equal + 1
    ) / (
        permutations + 1
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print(
        "\n## PERMUTATION RESULTS"
    )

    print("\n---")

    print(
        f"Real accuracy: "
        f"{real_accuracy * 100:.2f}%"
    )

    print(
        f"Mean shuffled accuracy: "
        f"{mean_accuracy * 100:.2f}%"
    )

    print(
        f"Standard deviation: "
        f"{std_accuracy * 100:.2f}%"
    )

    print(
        f"95th percentile: "
        f"{percentile_95 * 100:.2f}%"
    )

    print(
        f"99th percentile: "
        f"{percentile_99 * 100:.2f}%"
    )

    print(
        f"Shuffled runs >= real model: "
        f"{better_or_equal}/{permutations}"
    )

    print(
        f"Empirical p-value: "
        f"{empirical_p_value:.4f}"
    )

    # --------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------

    print(
        "\n## CONCLUSION"
    )

    print("\n---")

    if empirical_p_value < 0.05:

        print(
            "The real model performs "
            "significantly better than "
            "the shuffled-target models."
        )

        print(
            "This provides evidence that "
            "the features contain predictive "
            "information."
        )

    else:

        print(
            "The real model does NOT "
            "perform significantly better "
            "than shuffled-target models."
        )

        print(
            "The observed accuracy is "
            "consistent with random variation."
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()
