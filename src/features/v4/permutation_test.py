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

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import FEATURE_V4_FILE

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

N_PERMUTATIONS = 500

RANDOM_SEED = 42


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
        "Round Distance",
        "Step Volatility 10"
    ]

    df = df.dropna(
        subset=features + ["Next Digit"]
    )

    X = df[features]

    y = df["Next Digit"].astype(int)

    return X, y


# --------------------------------------------------
# TRAIN AND TEST MODEL
# --------------------------------------------------

def model_accuracy(
        X_train,
        X_test,
        y_train,
        y_test
):
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

    accuracy = (
    predictions ==
    np.asarray(y_test)
    ).mean()

    return accuracy


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    np.random.seed(
        RANDOM_SEED
    )

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
    # REAL MODEL
    # --------------------------------------------------

    real_accuracy = model_accuracy(
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

    print(
        "\n## PERMUTATION TEST"
    )

    print("\n---")

    print(
        f"Running "
        f"{N_PERMUTATIONS} "
        f"random permutations..."
    )

    shuffled_accuracies = []

    y_train_array = (
        y_train.to_numpy()
    )

    y_test_array = (
        y_test.to_numpy()
    )

    # --------------------------------------------------
    # SHUFFLE TARGET
    # --------------------------------------------------

    for i in range(
            N_PERMUTATIONS
    ):

        # Shuffle training target
        shuffled_y_train = (
            np.random.permutation(
                y_train_array
            )
        )

        accuracy = model_accuracy(
            X_train,
            X_test,
            shuffled_y_train,
            y_test_array
        )

        shuffled_accuracies.append(
            accuracy
        )

        if (
                (i + 1) % 100 == 0
        ):
            print(
                f"Completed "
                f"{i + 1}/"
                f"{N_PERMUTATIONS}"
            )

    shuffled_accuracies = np.array(
        shuffled_accuracies
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    mean_accuracy = (
        shuffled_accuracies.mean()
    )

    std_accuracy = (
        shuffled_accuracies.std(
            ddof=1
        )
    )

    percentile_95 = (
        np.percentile(
            shuffled_accuracies,
            95
        )
    )

    percentile_99 = (
        np.percentile(
            shuffled_accuracies,
            99
        )
    )

    # Number of shuffled models
    # at least as good as real model
    shuffled_equal_or_better = (
            shuffled_accuracies >=
            real_accuracy
    ).sum()

    # Add one to numerator and denominator
    # to avoid a zero empirical p-value.
    empirical_p_value = (
                                shuffled_equal_or_better + 1
                        ) / (
                                N_PERMUTATIONS + 1
                        )

    # --------------------------------------------------
    # OUTPUT
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
        f"Shuffled runs >= "
        f"real model: "
        f"{shuffled_equal_or_better}/"
        f"{N_PERMUTATIONS}"
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
            "shuffled-target models."
        )

    else:

        print(
            "The real model does NOT perform "
            "significantly better than "
            "shuffled-target models."
        )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()