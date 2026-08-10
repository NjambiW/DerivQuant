import sys
import os

# Allow Python to find config.py inside src
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import pandas as pd

from config import FEATURE_FILE

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier


def load_data():
    """Load feature data."""
    return pd.read_csv(FEATURE_FILE)


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    # Features used by every model
    features = [
        "Previous Digit",
        "Digit 2 Ago",
        "Digit 3 Ago",
        "Digit 5 Ago",
        "Digit 10 Ago",
        "Digit Change",
        "Even Percentage 10",
        "Even Percentage 25",
        "Even Percentage 50",
        "Even Percentage 100"
    ]

    target = "Next Digit"

    # Remove incomplete rows
    df = df.dropna(
        subset=features + [target]
    )

    print(f"Usable rows: {len(df)}")

    X = df[features]
    y = df[target]

    # --------------------------------------------------
    # CHRONOLOGICAL TRAIN / TEST SPLIT
    # --------------------------------------------------

    split = int(len(df) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "K Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=50,
            n_jobs=-1
        )
    }

    # --------------------------------------------------
    # TRAIN AND TEST
    # --------------------------------------------------

    print("\n## MODEL COMPARISON")
    print("-------------------")

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        correct = (
            predictions == y_test
        ).sum()

        total = len(y_test)

        accuracy = (
            correct / total
        ) * 100

        difference = accuracy - 10

        print(
            f"{name}: "
            f"{correct}/{total} "
            f"({accuracy:.2f}%)"
        )

        results.append({
            "Model": name,
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
        "Accuracy",
        ascending=False
    )

    print("\n## MODEL RESULTS")
    print("----------------")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2f}%".format,
                "Difference": "{:.2f}%".format
            }
        )
    )


if __name__ == "__main__":
    main()