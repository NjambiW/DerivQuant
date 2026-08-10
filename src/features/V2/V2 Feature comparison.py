import sys
import os

CURRENT_FILE = os.path.abspath(__file__)

V2_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(V2_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)
# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler


from config import FEATURE_V2_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(FEATURE_V2_FILE)


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_data(df):

    # Create target
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

    # Only use features that exist
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


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    X, y = prepare_data(df)

    print(f"Usable rows: {len(X)}")

    # --------------------------------------------------
    # CHRONOLOGICAL 80/20 SPLIT
    # --------------------------------------------------

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # --------------------------------------------------
    # STANDARDIZE FOR LOGISTIC REGRESSION + KNN
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

        "K Nearest Neighbors":
            KNeighborsClassifier(
                n_neighbors=10
            )
    }

    results = []

    print("\n## V2 MODEL COMPARISON")
    print("\n---")

    # --------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Logistic Regression and KNN
        # use standardized data

        if name in [
            "Logistic Regression",
            "K Nearest Neighbors"
        ]:

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

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
    # RESULTS TABLE
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\n## V2 MODEL RESULTS")
    print("\n---")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2f}%".format,
                "Difference": "{:.2f}%".format
            }
        )
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    main()