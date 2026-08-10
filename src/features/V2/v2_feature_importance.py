import sys
import os

CURRENT_FILE = os.path.abspath(__file__)

V2_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(V2_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import FEATURE_V2_FILE


def load_data():
    """Load V2 feature data."""
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

    # Keep only columns that exist
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


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    X, y, features = prepare_data(df)

    print(f"Usable rows: {len(X)}")

    # Chronological 80/20 split
    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Standardize features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Train model
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

    # Average absolute coefficient across
    # all ten digit classes
    importance = abs(coefficients).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n## V2 FEATURE IMPORTANCE")
    print("\n---")

    for _, row in importance_df.iterrows():

        print(
            f"{row['Feature']}: "
            f"{row['Importance']:.4f}"
        )


if __name__ == "__main__":
    main()