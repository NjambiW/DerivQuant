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

from config import TICKS_FILE, FEATURE_V4_FILE


# --------------------------------------------------
# LOAD RAW TICK DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(TICKS_FILE)


# --------------------------------------------------
# CREATE V4 FEATURES
# --------------------------------------------------

def create_features(df):

    df = df.copy()

    # --------------------------------------------------
    # BASIC PRICE INFORMATION
    # --------------------------------------------------

    # Previous price
    df["Previous Price"] = (
        df["price"].shift(1)
    )

    # Price movement
    df["Price Delta"] = (
        df["price"] -
        df["Previous Price"]
    )

    # Absolute size of movement
    df["Absolute Delta"] = (
        df["Price Delta"].abs()
    )

    # --------------------------------------------------
    # ROLLING STEP VOLATILITY
    # --------------------------------------------------

    df["Step Volatility 5"] = (
        df["Price Delta"]
        .rolling(5)
        .std()
    )

    df["Step Volatility 10"] = (
        df["Price Delta"]
        .rolling(10)
        .std()
    )

    # --------------------------------------------------
    # DISTANCE FROM ROUND NUMBER
    # --------------------------------------------------

    # Decimal portion of price
    decimal_part = (
        df["price"] -
        np.floor(df["price"])
    )

    # Distance from nearest whole number
    df["Round Distance"] = np.minimum(
        decimal_part,
        1 - decimal_part
    )

    # --------------------------------------------------
    # DISTANCE FROM .5
    # --------------------------------------------------

    df["Half Distance"] = (
        (decimal_part - 0.5).abs()
    )

    # --------------------------------------------------
    # PREVIOUS DIGIT
    # --------------------------------------------------

    df["Previous Digit"] = (
        df["Last digit"].shift(1)
    )

    # --------------------------------------------------
    # TARGET
    # --------------------------------------------------

    df["Next Digit"] = (
        df["Last digit"].shift(-1)
    )

    return df


# --------------------------------------------------
# DISPLAY FEATURES
# --------------------------------------------------

def show_features(df):

    features = [
        "price",
        "Last digit",
        "Previous Price",
        "Price Delta",
        "Absolute Delta",
        "Step Volatility 5",
        "Step Volatility 10",
        "Round Distance",
        "Half Distance",
        "Previous Digit",
        "Next Digit"
    ]

    print(
        "\n## V4 FEATURES"
    )

    print("\n---")

    print(
        df[features].tail(20)
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(
        f"Raw rows: {len(df)}"
    )

    # Create features
    df = create_features(df)

    print(
        f"\nFeature rows: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    show_features(df)

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    # Make sure destination directory exists
    output_directory = os.path.dirname(
        FEATURE_V4_FILE
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    df.to_csv(
        FEATURE_V4_FILE,
        index=False
    )

    print(
        "\nSaved V4 features to:"
    )

    print(
        FEATURE_V4_FILE
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()