import pandas as pd


CSV_FILE = "ticks_data.csv"
FEATURE_FILE = "feature_data.csv"


def load_data():
    """Load tick data from CSV."""

    df = pd.read_csv(CSV_FILE)

    return df


def create_features(df):
    """Create features from tick data."""

    # Previous digits
    df["Previous Digit"] = df["Last digit"].shift(1)
    df["Digit 2 Ago"] = df["Last digit"].shift(2)
    df["Digit 3 Ago"] = df["Last digit"].shift(3)
    df["Digit 5 Ago"] = df["Last digit"].shift(5)
    df["Digit 10 Ago"] = df["Last digit"].shift(10)

    # Current parity
    # 0 = Even
    # 1 = Odd
    df["Parity"] = df["Last digit"] % 2

    # Previous parity
    df["Previous Parity"] = df["Previous Digit"] % 2

    # Change from previous digit
    df["Digit Change"] = (
        df["Last digit"] - df["Previous Digit"]
    )

    # Rolling digit statistics

    # Prediction-safe rolling statistics
    # We shift by 1 so the current tick is never
    # included in its own features.

    previous_parity = df["Parity"].shift(1)

    # Rolling windows
    for window in [10, 25, 50, 100]:
        # Even count
        df[f"Even Count {window}"] = (
            (previous_parity == 0)
            .rolling(window)
            .sum()
        )

        # Odd count
        df[f"Odd Count {window}"] = (
            (previous_parity == 1)
            .rolling(window)
            .sum()
        )

        # Even percentage
        df[f"Even Percentage {window}"] = (
                (previous_parity == 0)
                .rolling(window)
                .mean()
                * 100
        )
        # Target: the digit of the NEXT tick
        df["Next Digit"] = df["Last digit"].shift(-1)

    return df


def main():

    df = load_data()

    df = create_features(df)

    # Save feature data
    df.to_csv(FEATURE_FILE, index=False)

    print("\nFEATURE DATA")
    print("------------------")

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nLatest 20 ticks:")

    # to_string() prevents pandas from hiding columns
    print(df.tail(20).to_string())


if __name__ == "__main__":
    main()