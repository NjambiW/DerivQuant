import pandas as pd

CSV_FILE = "ticks_data.csv"


def load_data():
    """Load the tick data."""
    df = pd.read_csv(CSV_FILE)
    return df


def analyze_previous_digit(df):
    """
    Analyze the relationship between the previous digit
    and the next digit.
    """

    print("\nPREVIOUS DIGIT → NEXT DIGIT")
    print("-----------------------------")

    table = pd.crosstab(
        df["Previous Digit"],
        df["Next Digit"],
        normalize="index"
    ) * 100

    print(table.round(2))


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    # Remove rows where the target does not exist
    df = df.dropna(subset=["Previous Digit", "Next Digit"])

    analyze_previous_digit(df)


if __name__ == "__main__":
    main()