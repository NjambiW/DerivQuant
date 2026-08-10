import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config import FEATURE_FILE
import pandas as pd

CSV_FILE = "ticks_data.csv"


def load_data():
    """Load the tick data."""
    df = pd.read_csv(FEATURE_FILE)
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

def best_digit_prediction(df):

    print("\nBEST NEXT DIGIT PREDICTION")
    print("-----------------------------")

    table = pd.crosstab(
        df["Previous Digit"],
        df["Next Digit"],
        normalize="index"
    ) * 100

    for digit in table.index:

        best_digit = table.loc[digit].idxmax()
        probability = table.loc[digit].max()

        print(
            f"When previous digit is {digit}: "
            f"predict {best_digit} "
            f"({probability:.2f}%)"
        )

def prediction_accuracy(df):

    table = pd.crosstab(
        df["Previous Digit"],
        df["Next Digit"],
        normalize="index"
    )

    predictions = {}

    for digit in table.index:
        predictions[digit] = table.loc[digit].idxmax()


    correct = 0
    total = 0

    for _, row in df.iterrows():

        previous = row["Previous Digit"]
        actual = row["Next Digit"]

        prediction = predictions[previous]

        if prediction == actual:
            correct += 1

        total += 1


    accuracy = (correct / total) * 100

    print("\nMODEL ACCURACY")
    print("----------------")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")

def analyze_previous_parity(df):
    """
    Analyze the relationship between previous parity
    and next parity.
    """

    print("\nPREVIOUS PARITY → NEXT PARITY")
    print("--------------------------------")

    table = pd.crosstab(
        df["Previous Parity"],
        df["Next Parity"],
        normalize="index"
    ) * 100

    table.index = table.index.map({
        0: "Even",
        1: "Odd"
    })

    table.columns = ["Even", "Odd"]

    print(table.round(2))

    print("\nBEST PARITY PREDICTION")
    print("----------------------------")

    correct = 0
    total = 0

    for parity in [0, 1]:

        row = table.loc[
            "Even" if parity == 0 else "Odd"
        ]

        prediction = row.idxmax()
        probability = row.max()

        print(
            f"When previous parity is "
            f"{'Even' if parity == 0 else 'Odd'}: "
            f"predict {prediction} ({probability:.2f}%)"
        )

        actual = df.loc[
            df["Previous Parity"] == parity,
            "Next Parity"
        ]

        correct += (actual == (0 if prediction == "Even" else 1)).sum()
        total += len(actual)

    accuracy = (correct / total) * 100

    print("\nPARITY MODEL ACCURACY")
    print("----------------------------")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")

def analyze_digit_features(df):
    """
    Test how individual digit-based features
    relate to the next digit.
    """

    features = [
        "Previous Digit",
        "Digit 2 Ago",
        "Digit 3 Ago",
        "Digit 5 Ago",
        "Digit 10 Ago"
    ]

    print("\nFEATURE PERFORMANCE")
    print("----------------------------")

    for feature in features:

        print(f"\n{feature} → Next Digit")

        table = pd.crosstab(
            df[feature],
            df["Next Digit"],
            normalize="index"
        ) * 100

        best_predictions = table.max(axis=1)

        average_accuracy = best_predictions.mean()

        print(
            f"Average best prediction: "
            f"{average_accuracy:.2f}%"
        )

        print(
            f"Baseline: 10.00%"
        )

        print(
            f"Potential edge: "
            f"{average_accuracy - 10:.2f}%"
        )

def analyze_numeric_features(df):
    """
    Analyze rolling and numerical features
    against the next digit.
    """

    features = [
        "Digit Change",
        "Even Percentage 10",
        "Even Percentage 25",
        "Even Percentage 50",
        "Even Percentage 100"
    ]

    print("\nNUMERICAL FEATURE ANALYSIS")
    print("----------------------------")

    for feature in features:

        print(f"\n{feature} → Next Digit")

        grouped = (
            df.groupby(feature)["Next Digit"]
            .value_counts(normalize=True)
            .mul(100)
        )

        best_predictions = grouped.groupby(level=0).max()

        print(
            f"Average best prediction: "
            f"{best_predictions.mean():.2f}%"
        )

        print("Baseline: 10.00%")

        print(
            f"Potential difference: "
            f"{best_predictions.mean() - 10:.2f}%"
        )

def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

    df = df.dropna(
        subset=[
            "Previous Digit",
            "Next Digit",
            "Previous Parity",
            "Next Parity"
        ]
    )

    analyze_previous_digit(df)

    analyze_previous_parity(df)

    analyze_digit_features(df)

    analyze_numeric_features(df)


if __name__ == "__main__":
    main()
