import pandas as pd
from scipy.stats import chisquare

def loadData(limit =None):
    """loads tick data"""
    df = pd.read_csv("ticks_data.csv")

    if limit is not None:
        df = df.tail(limit)
    return df

def parityAnalysis(df):
    """checks for even and odd numbers"""
    df["Parity"] = df["Last digit"].apply(
        lambda x: "Even" if x % 2 == 0 else "Odd"
    )

    parity = df["Parity"].value_counts(normalize=True) * 100
    print("\n EVEN /ODD")
    print("------------------")
    print(parity)

def digitAnalysis(df):
    counts = df["Last digit"].value_counts().sort_index()

    percentages = (
        df["Last digit"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    print("\n DIGIT DISTRIBUTION")
    print("-----------------------")

    for digit in counts.index:
        print(f"{digit:<8}{counts[digit]:<10}{percentages[digit]:.2f}%")

def chiSquareTest(df):
    observed = df["Last digit"].value_counts().sort_index()

    expected = [len(df)/10] * 10

    chi2, p = chisquare(observed, expected)

    print("Chi-square statistic:", chi2)
    print("P-value:", p)


def longestParityStreak(df):

    parity_list = df["Parity"].tolist()

    longest = 0
    current = 0
    previous = None

    for value in parity_list:
        if value == previous:
            current += 1
        else:
            current = 1

        if current > longest:
            longest = current

        previous = value

    print("Longest parity streak:", longest)

def main():
    df = loadData(limit=5000)

    parityAnalysis(df)
    digitAnalysis(df)
    chiSquareTest(df)
    longestParityStreak(df)


if __name__ == "__main__":
    main()