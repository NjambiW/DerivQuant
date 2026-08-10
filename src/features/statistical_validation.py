import sys
import os

# Allow Python to find config.py
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest

from config import FEATURE_FILE


def load_data():
    return pd.read_csv(FEATURE_FILE)


def main():

    df = load_data()

    print(f"Total rows: {len(df)}")

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

    df = df.dropna(
        subset=features + [target]
    )

    print(f"Usable rows: {len(df)}")

    X = df[features]
    y = df[target]

    # Chronological 80/20 split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    correct = (predictions == y_test.values).sum()
    total = len(y_test)

    accuracy = correct / total

    baseline = 0.10

    # Binomial test
    test = binomtest(
        correct,
        total,
        baseline,
        alternative="greater"
    )

    p_value = test.pvalue

    # Confidence interval
    confidence_interval = test.proportion_ci(
        confidence_level=0.95
    )

    lower = confidence_interval.low
    upper = confidence_interval.high

    expected_correct = total * baseline

    observed_difference = accuracy - baseline

    print("\n## STATISTICAL VALIDATION")
    print("-------------------------")

    print(
        f"Correct predictions: "
        f"{correct}/{total}"
    )

    print(
        f"Observed accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Random baseline: "
        f"{baseline * 100:.2f}%"
    )

    print(
        f"Expected correct at baseline: "
        f"{expected_correct:.0f}"
    )

    print(
        f"Observed advantage: "
        f"{observed_difference * 100:.2f} percentage points"
    )

    print(
        f"95% confidence interval: "
        f"{lower * 100:.2f}% - "
        f"{upper * 100:.2f}%"
    )

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    print("\n## CONCLUSION")
    print("-------------")

    if p_value < 0.05:
        print(
            "The model shows statistically significant "
            "evidence of accuracy above the 10% baseline."
        )
    else:
        print(
            "The model does NOT show statistically significant "
            "evidence of accuracy above the 10% baseline."
        )


if __name__ == "__main__":
    main()