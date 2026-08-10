import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

DEPENDENCE_DIR = os.path.dirname(CURRENT_FILE)
FEATURES_DIR = os.path.dirname(DEPENDENCE_DIR)
SRC_DIR = os.path.dirname(FEATURES_DIR)

sys.path.insert(0, SRC_DIR)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pandas as pd

from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

from config import TICKS_FILE


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    return pd.read_csv(TICKS_FILE)


# --------------------------------------------------
# LAG ANALYSIS
# --------------------------------------------------

def analyze_lag(df, lag):

    current = df["Last digit"]

    previous = df["Last digit"].shift(lag)

    data = pd.DataFrame({
        "Previous": previous,
        "Current": current
    }).dropna()

    # Convert to integers
    data["Previous"] = data["Previous"].astype(int)
    data["Current"] = data["Current"].astype(int)

    # --------------------------------------------------
    # CHI-SQUARE TEST
    # --------------------------------------------------

    table = pd.crosstab(
        data["Previous"],
        data["Current"]
    )

    chi2, p_value, dof, expected = (
        chi2_contingency(table)
    )

    # --------------------------------------------------
    # MUTUAL INFORMATION
    # --------------------------------------------------

    mutual_information = mutual_info_score(
        data["Previous"],
        data["Current"]
    )

    return (
        len(data),
        chi2,
        p_value,
        mutual_information
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    print(
        f"Total rows: {len(df)}"
    )

    print(
        "\n## DIGIT DEPENDENCE ANALYSIS"
    )

    print("\n---")

    lags = [
        1,
        2,
        3,
        5,
        10
    ]

    results = []

    for lag in lags:

        (
            observations,
            chi2,
            p_value,
            mutual_information
        ) = analyze_lag(
            df,
            lag
        )

        results.append({
            "Lag": lag,
            "Observations": observations,
            "Chi-Square": chi2,
            "P-Value": p_value,
            "Mutual Information": mutual_information
        })

        print(
            f"\nLag {lag}"
        )

        print(
            f"Observations: "
            f"{observations}"
        )

        print(
            f"Chi-square: "
            f"{chi2:.4f}"
        )

        print(
            f"P-value: "
            f"{p_value:.6f}"
        )

        print(
            f"Mutual information: "
            f"{mutual_information:.6f}"
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n## DEPENDENCE SUMMARY"
    )

    print("\n---")

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    print(
        "\n## INTERPRETATION"
    )

    print("\n---")

    for _, row in results_df.iterrows():

        lag = int(row["Lag"])
        p_value = row["P-Value"]

        if p_value < 0.05:

            print(
                f"Lag {lag}: "
                f"Statistically significant "
                f"dependence detected."
            )

        else:

            print(
                f"Lag {lag}: "
                f"No statistically significant "
                f"dependence detected."
            )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()
