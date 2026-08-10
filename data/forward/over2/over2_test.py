import os
import pandas as pd
from scipy.stats import binomtest


# ============================================================
# OVER 2 CONDITIONAL TEST
# ============================================================
#
# HYPOTHESIS:
#
# Low volatility
# + Previous digit is 0 OR 1
# -> Next digit is OVER 2
#
# OVER 2 means:
# 3, 4, 5, 6, 7, 8, 9
#
# Baseline probability = 70%
#
# PAPER / HISTORICAL TEST ONLY
# NO TRADES ARE PLACED
# ============================================================


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

OVER2_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(OVER2_DIR)

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    OVER2_DIR,
    "results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "over2_test_results.csv"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SYMBOL = "R_100"

VOLATILITY_THRESHOLD = 0.145358

PREVIOUS_DIGITS = [0, 1]

OVER2_DIGITS = [3, 4, 5, 6, 7, 8, 9]

BASELINE = 0.70

VOLATILITY_WINDOW = 20


# ============================================================
# DIGIT EXTRACTION
# ============================================================

def get_last_digit(price):

    formatted = f"{float(price):.2f}"

    return int(formatted[-1])


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("OVER 2 CONDITIONAL TEST")
print("=" * 70)

print()
print("Loading forward dataset...")

print()
print(f"Forward file: {FORWARD_FILE}")

if not os.path.isfile(FORWARD_FILE):

    raise FileNotFoundError(
        f"Forward dataset not found:\n{FORWARD_FILE}"
    )


df = pd.read_csv(
    FORWARD_FILE
)

print(
    f"Raw rows: {len(df)}"
)


# ============================================================
# CLEAN DATA
# ============================================================

required_columns = [
    "price"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df = df.dropna(
    subset=["price"]
).reset_index(
    drop=True
)


print(
    f"Usable rows: {len(df)}"
)


# ============================================================
# DIGITS
# ============================================================

df["digit"] = df["price"].apply(
    get_last_digit
)


# ============================================================
# DELTA
# ============================================================

df["delta"] = (
    df["price"]
    .diff()
)


# ============================================================
# VOLATILITY
# ============================================================

df["volatility"] = (
    df["delta"]
    .rolling(
        VOLATILITY_WINDOW
    )
    .std()
)


# ============================================================
# NEXT DIGIT
# ============================================================

df["next_digit"] = (
    df["digit"]
    .shift(-1)
)


# ============================================================
# OVER 2
# ============================================================

df["over_2"] = (
    df["next_digit"]
    .isin(OVER2_DIGITS)
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

usable = df.dropna(
    subset=[
        "volatility",
        "next_digit"
    ]
).copy()


# ============================================================
# HYPOTHESIS DESCRIPTION
# ============================================================

print()
print("=" * 70)

print("HYPOTHESIS")

print("=" * 70)

print()

print(
    "Low volatility"
)

print(
    f"Volatility <= {VOLATILITY_THRESHOLD:.6f}"
)

print()

print(
    "Previous digit: 0 OR 1"
)

print(
    "-> Predict NEXT DIGIT OVER 2"
)

print()

print(
    "Over 2 digits: "
    f"{OVER2_DIGITS}"
)

print(
    f"Baseline: {BASELINE:.2%}"
)


# ============================================================
# LOW VOLATILITY CONDITION
# ============================================================

low_volatility = (
    usable["volatility"]
    <= VOLATILITY_THRESHOLD
)


# ============================================================
# CONDITIONAL SIGNAL
# ============================================================

conditional = (
    low_volatility
    &
    usable["digit"].isin(
        PREVIOUS_DIGITS
    )
)


signals = usable[
    conditional
].copy()


print()
print("=" * 70)

print("SIGNAL SUMMARY")

print("=" * 70)

print()

print(
    f"Total usable observations: "
    f"{len(usable)}"
)

print(
    f"Low-volatility observations: "
    f"{low_volatility.sum()}"
)

print(
    f"Previous digit 0 or 1 + "
    f"low volatility signals: "
    f"{len(signals)}"
)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_condition(
    data,
    condition_name
):

    observations = len(data)

    if observations == 0:

        print()
        print(
            f"## {condition_name}"
        )

        print()
        print(
            "No observations."
        )

        return {
            "condition": condition_name,
            "observations": 0,
            "over_2": 0,
            "probability": None,
            "difference": None,
            "p_value": None
        }


    correct = int(
        data["over_2"].sum()
    )

    probability = (
        correct /
        observations
    )

    difference = (
        probability -
        BASELINE
    )


    result = binomtest(
        correct,
        observations,
        BASELINE,
        alternative="greater"
    )

    p_value = result.pvalue


    if p_value < 0.05:

        significance = (
            "STATISTICALLY SIGNIFICANT"
        )

    else:

        significance = (
            "NOT STATISTICALLY SIGNIFICANT"
        )


    print()
    print(
        f"## {condition_name}"
    )

    print()

    print(
        f"Observations: "
        f"{observations}"
    )

    print(
        f"Over 2: "
        f"{correct}"
    )

    print(
        f"Probability: "
        f"{probability:.2%}"
    )

    print(
        f"Difference: "
        f"{difference:+.2%}"
    )

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    print(
        f"Result: "
        f"{significance}"
    )


    return {
        "condition": condition_name,
        "observations": observations,
        "over_2": correct,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }


# ============================================================
# TEST 1
# PREVIOUS DIGIT 0 OR 1
# ============================================================

results = []


results.append(
    test_condition(
        signals,
        "LOW VOLATILITY + PREVIOUS DIGIT 0 OR 1"
    )
)


# ============================================================
# TEST 2
# PREVIOUS DIGIT 0
# ============================================================

digit_0 = signals[
    signals["digit"] == 0
].copy()


results.append(
    test_condition(
        digit_0,
        "LOW VOLATILITY + PREVIOUS DIGIT 0"
    )
)


# ============================================================
# TEST 3
# PREVIOUS DIGIT 1
# ============================================================

digit_1 = signals[
    signals["digit"] == 1
].copy()


results.append(
    test_condition(
        digit_1,
        "LOW VOLATILITY + PREVIOUS DIGIT 1"
    )
)


# ============================================================
# TEST 4
# ALL LOW VOLATILITY
# ============================================================

low_vol = usable[
    usable["volatility"]
    <= VOLATILITY_THRESHOLD
].copy()


results.append(
    test_condition(
        low_vol,
        "LOW VOLATILITY — ALL PREVIOUS DIGITS"
    )
)


# ============================================================
# TEST 5
# ALL DATA BASELINE CHECK
# ============================================================

results.append(
    test_condition(
        usable,
        "ALL USABLE OBSERVATIONS"
    )
)


# ============================================================
# BY PREVIOUS DIGIT
# ============================================================

print()
print("=" * 70)

print("OVER 2 PERFORMANCE BY PREVIOUS DIGIT")

print("=" * 70)

print()

print(
    "Previous | Observations | Over 2 | "
    "Probability | Difference | P-value"
)

print("-" * 70)


for digit in range(10):

    subset = low_vol[
        low_vol["digit"] == digit
    ]

    observations = len(subset)

    if observations == 0:

        continue

    correct = int(
        subset["over_2"].sum()
    )

    probability = (
        correct /
        observations
    )

    difference = (
        probability -
        BASELINE
    )

    p_value = binomtest(
        correct,
        observations,
        BASELINE,
        alternative="greater"
    ).pvalue


    print(
        f"{digit:8d} | "
        f"{observations:12d} | "
        f"{correct:6d} | "
        f"{probability:11.2%} | "
        f"{difference:+10.2%} | "
        f"{p_value:.6f}"
    )


# ============================================================
# BY HOUR
# ============================================================

print()
print("=" * 70)

print("OVER 2 PERFORMANCE BY HOUR")

print("=" * 70)

print()

if "time" in low_vol.columns:

    try:

        low_vol["datetime"] = pd.to_datetime(
            low_vol["time"],
            unit="s",
            errors="coerce"
        )

        low_vol["hour"] = (
            low_vol["datetime"]
            .dt.hour
        )


        print(
            "Hour | Signals | Over 2 | "
            "Probability | Difference | P-value"
        )

        print("-" * 70)


        for hour in range(24):

            subset = low_vol[
                low_vol["hour"] == hour
            ]

            observations = len(
                subset
            )

            if observations == 0:

                continue

            correct = int(
                subset["over_2"].sum()
            )

            probability = (
                correct /
                observations
            )

            difference = (
                probability -
                BASELINE
            )

            p_value = binomtest(
                correct,
                observations,
                BASELINE,
                alternative="greater"
            ).pvalue


            print(
                f"{hour:02d}:00 | "
                f"{observations:7d} | "
                f"{correct:6d} | "
                f"{probability:11.2%} | "
                f"{difference:+10.2%} | "
                f"{p_value:.6f}"
            )

    except Exception as e:

        print(
            f"Could not perform hourly analysis: {e}"
        )

else:

    print(
        "No time column found. "
        "Skipping hourly analysis."
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)

print("SUMMARY")

print("=" * 70)

print()

main_result = results[0]

print(
    f"Main condition:"
)

print(
    "Low volatility + previous digit 0 or 1"
)

print()

print(
    f"Observations: "
    f"{main_result['observations']}"
)

print(
    f"Over 2 occurrences: "
    f"{main_result['over_2']}"
)

print(
    f"Probability: "
    f"{main_result['probability']:.2%}"
)

print(
    f"Baseline: "
    f"{BASELINE:.2%}"
)

print(
    f"Difference: "
    f"{main_result['difference']:+.2%}"
)

print(
    f"P-value: "
    f"{main_result['p_value']:.6f}"
)


# ============================================================
# CONCLUSION
# ============================================================

print()
print("=" * 70)

print("CONCLUSION")

print("=" * 70)

print()


if (
    main_result["p_value"] < 0.05
    and
    main_result["probability"] > BASELINE
):

    print(
        "The low-volatility + previous digit "
        "0/1 condition shows statistically "
        "significant evidence above the "
        "70% Over 2 baseline."
    )

elif (
    main_result["probability"] > BASELINE
):

    print(
        "The condition is above the 70% "
        "baseline, but the evidence is "
        "not statistically significant."
    )

else:

    print(
        "The condition does not outperform "
        "the 70% Over 2 baseline."
    )


print()

print(
    "IMPORTANT:"
)

print(
    "This test does NOT establish a "
    "predictive mechanism or guarantee "
    "profitable trading."
)

print(
    "Any promising result should be "
    "validated on unseen data before "
    "being considered for live paper testing."
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


print()

print(
    f"Results saved to:\n"
    f"{RESULTS_FILE}"
)

print()

print(
    "Over 2 conditional test complete."
)