import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# H3 SENSITIVITY TEST
# Tests whether H3 remains effective when its thresholds change
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

SENSITIVITY_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(SENSITIVITY_DIR)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "R_100"

TARGET_DIGIT = 3
BASELINE = 0.10

# Original H3 thresholds
ORIGINAL_DELTA = -0.100000
ORIGINAL_VOLATILITY = 0.145358

# ============================================================
# FILE PATH
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    SENSITIVITY_DIR,
    "results",
    "h3"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "h3_sensitivity_results.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# TEST THRESHOLDS
# ============================================================

# Delta thresholds:
# More negative = stricter condition
DELTA_THRESHOLDS = [
    -0.05,
    -0.075,
    -0.10,
    -0.125,
    -0.15,
    -0.20
]

# Volatility thresholds:
# Lower = stricter low-volatility condition
VOLATILITY_THRESHOLDS = [
    0.10,
    0.12,
    0.145358,
    0.17,
    0.20
]

# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading forward dataset...")
print()

if not os.path.isfile(FORWARD_FILE):

    print("ERROR:")
    print("Forward dataset not found:")
    print(FORWARD_FILE)

    sys.exit(1)

df = pd.read_csv(FORWARD_FILE)

print(
    f"Forward file: {FORWARD_FILE}"
)

print(
    f"Raw rows: {len(df)}"
)

# ============================================================
# PREPARE DATA
# ============================================================

required_columns = [
    "time",
    "price",
    "Last digit"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print()
    print("ERROR: Missing columns:")
    print(missing_columns)

    sys.exit(1)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["Last digit"] = pd.to_numeric(
    df["Last digit"],
    errors="coerce"
)

df["time"] = pd.to_numeric(
    df["time"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "price",
        "Last digit",
        "time"
    ]
).copy()

df = df.reset_index(drop=True)

# ============================================================
# CREATE FEATURES
# ============================================================

# Price change from previous tick

df["delta"] = (
    df["price"].diff()
)

# Rolling volatility
#
# This uses the standard deviation of the
# previous 10 price changes.

df["volatility"] = (
    df["delta"]
    .rolling(10)
    .std()
)

# Next tick's digit

df["next_digit"] = (
    df["Last digit"]
    .shift(-1)
)

# Remove rows where features/target cannot be calculated

df = df.dropna(
    subset=[
        "delta",
        "volatility",
        "next_digit"
    ]
).copy()

df["next_digit"] = (
    df["next_digit"].astype(int)
)

print(
    f"Usable rows: {len(df)}"
)

# ============================================================
# TEST FUNCTION
# ============================================================

def run_test(
    delta_threshold,
    volatility_threshold
):

    condition = (
        (df["delta"] <= delta_threshold)
        &
        (df["volatility"] <= volatility_threshold)
    )

    subset = df.loc[
        condition
    ].copy()

    observations = len(subset)

    if observations == 0:

        return {
            "delta_threshold": delta_threshold,
            "volatility_threshold": volatility_threshold,
            "observations": 0,
            "correct": 0,
            "probability": 0,
            "difference": 0,
            "p_value": 1
        }

    correct = int(
        (
            subset["next_digit"]
            == TARGET_DIGIT
        ).sum()
    )

    probability = (
        correct / observations
    )

    difference = (
        probability - BASELINE
    )

    test = binomtest(
        correct,
        observations,
        BASELINE,
        alternative="greater"
    )

    p_value = test.pvalue

    return {
        "delta_threshold": delta_threshold,
        "volatility_threshold": volatility_threshold,
        "observations": observations,
        "correct": correct,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }


# ============================================================
# RUN SENSITIVITY TESTS
# ============================================================

print()
print("============================================================")
print("H3 SENSITIVITY ANALYSIS")
print("============================================================")

print()
print("Original H3:")
print(
    f"Large negative delta <= "
    f"{ORIGINAL_DELTA:.6f}"
)

print(
    f"Low volatility <= "
    f"{ORIGINAL_VOLATILITY:.6f}"
)

print(
    f"Target digit: {TARGET_DIGIT}"
)

print(
    f"Baseline: {BASELINE:.2%}"
)

print()
print("Testing different delta and volatility thresholds...")
print()

results = []

test_number = 0

for delta_threshold in DELTA_THRESHOLDS:

    for volatility_threshold in VOLATILITY_THRESHOLDS:

        test_number += 1

        result = run_test(
            delta_threshold,
            volatility_threshold
        )

        results.append(result)

        probability = result["probability"]

        difference = result["difference"]

        p_value = result["p_value"]

        observations = result["observations"]

        print(
            f"{test_number:2d} | "
            f"Delta <= {delta_threshold:7.3f} | "
            f"Vol <= {volatility_threshold:8.6f} | "
            f"N={observations:5d} | "
            f"P={probability:6.2%} | "
            f"Diff={difference:+6.2%} | "
            f"p={p_value:.6f}"
        )

# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

# ============================================================
# SUMMARY
# ============================================================

valid_results = results_df[
    results_df["observations"] > 0
]

above_baseline = valid_results[
    valid_results["probability"]
    > BASELINE
]

significant = valid_results[
    valid_results["p_value"]
    < 0.05
]

print()
print("============================================================")
print("SENSITIVITY SUMMARY")
print("============================================================")

print()

print(
    f"Threshold combinations tested: "
    f"{len(valid_results)}"
)

print(
    f"Combinations above 10%: "
    f"{len(above_baseline)}/"
    f"{len(valid_results)}"
)

print(
    f"Statistically significant: "
    f"{len(significant)}/"
    f"{len(valid_results)}"
)

if len(valid_results) > 0:

    print(
        f"Mean probability: "
        f"{valid_results['probability'].mean():.2%}"
    )

    print(
        f"Minimum probability: "
        f"{valid_results['probability'].min():.2%}"
    )

    print(
        f"Maximum probability: "
        f"{valid_results['probability'].max():.2%}"
    )

# ============================================================
# BEST THRESHOLD COMBINATION
# ============================================================

if len(valid_results) > 0:

    best = valid_results.loc[
        valid_results["probability"].idxmax()
    ]

    print()
    print("============================================================")
    print("BEST OBSERVED THRESHOLD COMBINATION")
    print("============================================================")

    print()

    print(
        f"Delta threshold: "
        f"{best['delta_threshold']:.6f}"
    )

    print(
        f"Volatility threshold: "
        f"{best['volatility_threshold']:.6f}"
    )

    print(
        f"Observations: "
        f"{int(best['observations'])}"
    )

    print(
        f"Correct: "
        f"{int(best['correct'])}"
    )

    print(
        f"Probability: "
        f"{best['probability']:.2%}"
    )

    print(
        f"Difference: "
        f"{best['difference']:+.2%}"
    )

    print(
        f"P-value: "
        f"{best['p_value']:.6f}"
    )

# ============================================================
# ORIGINAL H3 RESULT
# ============================================================

original_result = run_test(
    ORIGINAL_DELTA,
    ORIGINAL_VOLATILITY
)

print()
print("============================================================")
print("ORIGINAL H3")
print("============================================================")

print()

print(
    f"Observations: "
    f"{original_result['observations']}"
)

print(
    f"Digit {TARGET_DIGIT}: "
    f"{original_result['correct']}"
)

print(
    f"Probability: "
    f"{original_result['probability']:.2%}"
)

print(
    f"Difference: "
    f"{original_result['difference']:+.2%}"
)

print(
    f"P-value: "
    f"{original_result['p_value']:.6f}"
)

# ============================================================
# CONCLUSION
# ============================================================

print()
print("============================================================")
print("CONCLUSION")
print("============================================================")

print()

if len(significant) >= 3:

    print(
        "H3 appears relatively robust to threshold changes."
    )

    print(
        "Multiple threshold combinations remain "
        "statistically significant."
    )

elif len(above_baseline) >= (
    len(valid_results) * 0.60
):

    print(
        "H3 shows moderate threshold stability."
    )

    print(
        "The effect remains above baseline across "
        "many threshold combinations."
    )

else:

    print(
        "H3 appears sensitive to threshold selection."
    )

    print(
        "The observed effect weakens substantially "
        "when the thresholds change."
    )

print()
print(
    "IMPORTANT: Sensitivity does not prove a predictive mechanism."
)

print(
    "A robust signal should survive reasonable threshold "
    "changes without relying on one specially selected combination."
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print()
print(
    "Results saved to:"
)

print(
    RESULTS_FILE
)

print()
print(
    "H3 sensitivity test complete."
)