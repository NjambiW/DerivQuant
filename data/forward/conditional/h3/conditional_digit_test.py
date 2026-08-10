
import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# H3 CONDITIONAL DIGIT TEST
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)

# Move upward until we reach:
# data/forward
#
# conditional/h3/conditional_digit_test.py
#        ↑
#       h3
#        ↑
#   conditional
#        ↑
#     forward
#
CONDITIONAL_DIR = os.path.dirname(FORWARD_DIR)
FORWARD_ROOT = os.path.dirname(CONDITIONAL_DIR)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(FORWARD_ROOT)
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_DIR)

# ============================================================
# SETTINGS
# ============================================================

TARGET_DIGIT = 3
BASELINE = 0.10

DELTA_THRESHOLD = -0.100
VOLATILITY_THRESHOLD = 0.145358

FORWARD_FILE = os.path.join(
    FORWARD_ROOT,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    FORWARD_ROOT,
    "conditional",
    "results",
    "h3"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "conditional_digit_results.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading forward dataset...")
print()

print(
    f"Forward file: {FORWARD_FILE}"
)

if not os.path.isfile(FORWARD_FILE):

    print()
    print("ERROR: Forward dataset not found.")
    print(
        f"Expected file: {FORWARD_FILE}"
    )
    sys.exit(1)

df = pd.read_csv(FORWARD_FILE)

print(
    f"Raw rows: {len(df)}"
)

# ============================================================
# PREPARE DATA
# ============================================================

required_columns = [
    "time",
    "symbol",
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
    print("ERROR: Missing required columns:")
    print(missing_columns)
    sys.exit(1)

df = df.sort_values("time").reset_index(drop=True)

# Make sure price and digit are numeric

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["Last digit"] = pd.to_numeric(
    df["Last digit"],
    errors="coerce"
)

# Remove invalid rows

df = df.dropna(
    subset=[
        "price",
        "Last digit"
    ]
).reset_index(drop=True)

# ============================================================
# CREATE FEATURES
# ============================================================

# Previous price

df["previous_price"] = (
    df["price"].shift(1)
)

# Price delta

df["delta"] = (
    df["price"]
    - df["previous_price"]
)

# Rolling volatility
#
# We use the standard deviation of the
# previous 10 price changes.

df["volatility"] = (
    df["delta"]
    .rolling(10)
    .std()
)

# Previous tick digit

df["previous_digit"] = (
    df["Last digit"].shift(1)
)

# Next digit

df["next_digit"] = (
    df["Last digit"].shift(-1)
)

# ============================================================
# REMOVE INVALID FEATURE ROWS
# ============================================================

df = df.dropna(
    subset=[
        "delta",
        "volatility",
        "previous_digit",
        "next_digit"
    ]
).reset_index(drop=True)

df["previous_digit"] = (
    df["previous_digit"].astype(int)
)

df["next_digit"] = (
    df["next_digit"].astype(int)
)

# ============================================================
# H3 CONDITION
# ============================================================

df["H3"] = (
    (df["delta"] <= DELTA_THRESHOLD)
    &
    (df["volatility"] <= VOLATILITY_THRESHOLD)
)

h3 = df[df["H3"]].copy()

print(
    f"Usable rows: {len(df)}"
)

print(
    f"H3 observations found: {len(h3)}"
)

print()
print("Hypothesis:")
print(
    "Large Negative Delta + Low Volatility"
)
print(
    "-> Predict Next Digit 3"
)

print()
print(
    f"Large negative delta <= "
    f"{DELTA_THRESHOLD:.6f}"
)

print(
    f"Low volatility <= "
    f"{VOLATILITY_THRESHOLD:.6f}"
)

print(
    f"Target digit: {TARGET_DIGIT}"
)

print(
    f"Baseline: {BASELINE:.2%}"
)

# ============================================================
# FUNCTION FOR BINOMIAL TEST
# ============================================================

def analyze_group(
    group,
    label
):

    observations = len(group)

    if observations == 0:

        print()
        print(
            f"{label}: No observations."
        )

        return {
            "condition": label,
            "observations": 0,
            "digit_3": 0,
            "probability": 0,
            "difference": 0,
            "p_value": 1
        }

    correct = (
        group["next_digit"]
        == TARGET_DIGIT
    ).sum()

    probability = (
        correct
        / observations
    )

    difference = (
        probability
        - BASELINE
    )

    test = binomtest(
        correct,
        observations,
        BASELINE,
        alternative="greater"
    )

    p_value = test.pvalue

    result = (
        "STATISTICALLY SIGNIFICANT"
        if p_value < 0.05
        and probability > BASELINE
        else
        "NOT STATISTICALLY SIGNIFICANT"
    )

    print()
    print(
        f"## {label}"
    )

    print(
        f"Observations: {observations}"
    )

    print(
        f"Digit {TARGET_DIGIT}: {correct}"
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
        f"Result: {result}"
    )

    return {
        "condition": label,
        "observations": observations,
        "digit_3": correct,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }


# ============================================================
# OVERALL H3
# ============================================================

results = []

overall_result = analyze_group(
    h3,
    "ALL H3 SIGNALS"
)

results.append(
    overall_result
)

# ============================================================
# CONDITIONAL TEST BY PREVIOUS DIGIT
# ============================================================

print()
print("=" * 70)
print("## H3 CONDITIONAL ON PREVIOUS DIGIT")
print("=" * 70)

print()
print(
    "This tests whether H3 remains effective"
)

print(
    "when conditioned on the previous tick digit."
)

print()

for previous_digit in range(10):

    group = h3[
        h3["previous_digit"]
        == previous_digit
    ]

    result = analyze_group(
        group,
        f"PREVIOUS DIGIT {previous_digit}"
    )

    result["previous_digit"] = (
        previous_digit
    )

    results.append(result)

# ============================================================
# GROUPED PREVIOUS DIGITS
# ============================================================

print()
print("=" * 70)
print("## GROUPED PREVIOUS-DIGIT CONDITIONS")
print("=" * 70)

print()
print(
    "Low previous digits: 0-2"
)

low_digits = h3[
    h3["previous_digit"].isin(
        [0, 1, 2]
    )
]

result = analyze_group(
    low_digits,
    "PREVIOUS DIGIT 0-2"
)

result["previous_digit"] = "0-2"

results.append(result)

print()
print(
    "Middle previous digits: 3-6"
)

middle_digits = h3[
    h3["previous_digit"].isin(
        [3, 4, 5, 6]
    )
]

result = analyze_group(
    middle_digits,
    "PREVIOUS DIGIT 3-6"
)

result["previous_digit"] = "3-6"

results.append(result)

print()
print(
    "High previous digits: 7-9"
)

high_digits = h3[
    h3["previous_digit"].isin(
        [7, 8, 9]
    )
]

result = analyze_group(
    high_digits,
    "PREVIOUS DIGIT 7-9"
)

result["previous_digit"] = "7-9"

results.append(result)

# ============================================================
# SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("## CONDITIONAL DIGIT SUMMARY")
print("=" * 70)

print()

valid_results = results_df[
    results_df["observations"] > 0
]

above_baseline = valid_results[
    valid_results["probability"]
    > BASELINE
]

significant = valid_results[
    (valid_results["probability"] > BASELINE)
    &
    (valid_results["p_value"] < 0.05)
]

print(
    f"Conditions tested: "
    f"{len(valid_results)}"
)

print(
    f"Conditions above baseline: "
    f"{len(above_baseline)}"
    f"/{len(valid_results)}"
)

print(
    f"Statistically significant: "
    f"{len(significant)}"
    f"/{len(valid_results)}"
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
# BEST CONDITIONAL GROUP
# ============================================================

conditional_results = results_df[
    results_df["condition"]
    != "ALL H3 SIGNALS"
].copy()

conditional_results = (
    conditional_results[
        conditional_results["observations"] > 0
    ]
)

if not conditional_results.empty:

    best = conditional_results.loc[
        conditional_results["probability"].idxmax()
    ]

    print()
    print("=" * 70)
    print("## BEST CONDITIONAL RESULT")
    print("=" * 70)

    print()

    print(
        f"Condition: "
        f"{best['condition']}"
    )

    print(
        f"Observations: "
        f"{int(best['observations'])}"
    )

    print(
        f"Digit {TARGET_DIGIT}: "
        f"{int(best['digit_3'])}"
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
# CONCLUSION
# ============================================================

print()
print("=" * 70)
print("## CONCLUSION")
print("=" * 70)

print()

if len(significant) == 0:

    print(
        "H3 does not show statistically significant "
        "conditional evidence across the tested "
        "previous-digit groups."
    )

elif len(significant) == 1:

    print(
        "One conditional group shows statistically "
        "significant evidence above the 10% baseline."
    )

    print(
        "This is worth investigating further, but "
        "should not yet be treated as a new rule."
    )

else:

    print(
        f"{len(significant)} conditional groups show "
        "statistically significant evidence above "
        "the 10% baseline."
    )

    print(
        "This suggests the previous digit may contain "
        "additional information worth testing."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "A significant conditional group does not prove "
    "causation or a PRNG mechanism."
)

print(
    "We should validate any apparent effect on "
    "unseen data before changing H3."
)

print()
print(
    f"Results saved to:"
)

print(
    RESULTS_FILE
)

print()
print(
    "Conditional digit test complete."
)

