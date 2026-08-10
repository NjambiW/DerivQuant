
import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# H3 CONDITIONAL OUT-OF-SAMPLE TEST
#
# Frozen hypothesis:
#
# H3 + Previous Digit 4
# -> Predict Next Digit 3
#
# IMPORTANT:
# The condition is fixed BEFORE testing the OOS data.
# We do NOT search the OOS data for a better digit,
# threshold, or condition.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

H3_DIR = os.path.dirname(CURRENT_FILE)

CONDITIONAL_DIR = os.path.dirname(H3_DIR)

FORWARD_DIR = os.path.dirname(CONDITIONAL_DIR)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(FORWARD_DIR)
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

PREVIOUS_DIGIT = 4

BASELINE = 0.10

DELTA_THRESHOLD = -0.100

VOLATILITY_THRESHOLD = 0.145358

# Number of observations reserved before OOS testing

TRAINING_H3_OBSERVATIONS = 2500

# Size of each OOS block

OOS_BLOCK_SIZE = 500


# ============================================================
# FILE PATHS
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    CONDITIONAL_DIR,
    "results",
    "h3",
    "conditional_oos"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "conditional_oos_results.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("H3 CONDITIONAL OUT-OF-SAMPLE TEST")
print("=" * 70)

print()

print("Frozen hypothesis:")

print(
    "Large Negative Delta + Low Volatility"
)

print(
    "+ Previous Digit 4"
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
    f"Previous digit: {PREVIOUS_DIGIT}"
)

print(
    f"Target digit: {TARGET_DIGIT}"
)

print(
    f"Baseline: {BASELINE:.2%}"
)

print(
    f"Training H3 observations: "
    f"{TRAINING_H3_OBSERVATIONS}"
)

print(
    f"OOS block size: "
    f"{OOS_BLOCK_SIZE}"
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
        f"Expected: {FORWARD_FILE}"
    )

    sys.exit(1)


df = pd.read_csv(
    FORWARD_FILE
)

print(
    f"Raw rows: {len(df)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
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
    print(
        "ERROR: Missing required columns:"
    )

    print(
        missing_columns
    )

    sys.exit(1)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = (
    df
    .sort_values("time")
    .reset_index(drop=True)
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["Last digit"] = pd.to_numeric(
    df["Last digit"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = (
    df
    .dropna(
        subset=[
            "price",
            "Last digit"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# CREATE FEATURES
# ============================================================

# Previous price

df["previous_price"] = (
    df["price"].shift(1)
)


# Price change

df["delta"] = (
    df["price"]
    - df["previous_price"]
)


# Rolling volatility
#
# Standard deviation of the previous
# 10 price changes.

df["volatility"] = (
    df["delta"]
    .rolling(10)
    .std()
)


# Previous digit

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

df = (
    df
    .dropna(
        subset=[
            "delta",
            "volatility",
            "previous_digit",
            "next_digit"
        ]
    )
    .reset_index(drop=True)
)


df["previous_digit"] = (
    df["previous_digit"]
    .astype(int)
)

df["next_digit"] = (
    df["next_digit"]
    .astype(int)
)


# ============================================================
# CREATE ORIGINAL H3 CONDITION
# ============================================================

df["H3"] = (
    (df["delta"] <= DELTA_THRESHOLD)
    &
    (df["volatility"] <= VOLATILITY_THRESHOLD)
)


# ============================================================
# CREATE FROZEN CONDITIONAL HYPOTHESIS
# ============================================================

df["H3_CONDITIONAL"] = (
    df["H3"]
    &
    (
        df["previous_digit"]
        == PREVIOUS_DIGIT
    )
)


# ============================================================
# GET CONDITIONAL SIGNALS
# ============================================================

signals = df[
    df["H3_CONDITIONAL"]
].copy()


print()

print(
    f"Usable rows: {len(df)}"
)

print(
    f"Conditional H3 observations found: "
    f"{len(signals)}"
)


# ============================================================
# CHECK SAMPLE SIZE
# ============================================================

minimum_required = (
    TRAINING_H3_OBSERVATIONS
    +
    OOS_BLOCK_SIZE
)

if len(signals) < minimum_required:

    print()
    print(
        "Not enough conditional observations "
        "for the requested OOS test."
    )

    print(
        f"Required: {minimum_required}"
    )

    print(
        f"Available: {len(signals)}"
    )

    sys.exit(0)


# ============================================================
# TRAINING / OOS SPLIT
# ============================================================

training = signals[
    :TRAINING_H3_OBSERVATIONS
].copy()

oos = signals[
    TRAINING_H3_OBSERVATIONS:
].copy()


print()

print(
    "============================================================"
)

print(
    "FROZEN TRAINING / OOS SPLIT"
)

print(
    "============================================================"
)

print()

print(
    f"Training observations: "
    f"{len(training)}"
)

print(
    f"OOS observations available: "
    f"{len(oos)}"
)


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

training_correct = (
    training["next_digit"]
    == TARGET_DIGIT
).sum()

training_probability = (
    training_correct
    / len(training)
)

training_difference = (
    training_probability
    - BASELINE
)

training_test = binomtest(
    training_correct,
    len(training),
    BASELINE,
    alternative="greater"
)

training_p = (
    training_test.pvalue
)


print()

print(
    "## TRAINING PERFORMANCE"
)

print()

print(
    f"Training observations: "
    f"{len(training)}"
)

print(
    f"Correct: "
    f"{training_correct}"
)

print(
    f"Accuracy: "
    f"{training_probability:.2%}"
)

print(
    f"Difference: "
    f"{training_difference:+.2%}"
)

print(
    f"P-value: "
    f"{training_p:.6f}"
)


# ============================================================
# OOS BLOCK TEST
# ============================================================

print()

print(
    "============================================================"
)

print(
    "## OUT-OF-SAMPLE BLOCK RESULTS"
)

print(
    "============================================================"
)

print()

print(
    "## Block | Train | Test | Correct | "
    "Accuracy | Difference | P-value"
)

results = []

total_oos_correct = 0

total_oos_observations = 0

block_number = 1

start = 0


while (
    start + OOS_BLOCK_SIZE
    <= len(oos)
):

    end = (
        start
        + OOS_BLOCK_SIZE
    )

    block = oos[
        start:end
    ].copy()

    correct = (
        block["next_digit"]
        == TARGET_DIGIT
    ).sum()

    observations = len(block)

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

    p_value = (
        test.pvalue
    )

    print(
        f"OOS Block {block_number:2d} | "
        f"Train: "
        f"{TRAINING_H3_OBSERVATIONS + start:5d} | "
        f"Test: "
        f"{observations:4d} | "
        f"Correct: "
        f"{correct:3d} | "
        f"Accuracy: "
        f"{probability:6.2%} | "
        f"Difference: "
        f"{difference:+6.2%} | "
        f"P-value: "
        f"{p_value:.6f}"
    )

    results.append(
        {
            "block": block_number,
            "training_observations": (
                TRAINING_H3_OBSERVATIONS
                + start
            ),
            "test_observations": observations,
            "correct": correct,
            "probability": probability,
            "difference": difference,
            "p_value": p_value
        }
    )

    total_oos_correct += correct

    total_oos_observations += observations

    start = end

    block_number += 1


# ============================================================
# COMBINED OOS PERFORMANCE
# ============================================================

if total_oos_observations == 0:

    print()
    print(
        "No complete OOS block available."
    )

    sys.exit(0)


combined_probability = (
    total_oos_correct
    / total_oos_observations
)

combined_difference = (
    combined_probability
    - BASELINE
)

combined_test = binomtest(
    total_oos_correct,
    total_oos_observations,
    BASELINE,
    alternative="greater"
)

combined_p = (
    combined_test.pvalue
)


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


blocks_tested = len(
    results_df
)

blocks_above_baseline = (
    results_df["probability"]
    > BASELINE
).sum()

significant_blocks = (
    (
        results_df["probability"]
        > BASELINE
    )
    &
    (
        results_df["p_value"]
        < 0.05
    )
).sum()


print()

print(
    "============================================================"
)

print(
    "## OUT-OF-SAMPLE SUMMARY"
)

print(
    "============================================================"
)

print()

print(
    f"OOS blocks tested: "
    f"{blocks_tested}"
)

print(
    f"Blocks above baseline: "
    f"{blocks_above_baseline}"
    f"/{blocks_tested}"
)

print(
    f"Statistically significant blocks: "
    f"{significant_blocks}"
    f"/{blocks_tested}"
)

print(
    f"Mean OOS probability: "
    f"{results_df['probability'].mean():.2%}"
)

print(
    f"Minimum OOS probability: "
    f"{results_df['probability'].min():.2%}"
)

print(
    f"Maximum OOS probability: "
    f"{results_df['probability'].max():.2%}"
)

print(
    f"Median OOS probability: "
    f"{results_df['probability'].median():.2%}"
)


# ============================================================
# COMBINED OOS
# ============================================================

print()

print(
    "============================================================"
)

print(
    "## COMBINED OOS PERFORMANCE"
)

print(
    "============================================================"
)

print()

print(
    f"Total OOS observations: "
    f"{total_oos_observations}"
)

print(
    f"Total correct: "
    f"{total_oos_correct}"
)

print(
    f"OOS probability: "
    f"{combined_probability:.2%}"
)

print(
    f"Baseline: "
    f"{BASELINE:.2%}"
)

print(
    f"Difference: "
    f"{combined_difference:+.2%}"
)

print(
    f"P-value: "
    f"{combined_p:.6f}"
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
# CONCLUSION
# ============================================================

print()

print(
    "============================================================"
)

print(
    "## CONCLUSION"
)

print(
    "============================================================"
)

print()

if (
    combined_probability > BASELINE
    and combined_p < 0.05
):

    print(
        "H3 + Previous Digit 4 PASSES "
        "the combined out-of-sample test."
    )

    print()

    print(
        "The conditional effect remains "
        "statistically above the 10% baseline "
        "on unseen observations."
    )

elif (
    combined_probability > BASELINE
):

    print(
        "H3 + Previous Digit 4 remains "
        "above baseline in the combined "
        "out-of-sample data."
    )

    print()

    print(
        "However, the OOS evidence is not "
        "statistically significant."
    )

else:

    print(
        "H3 + Previous Digit 4 does NOT "
        "remain above the 10% baseline "
        "in the combined OOS data."
    )

print()

print(
    "IMPORTANT:"
)

print(
    "This test evaluates a frozen hypothesis."
)

print(
    "No thresholds, target digits, or "
    "previous-digit conditions were selected "
    "from the OOS observations."
)

print(
    "A successful OOS result would justify "
    "additional validation, not immediate "
    "deployment for trading."
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
    "Conditional OOS test complete."
)
