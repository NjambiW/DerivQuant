import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# H3 CONDITIONAL DATA ACCUMULATION TEST
#
# Frozen hypothesis:
#
# H3 + Previous Digit 4
# -> Predict Next Digit 3
#
# This is NOT an OOS test.
# It monitors whether we are accumulating enough
# qualifying observations for a proper OOS test.
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

# Target number of conditional observations
# before we perform the proper OOS test.

TARGET_CONDITIONAL_OBSERVATIONS = 3000


# ============================================================
# FILE PATH
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("H3 CONDITIONAL DATA ACCUMULATION")
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
    f"Target conditional observations: "
    f"{TARGET_CONDITIONAL_OBSERVATIONS}"
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
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "time",
    "symbol",
    "price",
    "Last digit"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    print()
    print(
        "ERROR: Missing columns:"
    )

    print(missing)

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
# CONVERT NUMERIC COLUMNS
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
# REMOVE INVALID DATA
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

df["previous_price"] = (
    df["price"].shift(1)
)

df["delta"] = (
    df["price"]
    - df["previous_price"]
)

df["volatility"] = (
    df["delta"]
    .rolling(10)
    .std()
)

df["previous_digit"] = (
    df["Last digit"].shift(1)
)

df["next_digit"] = (
    df["Last digit"].shift(-1)
)


# ============================================================
# REMOVE FEATURE NA VALUES
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
# ORIGINAL H3
# ============================================================

df["H3"] = (
    (df["delta"] <= DELTA_THRESHOLD)
    &
    (df["volatility"] <= VOLATILITY_THRESHOLD)
)


# ============================================================
# FROZEN CONDITIONAL HYPOTHESIS
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
# EXTRACT SIGNALS
# ============================================================

signals = df[
    df["H3_CONDITIONAL"]
].copy()


print()

print(
    f"Usable rows: {len(df)}"
)

print(
    f"Conditional H3 observations: "
    f"{len(signals)}"
)


# ============================================================
# CURRENT PERFORMANCE
# ============================================================

if len(signals) > 0:

    correct = (
        signals["next_digit"]
        == TARGET_DIGIT
    ).sum()

    observations = len(signals)

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

else:

    correct = 0
    observations = 0
    probability = 0
    difference = 0
    p_value = 1


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)

print("CURRENT CONDITIONAL PERFORMANCE")

print("=" * 70)

print()

print(
    f"Observations: {observations}"
)

print(
    f"Digit {TARGET_DIGIT} occurrences: "
    f"{correct}"
)

print(
    f"Observed probability: "
    f"{probability:.2%}"
)

print(
    f"Baseline: "
    f"{BASELINE:.2%}"
)

print(
    f"Difference: "
    f"{difference:+.2%}"
)

print(
    f"P-value: "
    f"{p_value:.6f}"
)


# ============================================================
# PROGRESS TOWARD OOS SAMPLE
# ============================================================

remaining = (
    TARGET_CONDITIONAL_OBSERVATIONS
    - observations
)

if remaining < 0:
    remaining = 0


progress = (
    observations
    / TARGET_CONDITIONAL_OBSERVATIONS
) * 100


print()
print("=" * 70)

print("DATA ACCUMULATION PROGRESS")

print("=" * 70)

print()

print(
    f"Current observations: "
    f"{observations}"
)

print(
    f"Target observations: "
    f"{TARGET_CONDITIONAL_OBSERVATIONS}"
)

print(
    f"Remaining required: "
    f"{remaining}"
)

print(
    f"Progress: "
    f"{min(progress, 100):.2f}%"
)


# ============================================================
# CHECKPOINTS
# ============================================================

print()

print("CHECKPOINTS")

print()

checkpoints = [
    500,
    1000,
    1500,
    2000,
    2500,
    3000
]

for checkpoint in checkpoints:

    if observations >= checkpoint:

        checkpoint_data = signals.iloc[
            :checkpoint
        ]

        checkpoint_correct = (
            checkpoint_data["next_digit"]
            == TARGET_DIGIT
        ).sum()

        checkpoint_probability = (
            checkpoint_correct
            / checkpoint
        )

        checkpoint_test = binomtest(
            checkpoint_correct,
            checkpoint,
            BASELINE,
            alternative="greater"
        )

        print(
            f"{checkpoint:4d} observations | "
            f"Correct: "
            f"{checkpoint_correct:3d} | "
            f"Probability: "
            f"{checkpoint_probability:6.2%} | "
            f"p-value: "
            f"{checkpoint_test.pvalue:.6f}"
        )

    else:

        print(
            f"{checkpoint:4d} observations | "
            f"NOT YET REACHED"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

print("STATUS")

print("=" * 70)

print()

if observations < TARGET_CONDITIONAL_OBSERVATIONS:

    print(
        "NOT READY FOR OOS TEST."
    )

    print()

    print(
        f"We need approximately "
        f"{remaining} more H3 + previous-digit-4 "
        f"observations."
    )

    print()

    print(
        "Keep collecting forward data."
    )

elif observations >= TARGET_CONDITIONAL_OBSERVATIONS:

    print(
        "ENOUGH CONDITIONAL DATA."
    )

    print()

    print(
        "The frozen H3 + previous digit 4 "
        "condition now has enough observations "
        "to perform the planned OOS test."
    )


# ============================================================
# IMPORTANT WARNING
# ============================================================

print()

print("=" * 70)

print("IMPORTANT")

print("=" * 70)

print()

print(
    "This test does NOT change H3."
)

print(
    "The previous digit 4 condition remains frozen."
)

print(
    "Do not select another previous digit based "
    "on these results."
)

print(
    "Do not change the delta or volatility thresholds."
)

print(
    "The next formal step, once enough observations "
    "are available, is the OOS test."
)

print()

print(
    "Conditional accumulation test complete."
)