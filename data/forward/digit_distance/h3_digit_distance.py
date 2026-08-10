import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# H3 DIGIT DISTANCE TEST
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

DIGIT_DISTANCE_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(DIGIT_DISTANCE_DIR)
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

# Original H3 conditions
DELTA_THRESHOLD = -0.100000
VOLATILITY_THRESHOLD = 0.145358

# Proposed "slingshot" digits
FAR_DIGITS = [8, 9]

# ============================================================
# FILE PATHS
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    DIGIT_DISTANCE_DIR,
    "results",
    "h3"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "h3_digit_distance_results.csv"
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
    print()

    sys.exit()

df = pd.read_csv(FORWARD_FILE)

print(
    f"Raw rows: {len(df)}"
)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "time",
    "price",
    "Last digit"
]

for column in required_columns:

    if column not in df.columns:

        print()
        print(
            f"ERROR: Missing column: {column}"
        )

        sys.exit()

# ============================================================
# CLEAN DATA
# ============================================================

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
        "time",
        "price",
        "Last digit"
    ]
).copy()

df = df.sort_values(
    "time"
).reset_index(drop=True)

# ============================================================
# CREATE FEATURES
# ============================================================

# Price movement from previous tick
df["delta"] = df["price"].diff()

# Short-term volatility
df["volatility"] = (
    df["delta"]
    .rolling(10)
    .std()
)

# Previous tick's digit
df["previous_digit"] = (
    df["Last digit"]
    .shift(1)
)

# Digit that occurs AFTER the current signal
df["next_digit"] = (
    df["Last digit"]
    .shift(-1)
)

# Remove rows where features cannot be calculated
df = df.dropna(
    subset=[
        "delta",
        "volatility",
        "previous_digit",
        "next_digit"
    ]
).copy()

df["previous_digit"] = (
    df["previous_digit"]
    .astype(int)
)

df["next_digit"] = (
    df["next_digit"]
    .astype(int)
)

df["Last digit"] = (
    df["Last digit"]
    .astype(int)
)

# ============================================================
# DIGIT DISTANCE
# ============================================================

df["digit_distance"] = (
    df["previous_digit"]
    - TARGET_DIGIT
).abs()

# ============================================================
# BASE H3 CONDITION
# ============================================================

df["H3"] = (
    (df["delta"] <= DELTA_THRESHOLD)
    &
    (df["volatility"] <= VOLATILITY_THRESHOLD)
)

# ============================================================
# DISPLAY HYPOTHESIS
# ============================================================

print()
print("=" * 70)
print("H3 DIGIT DISTANCE TEST")
print("=" * 70)

print()

print(
    "Original H3:"
)

print(
    f"Large negative delta <= "
    f"{DELTA_THRESHOLD:.6f}"
)

print(
    f"Low volatility <= "
    f"{VOLATILITY_THRESHOLD:.6f}"
)

print()

print(
    "New condition:"
)

print(
    f"Previous digit is far from target "
    f"{TARGET_DIGIT}"
)

print(
    f"Proposed far digits: {FAR_DIGITS}"
)

print()

print(
    f"Target digit: {TARGET_DIGIT}"
)

print(
    f"Baseline: {BASELINE:.2%}"
)

# ============================================================
# GET H3 SIGNALS
# ============================================================

h3 = df[df["H3"]].copy()

print()
print(
    f"Total H3 signals: {len(h3)}"
)

if h3.empty:

    print()
    print("No H3 signals found.")
    sys.exit()

# ============================================================
# DIGIT DISTANCE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("## H3 DIGIT DISTANCE DISTRIBUTION")
print("=" * 70)

print()

distance_counts = (
    h3["digit_distance"]
    .value_counts()
    .sort_index()
)

for distance in range(0, 7):

    count = int(
        distance_counts.get(
            distance,
            0
        )
    )

    percentage = (
        count / len(h3)
        if len(h3) > 0
        else 0
    )

    print(
        f"Distance {distance}: "
        f"{count:5d} | "
        f"{percentage:.2%}"
    )

# ============================================================
# TEST FUNCTION
# ============================================================

def test_subset(name, subset):

    observations = len(subset)

    if observations == 0:

        print()
        print(
            f"{name}: NO OBSERVATIONS"
        )

        return None

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

    print()
    print(
        f"## {name}"
    )

    print()

    print(
        f"Observations: {observations}"
    )

    print(
        f"Digit {TARGET_DIGIT}: {correct}"
    )

    print(
        f"Probability: {probability:.2%}"
    )

    print(
        f"Difference: {difference:+.2%}"
    )

    print(
        f"P-value: {p_value:.6f}"
    )

    if p_value < 0.05:

        print(
            "Result: STATISTICALLY SIGNIFICANT"
        )

    else:

        print(
            "Result: NOT STATISTICALLY SIGNIFICANT"
        )

    return {
        "test": name,
        "observations": observations,
        "correct": correct,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }

# ============================================================
# ORIGINAL H3
# ============================================================

results = []

result = test_subset(
    "ORIGINAL H3",
    h3
)

if result:
    results.append(result)

# ============================================================
# DISTANCE GROUPS
# ============================================================

distance_0_1 = h3[
    h3["digit_distance"] <= 1
]

distance_2_3 = h3[
    h3["digit_distance"].between(2, 3)
]

distance_4_6 = h3[
    h3["digit_distance"] >= 4
]

result = test_subset(
    "DISTANCE 0-1",
    distance_0_1
)

if result:
    results.append(result)

result = test_subset(
    "DISTANCE 2-3",
    distance_2_3
)

if result:
    results.append(result)

result = test_subset(
    "DISTANCE 4-6",
    distance_4_6
)

if result:
    results.append(result)

# ============================================================
# SPECIFIC .8 / .9 TEST
# ============================================================

far_digits = h3[
    h3["previous_digit"].isin(
        FAR_DIGITS
    )
]

result = test_subset(
    "PREVIOUS DIGIT 8 OR 9",
    far_digits
)

if result:
    results.append(result)

# ============================================================
# EACH PREVIOUS DIGIT
# ============================================================

print()
print("=" * 70)
print("## H3 BY PREVIOUS DIGIT")
print("=" * 70)

print()

print(
    "Previous | Observations | Digit 3 | "
    "Probability | Difference | P-value"
)

for digit in range(10):

    subset = h3[
        h3["previous_digit"]
        == digit
    ]

    observations = len(subset)

    if observations == 0:

        continue

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

    p_value = binomtest(
        correct,
        observations,
        BASELINE,
        alternative="greater"
    ).pvalue

    print(
        f"{digit:^8} | "
        f"{observations:>12} | "
        f"{correct:>7} | "
        f"{probability:>10.2%} | "
        f"{difference:>+10.2%} | "
        f"{p_value:.6f}"
    )

# ============================================================
# DISTANCE 4-6 BY INDIVIDUAL DISTANCE
# ============================================================

print()
print("=" * 70)
print("## TARGET DISTANCE PERFORMANCE")
print("=" * 70)

print()

print(
    "Distance | Observations | Digit 3 | "
    "Probability | Difference | P-value"
)

for distance in range(7):

    subset = h3[
        h3["digit_distance"]
        == distance
    ]

    observations = len(subset)

    if observations == 0:

        continue

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

    p_value = binomtest(
        correct,
        observations,
        BASELINE,
        alternative="greater"
    ).pvalue

    print(
        f"{distance:^8} | "
        f"{observations:>12} | "
        f"{correct:>7} | "
        f"{probability:>10.2%} | "
        f"{difference:>+10.2%} | "
        f"{p_value:.6f}"
    )

# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

print()
print("=" * 70)
print("## SUMMARY")
print("=" * 70)

print()

if not results_df.empty:

    above_baseline = (
        results_df["probability"]
        > BASELINE
    ).sum()

    significant = (
        results_df["p_value"]
        < 0.05
    ).sum()

    print(
        f"Tests performed: "
        f"{len(results_df)}"
    )

    print(
        f"Tests above baseline: "
        f"{above_baseline}/{len(results_df)}"
    )

    print(
        f"Statistically significant: "
        f"{significant}/{len(results_df)}"
    )

    best = results_df.loc[
        results_df["probability"].idxmax()
    ]

    print()

    print(
        f"Highest observed probability: "
        f"{best['probability']:.2%}"
    )

    print(
        f"Test: {best['test']}"
    )

    print(
        f"Observations: "
        f"{int(best['observations'])}"
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

far_result = results_df[
    results_df["test"]
    == "PREVIOUS DIGIT 8 OR 9"
]

if not far_result.empty:

    row = far_result.iloc[0]

    if row["p_value"] < 0.05:

        print(
            "The H3 + previous digit 8/9 "
            "condition shows statistically significant "
            "evidence above the 10% baseline."
        )

        print()
        print(
            "This supports further testing of the "
            "digit-distance hypothesis."
        )

    elif row["probability"] > BASELINE:

        print(
            "The H3 + previous digit 8/9 "
            "condition is above baseline, "
            "but the evidence is not statistically strong."
        )

    else:

        print(
            "The previous digit 8/9 condition "
            "does not improve H3 performance."
        )

print()
print(
    "IMPORTANT: This test does not establish "
    "a PRNG mechanism."
)

print(
    "It only tests whether the observed data "
    "supports the proposed digit-distance relationship."
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
    f"Results saved to:"
)

print(
    RESULTS_FILE
)

print()
print(
    "H3 digit distance test complete."
)