import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# H3 INDEPENDENCE / CLUSTERING TEST
# ============================================================

# ------------------------------------------------------------
# FIND PROJECT PATHS
# ------------------------------------------------------------

CURRENT_FILE = os.path.abspath(__file__)

INDEPENDENCE_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(INDEPENDENCE_DIR)

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
BASELINE = 0.10

LARGE_NEGATIVE_DELTA = -0.100000
LOW_VOLATILITY = 0.145358

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
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
    print(FORWARD_FILE)
    sys.exit(1)


df = pd.read_csv(FORWARD_FILE)

print(
    f"Raw rows: {len(df)}"
)


# ============================================================
# CLEAN DATA
# ============================================================

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["time"] = pd.to_numeric(
    df["time"],
    errors="coerce"
)

df = df.dropna(
    subset=["price", "time"]
).copy()

df = df.sort_values(
    "time"
).reset_index(
    drop=True
)


# ============================================================
# CALCULATE FEATURES
# ============================================================

df["delta"] = df["price"].diff()

df["volatility"] = (
    df["delta"]
    .rolling(
        window=10
    )
    .std()
)


# ============================================================
# NEXT DIGIT
# ============================================================

df["next_digit"] = (
    df["price"]
    .shift(-1)
    .apply(
        lambda x:
        int(f"{x:.2f}"[-1])
        if pd.notna(x)
        else None
    )
)


# ============================================================
# CREATE H3 CONDITION
# ============================================================

df["H3"] = (
    (df["delta"] <= LARGE_NEGATIVE_DELTA)
    &
    (df["volatility"] <= LOW_VOLATILITY)
)


df = df.dropna(
    subset=[
        "delta",
        "volatility",
        "next_digit"
    ]
).copy()


print(
    f"Usable rows: {len(df)}"
)


# ============================================================
# EXTRACT H3 SIGNALS
# ============================================================

h3 = df[df["H3"]].copy()

print(
    f"H3 observations found: {len(h3)}"
)


# ============================================================
# IDENTIFY H3 RUNS
# ============================================================

# Keep original dataframe position
# so we can determine whether H3
# observations are consecutive.

h3_positions = h3.index.to_list()

h3["position"] = h3_positions

h3["previous_h3_position"] = (
    h3["position"].shift(1)
)

h3["consecutive"] = (
    h3["position"]
    ==
    h3["previous_h3_position"] + 1
)


# ============================================================
# IDENTIFY RUN NUMBER
# ============================================================

h3["new_run"] = (
    ~h3["consecutive"]
)

h3["run_id"] = (
    h3["new_run"]
    .cumsum()
)


# ============================================================
# RUN LENGTHS
# ============================================================

run_lengths = (
    h3
    .groupby("run_id")
    .size()
)


# ============================================================
# CLASSIFY SIGNALS BY POSITION IN RUN
# ============================================================

run_sizes = (
    run_lengths
    .rename("run_size")
)

h3 = h3.merge(
    run_sizes,
    left_on="run_id",
    right_index=True
)


# ------------------------------------------------------------
# SIGNAL TYPES
# ------------------------------------------------------------

h3["signal_type"] = "Isolated"

h3.loc[
    h3["run_size"] > 1,
    "signal_type"
] = "Clustered"


# ============================================================
# POSITION INSIDE RUN
# ============================================================

h3["run_position"] = (
    h3
    .groupby("run_id")
    .cumcount()
    + 1
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("H3 INDEPENDENCE / CLUSTERING TEST")
print("=" * 60)

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
    f"{LARGE_NEGATIVE_DELTA:.6f}"
)

print(
    f"Low volatility <= "
    f"{LOW_VOLATILITY:.6f}"
)

print(
    f"Target digit: {TARGET_DIGIT}"
)

print(
    f"Baseline: {BASELINE:.2%}"
)


# ============================================================
# RUN SUMMARY
# ============================================================

print()
print("=" * 60)
print("H3 RUN STRUCTURE")
print("=" * 60)

print()

print(
    f"Total H3 signals: {len(h3)}"
)

print(
    f"Total H3 runs: {len(run_lengths)}"
)

print(
    f"Average run length: "
    f"{run_lengths.mean():.2f}"
)

print(
    f"Median run length: "
    f"{run_lengths.median():.2f}"
)

print(
    f"Maximum run length: "
    f"{run_lengths.max()}"
)


# ============================================================
# ISOLATED VS CLUSTERED
# ============================================================

isolated = h3[
    h3["run_size"] == 1
].copy()

clustered = h3[
    h3["run_size"] > 1
].copy()


print()
print("=" * 60)
print("ISOLATED VS CLUSTERED SIGNALS")
print("=" * 60)

print()

print(
    f"Isolated H3 signals: "
    f"{len(isolated)}"
)

print(
    f"Clustered H3 signals: "
    f"{len(clustered)}"
)


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_performance(
    data,
    label
):

    observations = len(data)

    if observations == 0:

        print()
        print(
            f"{label}: No observations."
        )

        return None


    correct = int(
        (
            data["next_digit"]
            == TARGET_DIGIT
        ).sum()
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
        alternative="two-sided"
    )


    p_value = result.pvalue


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
        "group": label,
        "observations": observations,
        "correct": correct,
        "probability": probability,
        "difference": difference,
        "p_value": p_value
    }


# ============================================================
# TEST ISOLATED SIGNALS
# ============================================================

isolated_result = calculate_performance(
    isolated,
    "ISOLATED H3 SIGNALS"
)


# ============================================================
# TEST CLUSTERED SIGNALS
# ============================================================

clustered_result = calculate_performance(
    clustered,
    "CLUSTERED H3 SIGNALS"
)


# ============================================================
# TEST FIRST SIGNAL OF EACH RUN
# ============================================================

first_of_run = h3[
    h3["run_position"] == 1
].copy()


first_result = calculate_performance(
    first_of_run,
    "FIRST SIGNAL OF EACH H3 RUN"
)


# ============================================================
# TEST SECOND+ SIGNALS
# ============================================================

continuation_signals = h3[
    h3["run_position"] > 1
].copy()


continuation_result = calculate_performance(
    continuation_signals,
    "CONTINUATION SIGNALS"
)


# ============================================================
# PERFORMANCE BY RUN POSITION
# ============================================================

print()
print("=" * 60)
print("PERFORMANCE BY RUN POSITION")
print("=" * 60)

print()

print(
    "Position | Observations | Digit 3 | "
    "Probability | Difference | P-value"
)

print("-" * 75)


position_results = []


for position in sorted(
    h3["run_position"].unique()
):

    group = h3[
        h3["run_position"]
        == position
    ]


    observations = len(group)

    correct = int(
        (
            group["next_digit"]
            == TARGET_DIGIT
        ).sum()
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
        alternative="two-sided"
    ).pvalue


    print(
        f"{position:^8} | "
        f"{observations:^12} | "
        f"{correct:^7} | "
        f"{probability:>10.2%} | "
        f"{difference:>+10.2%} | "
        f"{p_value:.6f}"
    )


    position_results.append(
        {
            "run_position": position,
            "observations": observations,
            "correct": correct,
            "probability": probability,
            "difference": difference,
            "p_value": p_value
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    INDEPENDENCE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


results = []


for result in [
    isolated_result,
    clustered_result,
    first_result,
    continuation_result
]:

    if result is not None:

        results.append(result)


results_df = pd.DataFrame(
    results
)


RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "h3_independence_results.csv"
)


results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# CONCLUSION
# ============================================================

print()
print("=" * 60)
print("INDEPENDENCE TEST CONCLUSION")
print("=" * 60)

print()


if (
    isolated_result is not None
    and clustered_result is not None
):

    isolated_prob = (
        isolated_result["probability"]
    )

    clustered_prob = (
        clustered_result["probability"]
    )


    print(
        f"Isolated H3 probability: "
        f"{isolated_prob:.2%}"
    )

    print(
        f"Clustered H3 probability: "
        f"{clustered_prob:.2%}"
    )

    print()


    if (
        isolated_prob > BASELINE
        and isolated_result["p_value"] < 0.05
    ):

        print(
            "H3 remains statistically "
            "significant among isolated signals."
        )

        print(
            "This supports the possibility "
            "that the effect is not solely "
            "caused by clustering."
        )

    elif isolated_prob > BASELINE:

        print(
            "H3 remains above baseline "
            "among isolated signals, "
            "but evidence is not statistically strong."
        )

    else:

        print(
            "H3 does not remain above baseline "
            "among isolated signals."
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
    "H3 independence test complete."
)