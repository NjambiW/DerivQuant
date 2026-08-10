import sys
import os
import pandas as pd
from scipy.stats import binomtest

# ============================================================
# V4 TIME STABILITY TEST
# ============================================================
CURRENT_FILE = os.path.abspath(__file__)

FORWARD_DIR = os.path.dirname(CURRENT_FILE)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

# ============================================================
# SETTINGS
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

TARGET_DIGIT = 3
BASELINE = 0.10

# H3 thresholds from our previous validation
LARGE_NEGATIVE_DELTA = -0.100000
LOW_VOLATILITY = 0.145358

# Time window size in hours
WINDOW_HOURS = 2


# ============================================================
# LOAD DATA
# ============================================================

print()
print("============================================================")
print("V4 H3 TIME-WINDOW STABILITY TEST")
print("============================================================")

print()
print("Hypothesis:")
print("Large Negative Delta + Low Volatility -> Next Digit 3")

print()
print(f"Large negative delta <= {LARGE_NEGATIVE_DELTA:.6f}")
print(f"Low volatility <= {LOW_VOLATILITY:.6f}")
print(f"Target digit: {TARGET_DIGIT}")
print(f"Baseline: {BASELINE:.2%}")
print(f"Time window: {WINDOW_HOURS} hours")

print()
print("Loading forward dataset...")
print()

if not os.path.isfile(FORWARD_FILE):

    print("ERROR:")
    print("Forward dataset not found.")

    print()
    print(f"Expected file:")
    print(FORWARD_FILE)

    sys.exit(1)


df = pd.read_csv(FORWARD_FILE)

print(f"Forward file: {FORWARD_FILE}")
print(f"Raw rows: {len(df)}")


# ============================================================
# PREPARE DATA
# ============================================================

df["time"] = pd.to_numeric(
    df["time"],
    errors="coerce"
)

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["Last digit"] = pd.to_numeric(
    df["Last digit"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "time",
        "price",
        "Last digit"
    ]
).copy()


df["Last digit"] = df["Last digit"].astype(int)


# ============================================================
# CREATE FEATURES
# ============================================================

df["Price Delta"] = (
    df["price"].diff()
)


df["Step Volatility 10"] = (
    df["Price Delta"]
    .rolling(10)
    .std()
)


# ============================================================
# CREATE NEXT DIGIT
# ============================================================

df["Next Digit"] = (
    df["Last digit"].shift(-1)
)


# Remove rows where features/target aren't available

df = df.dropna(
    subset=[
        "Price Delta",
        "Step Volatility 10",
        "Next Digit"
    ]
).copy()


df["Next Digit"] = (
    df["Next Digit"].astype(int)
)


# ============================================================
# CREATE H3 CONDITION
# ============================================================

df["H3"] = (
    (df["Price Delta"] <= LARGE_NEGATIVE_DELTA)
    &
    (df["Step Volatility 10"] <= LOW_VOLATILITY)
)


# ============================================================
# CREATE DATETIME
# ============================================================

df["datetime"] = pd.to_datetime(
    df["time"],
    unit="s",
    utc=True
).dt.tz_convert(
    "Africa/Nairobi"
)


# ============================================================
# DETERMINE TIME RANGE
# ============================================================

start_time = df["datetime"].min()
end_time = df["datetime"].max()

print()
print(f"Usable rows: {len(df)}")
print(f"Start time: {start_time}")
print(f"End time:   {end_time}")


# ============================================================
# CREATE TIME WINDOWS
# ============================================================

window_seconds = WINDOW_HOURS * 60 * 60

df["window_number"] = (
    (
        df["time"] - df["time"].min()
    )
    // window_seconds
)


# ============================================================
# ANALYZE EACH WINDOW
# ============================================================

print()
print("## TIME-WINDOW RESULTS")
print()
print(
    "Window | Start | End | H3 Rows | Digit 3 | "
    "Probability | Difference | P-value"
)
print("-" * 110)


results = []


for window_number, group in df.groupby(
    "window_number"
):

    h3 = group[
        group["H3"]
    ]

    observations = len(h3)

    if observations == 0:
        continue


    digit_3_count = (
        h3["Next Digit"] == TARGET_DIGIT
    ).sum()


    probability = (
        digit_3_count / observations
    )


    difference = (
        probability - BASELINE
    )


    p_value = binomtest(
        digit_3_count,
        observations,
        BASELINE,
        alternative="greater"
    ).pvalue


    window_start = (
        group["datetime"].min()
    )

    window_end = (
        group["datetime"].max()
    )


    results.append(
        {
            "window": int(window_number),
            "start": window_start,
            "end": window_end,
            "observations": observations,
            "digit_3": digit_3_count,
            "probability": probability,
            "difference": difference,
            "p_value": p_value
        }
    )


    print(
        f"{int(window_number):>6} | "
        f"{window_start.strftime('%m-%d %H:%M'):>14} | "
        f"{window_end.strftime('%m-%d %H:%M'):>14} | "
        f"{observations:>7} | "
        f"{digit_3_count:>7} | "
        f"{probability:>10.2%} | "
        f"{difference:>10.2%} | "
        f"{p_value:.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print()
print("============================================================")
print("TIME-WINDOW SUMMARY")
print("============================================================")


if results_df.empty:

    print()
    print("No H3 observations were found.")

    sys.exit(0)


above_baseline = (
    results_df["probability"] > BASELINE
).sum()


significant = (
    results_df["p_value"] < 0.05
).sum()


mean_probability = (
    results_df["probability"].mean()
)


minimum_probability = (
    results_df["probability"].min()
)


maximum_probability = (
    results_df["probability"].max()
)


print()
print(
    f"Windows tested: "
    f"{len(results_df)}"
)

print(
    f"Windows above 10%: "
    f"{above_baseline}/{len(results_df)}"
)

print(
    f"Statistically significant windows: "
    f"{significant}/{len(results_df)}"
)

print(
    f"Mean H3 probability: "
    f"{mean_probability:.2%}"
)

print(
    f"Minimum probability: "
    f"{minimum_probability:.2%}"
)

print(
    f"Maximum probability: "
    f"{maximum_probability:.2%}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("## CONCLUSION")
print()


if significant >= 2 and above_baseline >= len(results_df) / 2:

    print(
        "H3 shows meaningful time stability."
    )

    print(
        "The effect appears repeatedly across "
        "multiple time windows."
    )

elif above_baseline >= len(results_df) / 2:

    print(
        "H3 shows partial time stability."
    )

    print(
        "The effect appears above baseline in "
        "multiple windows, but statistical evidence "
        "is not consistently strong."
    )

else:

    print(
        "H3 does NOT show strong time stability."
    )

    print(
        "The observed effect appears concentrated "
        "in a limited number of time windows."
    )


print()
print(
    "Time-window stability test complete."
)
print()

