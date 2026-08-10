import sys
import os
import pandas as pd


# ============================================================
# V4 H3 FREQUENCY AND SIGNAL DISTRIBUTION TEST
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

FREQUENCY_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(FREQUENCY_DIR)
DATA_DIR = os.path.dirname(FORWARD_DIR)
PROJECT_ROOT = os.path.dirname(DATA_DIR)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_DIR)


# ============================================================
# SETTINGS
# ============================================================

TARGET_DIGIT = 3

LARGE_NEGATIVE_DELTA = -0.10
LOW_VOLATILITY = 0.145358

# Number of rows used to calculate volatility
VOLATILITY_WINDOW = 10


# ============================================================
# FILE PATHS
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    FREQUENCY_DIR,
    "results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "h3_frequency_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

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

    df = pd.read_csv(
        FORWARD_FILE
    )

    print(
        f"Raw rows: {len(df)}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    required_columns = [
        "time",
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
            f"ERROR: Missing columns: {missing}"
        )

        sys.exit(1)

    df = df.copy()

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    df = df.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRICE DELTA
    # --------------------------------------------------------

    df["price_delta"] = (
        df["price"].diff()
    )

    # --------------------------------------------------------
    # STEP VOLATILITY
    # --------------------------------------------------------

    df["step_volatility"] = (
        df["price_delta"]
        .rolling(
            VOLATILITY_WINDOW
        )
        .std()
    )

    # --------------------------------------------------------
    # NEXT DIGIT
    # --------------------------------------------------------

    df["next_digit"] = (
        df["Last digit"].shift(-1)
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "price_delta",
            "step_volatility",
            "next_digit"
        ]
    ).copy()

    df["next_digit"] = (
        df["next_digit"].astype(int)
    )

    return df


# ============================================================
# FIND H3 SIGNALS
# ============================================================

def find_h3_signals(df):

    h3 = df[
        (df["price_delta"] <= LARGE_NEGATIVE_DELTA)
        &
        (df["step_volatility"] <= LOW_VOLATILITY)
    ].copy()

    return h3


# ============================================================
# CALCULATE SIGNAL FREQUENCY
# ============================================================

def analyze_frequency(df, h3):

    total_rows = len(df)
    h3_rows = len(h3)

    frequency = (
        h3_rows /
        total_rows
    )

    print()
    print("============================================================")
    print("## H3 SIGNAL FREQUENCY")
    print("============================================================")

    print()

    print(
        f"Usable rows: {total_rows}"
    )

    print(
        f"H3 signals: {h3_rows}"
    )

    print(
        f"H3 frequency: {frequency * 100:.2f}%"
    )

    print(
        f"Approximately 1 signal every "
        f"{total_rows / h3_rows:.2f} rows"
    )


# ============================================================
# SIGNAL SPACING
# ============================================================

def analyze_spacing(df, h3):

    print()
    print("============================================================")
    print("## H3 SIGNAL SPACING")
    print("============================================================")

    print()

    if len(h3) < 2:

        print(
            "Not enough H3 signals to calculate spacing."
        )

        return None

    # Original row positions
    h3_positions = h3.index.to_series()

    gaps = h3_positions.diff().dropna()

    print(
        f"Average rows between H3 signals: "
        f"{gaps.mean():.2f}"
    )

    print(
        f"Minimum gap: "
        f"{gaps.min():.0f}"
    )

    print(
        f"Maximum gap: "
        f"{gaps.max():.0f}"
    )

    print(
        f"Median gap: "
        f"{gaps.median():.2f}"
    )

    return gaps


# ============================================================
# CONSECUTIVE SIGNALS
# ============================================================

def analyze_consecutive_signals(h3):

    print()
    print("============================================================")
    print("## CONSECUTIVE H3 SIGNALS")
    print("============================================================")

    print()

    if len(h3) < 2:

        print(
            "Not enough H3 signals."
        )

        return

    positions = h3.index.to_series()

    gaps = positions.diff()

    consecutive = (
        gaps == 1
    ).sum()

    print(
        f"Consecutive H3 transitions: "
        f"{int(consecutive)}"
    )

    print(
        f"Percentage of transitions consecutive: "
        f"{consecutive / (len(h3) - 1) * 100:.2f}%"
    )


# ============================================================
# H3 RUN LENGTHS
# ============================================================

def analyze_runs(h3):

    print()
    print("============================================================")
    print("## H3 RUN LENGTHS")
    print("============================================================")

    print()

    if len(h3) == 0:

        print(
            "No H3 signals found."
        )

        return

    positions = h3.index.to_series()

    runs = []

    current_run = 1

    previous_position = positions.iloc[0]

    for position in positions.iloc[1:]:

        if position == previous_position + 1:

            current_run += 1

        else:

            runs.append(
                current_run
            )

            current_run = 1

        previous_position = position

    runs.append(
        current_run
    )

    runs = pd.Series(
        runs
    )

    print(
        f"Number of H3 runs: "
        f"{len(runs)}"
    )

    print(
        f"Average run length: "
        f"{runs.mean():.2f}"
    )

    print(
        f"Longest H3 run: "
        f"{runs.max()}"
    )

    print(
        f"Median run length: "
        f"{runs.median():.2f}"
    )

    print()

    print(
        "Run length distribution:"
    )

    print()

    print(
        runs.value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# H3 DIGIT PERFORMANCE
# ============================================================

def analyze_digits(h3):

    print()
    print("============================================================")
    print("## H3 NEXT-DIGIT DISTRIBUTION")
    print("============================================================")

    print()

    counts = (
        h3["next_digit"]
        .value_counts()
        .reindex(
            range(10),
            fill_value=0
        )
    )

    total = len(h3)

    results = []

    for digit in range(10):

        count = int(
            counts[digit]
        )

        probability = (
            count /
            total
        )

        difference = (
            probability -
            0.10
        )

        results.append({

            "digit": digit,

            "count": count,

            "probability": probability,

            "difference": difference
        })

        print(
            f"Digit {digit}: "
            f"{count:5d} | "
            f"{probability * 100:6.2f}% | "
            f"{difference * 100:+6.2f}%"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# TARGET DIGIT
# ============================================================

def analyze_target(h3):

    print()
    print("============================================================")
    print("## TARGET DIGIT PERFORMANCE")
    print("============================================================")

    print()

    total = len(h3)

    correct = int(
        (
            h3["next_digit"]
            == TARGET_DIGIT
        ).sum()
    )

    probability = (
        correct /
        total
    )

    difference = (
        probability -
        0.10
    )

    print(
        f"Target digit: "
        f"{TARGET_DIGIT}"
    )

    print(
        f"H3 observations: "
        f"{total}"
    )

    print(
        f"Digit {TARGET_DIGIT} occurrences: "
        f"{correct}"
    )

    print(
        f"Observed probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Baseline: "
        f"10.00%"
    )

    print(
        f"Difference: "
        f"{difference * 100:+.2f}%"
    )


# ============================================================
# TIME DISTRIBUTION
# ============================================================

def analyze_time_distribution(h3):

    print()
    print("============================================================")
    print("## H3 SIGNALS BY HOUR")
    print("============================================================")

    print()

    h3 = h3.copy()

    h3["datetime"] = pd.to_datetime(
        h3["time"],
        unit="s",
        utc=True
    ).dt.tz_convert(
        "Africa/Nairobi"
    )

    h3["hour"] = (
        h3["datetime"].dt.hour
    )

    hourly = (
        h3.groupby("hour")
        .agg(
            observations=("next_digit", "size"),
            digit_3=(
                "next_digit",
                lambda x:
                (x == TARGET_DIGIT).sum()
            )
        )
        .reset_index()
    )

    hourly["probability"] = (
        hourly["digit_3"] /
        hourly["observations"]
    )

    hourly["difference"] = (
        hourly["probability"] -
        0.10
    )

    print(
        "Hour | H3 Signals | Digit 3 | Probability | Difference"
    )

    print()

    for _, row in hourly.iterrows():

        print(
            f"{int(row['hour']):02d}:00 | "
            f"{int(row['observations']):10d} | "
            f"{int(row['digit_3']):7d} | "
            f"{row['probability'] * 100:10.2f}% | "
            f"{row['difference'] * 100:+9.2f}%"
        )

    return hourly


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    h3,
    digit_results
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    output = h3.copy()

    output["h3_signal"] = 1

    output["target_digit"] = TARGET_DIGIT

    output["correct_prediction"] = (
        output["next_digit"]
        == TARGET_DIGIT
    ).astype(int)

    output.to_csv(
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


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_data(
        df
    )

    print(
        f"Usable rows: {len(df)}"
    )

    h3 = find_h3_signals(
        df
    )

    print(
        f"H3 observations found: "
        f"{len(h3)}"
    )

    if len(h3) == 0:

        print()
        print(
            "No H3 signals found."
        )

        return

    # --------------------------------------------------------
    # ANALYSES
    # --------------------------------------------------------

    analyze_frequency(
        df,
        h3
    )

    analyze_spacing(
        df,
        h3
    )

    analyze_consecutive_signals(
        h3
    )

    analyze_runs(
        h3
    )

    digit_results = analyze_digits(
        h3
    )

    analyze_target(
        h3
    )

    analyze_time_distribution(
        h3
    )

    save_results(
        h3,
        digit_results
    )

    print()
    print("============================================================")
    print("H3 FREQUENCY TEST COMPLETE")
    print("============================================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()