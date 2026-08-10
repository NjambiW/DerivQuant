import sys
import os
import pandas as pd
from scipy.stats import binomtest


# ============================================================
# V4 H3 OUT-OF-SAMPLE TEST
# ============================================================

CURRENT_FILE = os.path.abspath(__file__)

OOS_DIR = os.path.dirname(CURRENT_FILE)
FORWARD_DIR = os.path.dirname(OOS_DIR)
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
BASELINE = 0.10

LARGE_NEGATIVE_DELTA = -0.10
LOW_VOLATILITY = 0.145358

# Number of H3 observations used to establish
# the historical training baseline.
TRAINING_SIZE = 2500

# Number of H3 observations evaluated out-of-sample
# in each test block.
TEST_BLOCK_SIZE = 500


# ============================================================
# FILE PATHS
# ============================================================

FORWARD_FILE = os.path.join(
    FORWARD_DIR,
    "r100_forward.csv"
)

RESULTS_DIR = os.path.join(
    OOS_DIR,
    "results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "h3_oos_results.csv"
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

    # Sort chronologically
    df = df.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    # Numeric conversion
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
        .rolling(10)
        .std()
    )

    # --------------------------------------------------------
    # NEXT DIGIT
    # --------------------------------------------------------

    df["next_digit"] = (
        df["Last digit"].shift(-1)
    )

    # Remove unusable rows
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
# GET H3 OBSERVATIONS
# ============================================================

def get_h3_rows(df):

    h3 = df[
        (df["price_delta"] <= LARGE_NEGATIVE_DELTA)
        &
        (df["step_volatility"] <= LOW_VOLATILITY)
    ].copy()

    return h3.reset_index(
        drop=True
    )


# ============================================================
# OUT-OF-SAMPLE TEST
# ============================================================

def run_oos_test(h3):

    print()
    print("============================================================")
    print("V4 H3 OUT-OF-SAMPLE TEST")
    print("============================================================")

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

    print(
        f"Training H3 observations: "
        f"{TRAINING_SIZE}"
    )

    print(
        f"OOS test block size: "
        f"{TEST_BLOCK_SIZE}"
    )

    print()

    total_h3 = len(h3)

    print(
        f"Total H3 observations: "
        f"{total_h3}"
    )


    # ========================================================
    # CHECK DATA
    # ========================================================

    minimum_required = (
        TRAINING_SIZE +
        TEST_BLOCK_SIZE
    )

    if total_h3 < minimum_required:

        print()
        print(
            "ERROR: Not enough H3 observations."
        )

        print(
            f"Need at least {minimum_required}."
        )

        return


    # ========================================================
    # CREATE RESULTS DIRECTORY
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


    results = []

    position = TRAINING_SIZE

    block_number = 0


    # ========================================================
    # WALK FORWARD THROUGH TIME
    # ========================================================

    while (
        position + TEST_BLOCK_SIZE
        <= total_h3
    ):

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        training = h3.iloc[
            :position
        ]

        # ----------------------------------------------------
        # OUT-OF-SAMPLE DATA
        # ----------------------------------------------------

        test = h3.iloc[
            position:
            position + TEST_BLOCK_SIZE
        ]

        block_number += 1


        # ====================================================
        # TRAINING STATISTICS
        # ====================================================

        training_correct = int(
            (
                training["next_digit"]
                == TARGET_DIGIT
            ).sum()
        )

        training_probability = (
            training_correct /
            len(training)
        )


        # ====================================================
        # OOS PERFORMANCE
        # ====================================================

        test_correct = int(
            (
                test["next_digit"]
                == TARGET_DIGIT
            ).sum()
        )

        test_observations = len(test)

        test_probability = (
            test_correct /
            test_observations
        )

        difference = (
            test_probability -
            BASELINE
        )


        # ====================================================
        # BINOMIAL TEST
        # ====================================================

        test_result = binomtest(
            test_correct,
            test_observations,
            BASELINE,
            alternative="greater"
        )

        p_value = test_result.pvalue


        # ====================================================
        # TIME
        # ====================================================

        start_time = pd.to_datetime(
            test["time"].iloc[0],
            unit="s",
            utc=True
        ).tz_convert(
            "Africa/Nairobi"
        )

        end_time = pd.to_datetime(
            test["time"].iloc[-1],
            unit="s",
            utc=True
        ).tz_convert(
            "Africa/Nairobi"
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        result = {

            "block": block_number,

            "training_observations": len(training),

            "training_probability":
                training_probability,

            "test_observations":
                test_observations,

            "correct":
                test_correct,

            "test_probability":
                test_probability,

            "difference":
                difference,

            "p_value":
                p_value,

            "start_time":
                start_time,

            "end_time":
                end_time
        }

        results.append(
            result
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        print(
            f"OOS Block {block_number:2d} | "
            f"Train: {len(training):5d} | "
            f"Test: {test_observations:4d} | "
            f"Correct: {test_correct:3d} | "
            f"Accuracy: {test_probability * 100:6.2f}% | "
            f"Difference: {difference * 100:+6.2f}% | "
            f"P-value: {p_value:.6f}"
        )


        # Move forward without overlapping
        position += TEST_BLOCK_SIZE


    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    probabilities = (
        results_df["test_probability"]
    )

    significant = (
        results_df["p_value"] < 0.05
    )

    above_baseline = (
        results_df["test_probability"]
        > BASELINE
    )


    print()
    print("============================================================")
    print("## OUT-OF-SAMPLE SUMMARY")
    print("============================================================")

    print()

    print(
        f"OOS blocks tested: "
        f"{len(results_df)}"
    )

    print(
        f"Blocks above baseline: "
        f"{int(above_baseline.sum())}/"
        f"{len(results_df)}"
    )

    print(
        f"Statistically significant blocks: "
        f"{int(significant.sum())}/"
        f"{len(results_df)}"
    )

    print(
        f"Mean OOS probability: "
        f"{probabilities.mean() * 100:.2f}%"
    )

    print(
        f"Minimum OOS probability: "
        f"{probabilities.min() * 100:.2f}%"
    )

    print(
        f"Maximum OOS probability: "
        f"{probabilities.max() * 100:.2f}%"
    )

    print(
        f"Median OOS probability: "
        f"{probabilities.median() * 100:.2f}%"
    )


    # ========================================================
    # TOTAL OOS PERFORMANCE
    # ========================================================

    total_correct = int(
        results_df["correct"].sum()
    )

    total_observations = int(
        results_df["test_observations"].sum()
    )

    total_probability = (
        total_correct /
        total_observations
    )

    total_difference = (
        total_probability -
        BASELINE
    )

    total_test = binomtest(
        total_correct,
        total_observations,
        BASELINE,
        alternative="greater"
    )

    total_p_value = (
        total_test.pvalue
    )


    print()
    print("============================================================")
    print("## COMBINED OOS PERFORMANCE")
    print("============================================================")

    print()

    print(
        f"Total OOS observations: "
        f"{total_observations}"
    )

    print(
        f"Total correct: "
        f"{total_correct}"
    )

    print(
        f"OOS probability: "
        f"{total_probability * 100:.2f}%"
    )

    print(
        f"Baseline: "
        f"{BASELINE * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{total_difference * 100:+.2f}%"
    )

    print(
        f"P-value: "
        f"{total_p_value:.6f}"
    )


    # ========================================================
    # CONCLUSION
    # ========================================================

    print()
    print("============================================================")
    print("## CONCLUSION")
    print("============================================================")

    print()

    if (
        total_probability > BASELINE
        and
        total_p_value < 0.05
    ):

        print(
            "H3 PASSES the combined out-of-sample test."
        )

        print(
            "The effect remains statistically above "
            "the 10% baseline on unseen observations."
        )

    elif (
        total_probability > BASELINE
    ):

        print(
            "H3 remains above baseline out-of-sample, "
            "but the evidence is not statistically strong."
        )

    else:

        print(
            "H3 does NOT outperform the 10% baseline "
            "on the combined out-of-sample observations."
        )


    # ========================================================
    # SAVE
    # ========================================================

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
        "Out-of-sample test complete."
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

    h3 = get_h3_rows(
        df
    )

    print(
        f"H3 observations found: "
        f"{len(h3)}"
    )

    run_oos_test(
        h3
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()