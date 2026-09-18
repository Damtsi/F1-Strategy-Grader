import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "2025_counterfactual_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "2025_strategy_grades.csv"
)


# ============================================================
# STRATEGY GRADING
# ============================================================

def classify_strategy(difference):

    if difference >= 10:
        return "EXCELLENT_PIT"

    elif difference >= 5:
        return "VERY_GOOD_PIT"

    elif difference >= 2:
        return "GOOD_PIT"

    elif difference > -2:
        return "NEUTRAL"

    elif difference > -5:
        return "SLIGHTLY_BETTER_STAY_OUT"

    elif difference > -10:
        return "POOR_PIT"

    else:
        return "VERY_POOR_PIT"


# ============================================================
# STRATEGY SCORE
# ============================================================

def calculate_score(difference):

    if difference >= 10:
        return 100

    elif difference >= 5:
        return 85

    elif difference >= 2:
        return 70

    elif difference > -2:
        return 50

    elif difference > -5:
        return 35

    elif difference > -10:
        return 20

    else:
        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY GRADER")
    print("STRATEGY GRADING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD COUNTERFACTUAL RESULTS
    # --------------------------------------------------------

    print("\nLoading counterfactual results...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Counterfactual records loaded: {len(df)}"
    )

    if df.empty:

        raise RuntimeError(
            "Counterfactual results file is empty."
        )

    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "counterfactual_difference",
        "counterfactual_confidence"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # CLEAN DIFFERENCE COLUMN
    # --------------------------------------------------------

    df["counterfactual_difference"] = pd.to_numeric(
        df["counterfactual_difference"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["counterfactual_difference"]
    ).copy()

    if df.empty:

        raise RuntimeError(
            "No valid counterfactual differences found."
        )

    # --------------------------------------------------------
    # STRATEGY CLASSIFICATION
    # --------------------------------------------------------

    print("\nClassifying strategies...")

    df["strategy_assessment"] = (
        df["counterfactual_difference"]
        .apply(classify_strategy)
    )

    # --------------------------------------------------------
    # STRATEGY SCORE
    # --------------------------------------------------------

    df["strategy_score"] = (
        df["counterfactual_difference"]
        .apply(calculate_score)
    )

    # --------------------------------------------------------
    # CONFIDENCE ADJUSTMENT
    # --------------------------------------------------------

    print("\nApplying confidence adjustment...")

    confidence_weights = {
        "HIGH": 1.00,
        "MEDIUM": 0.75
    }

    df["confidence_weight"] = (
        df["counterfactual_confidence"]
        .map(confidence_weights)
        .fillna(0.50)
    )

    df["confidence_adjusted_score"] = (
        df["strategy_score"]
        * df["confidence_weight"]
    )

    print("\nConfidence-adjusted score statistics:")

    print(
        df["confidence_adjusted_score"]
        .describe()
    )

    # --------------------------------------------------------
    # ADVANTAGE PER LAP
    # --------------------------------------------------------

    if "laps_compared" in df.columns:

        df["counterfactual_difference_per_lap"] = (
            df["counterfactual_difference"]
            / df["laps_compared"]
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRATEGY GRADING VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # STRATEGY DISTRIBUTION
    # --------------------------------------------------------

    print("\nStrategy distribution:")

    print(
        df["strategy_assessment"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # RAW SCORE STATISTICS
    # --------------------------------------------------------

    print("\nRaw strategy score statistics:")

    print(
        df["strategy_score"]
        .describe()
    )

    # --------------------------------------------------------
    # CONFIDENCE-ADJUSTED SCORE STATISTICS
    # --------------------------------------------------------

    print("\nConfidence-adjusted score statistics:")

    print(
        df["confidence_adjusted_score"]
        .describe()
    )

    # --------------------------------------------------------
    # COUNTERFACTUAL DECISION SUMMARY
    # --------------------------------------------------------

    print("\nCounterfactual decision summary:")

    pit_faster = (
        df["counterfactual_difference"] > 0
    ).sum()

    stay_out_faster = (
        df["counterfactual_difference"] < 0
    ).sum()

    approximately_equal = (
        df["counterfactual_difference"] == 0
    ).sum()

    print(
        f"Pit estimated faster       : {pit_faster}"
    )

    print(
        f"Stay-out estimated faster  : {stay_out_faster}"
    )

    print(
        f"Approximately equal        : {approximately_equal}"
    )

    # --------------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # --------------------------------------------------------

    print("\nConfidence distribution:")

    print(
        df["counterfactual_confidence"]
        .value_counts()
    )

    # --------------------------------------------------------
    # SAMPLE RESULTS
    # --------------------------------------------------------

    print("\nSample graded strategies:")

    sample_columns = [
        "race_id",
        "driver_id",
        "pit_lap",
        "tyre_before",
        "tyre_age_before",
        "counterfactual_difference",
        "counterfactual_difference_per_lap",
        "counterfactual_confidence",
        "confidence_weight",
        "strategy_assessment",
        "strategy_score",
        "confidence_adjusted_score"
    ]

    sample_columns = [
        column
        for column in sample_columns
        if column in df.columns
    ]

    print(
        df[sample_columns]
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")

    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("STRATEGY GRADING COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()