from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / f"{YEAR}_strategy_grades.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

DRIVER_OUTPUT = (
    OUTPUT_DIR
    / f"{YEAR}_driver_strategy_summary.csv"
)

RACE_OUTPUT = (
    OUTPUT_DIR
    / f"{YEAR}_race_strategy_summary.csv"
)

TYRE_OUTPUT = (
    OUTPUT_DIR
    / f"{YEAR}_tyre_strategy_summary.csv"
)

OVERALL_OUTPUT = (
    OUTPUT_DIR
    / f"{YEAR}_overall_strategy_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading strategy grades...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Strategy grades file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Strategy records loaded: {len(df):,}"
    )

    if df.empty:

        raise RuntimeError(
            "Strategy grades file is empty."
        )

    required_columns = [
        "race_id",
        "race_round",
        "driver_id",
        "pit_lap",
        "tyre_before",
        "counterfactual_difference",
        "strategy_assessment",
        "strategy_score",
        "counterfactual_confidence",
        "confidence_adjusted_score",
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

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    numeric_columns = [
        "race_id",
        "race_round",
        "driver_id",
        "pit_lap",
        "tyre_age_before",
        "counterfactual_difference",
        "strategy_score",
        "confidence_weight",
        "confidence_adjusted_score",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "race_id",
            "driver_id",
            "counterfactual_difference",
            "strategy_score",
            "confidence_adjusted_score",
        ]
    ).copy()

    return df


# ============================================================
# DRIVER ANALYSIS
# ============================================================

def analyze_drivers(df):

    print("\n" + "=" * 70)
    print("DRIVER STRATEGY ANALYSIS")
    print("=" * 70)

    summary = (
        df.groupby("driver_id")
        .agg(
            evaluated_pit_stops=(
                "counterfactual_difference",
                "count"
            ),

            average_strategy_score=(
                "strategy_score",
                "mean"
            ),

            average_confidence_adjusted_score=(
                "confidence_adjusted_score",
                "mean"
            ),

            average_counterfactual_difference=(
                "counterfactual_difference",
                "mean"
            ),

            median_counterfactual_difference=(
                "counterfactual_difference",
                "median"
            ),

            total_counterfactual_difference=(
                "counterfactual_difference",
                "sum"
            ),

            pit_faster_count=(
                "counterfactual_difference",
                lambda x: (x > 0).sum()
            ),

            stay_out_faster_count=(
                "counterfactual_difference",
                lambda x: (x < 0).sum()
            ),

            neutral_count=(
                "counterfactual_difference",
                lambda x: (x.abs() < 2).sum()
            ),

            high_confidence_count=(
                "counterfactual_confidence",
                lambda x: (x == "HIGH").sum()
            ),
        )
        .reset_index()
    )

    summary["pit_faster_rate"] = (
        summary["pit_faster_count"]
        / summary["evaluated_pit_stops"]
        * 100
    )

    summary["high_confidence_rate"] = (
        summary["high_confidence_count"]
        / summary["evaluated_pit_stops"]
        * 100
    )

    summary = summary.sort_values(
        [
            "average_confidence_adjusted_score",
            "average_strategy_score",
        ],
        ascending=False
    )

    print("\nTop drivers by confidence-adjusted score:")

    print(
        summary[
            [
                "driver_id",
                "evaluated_pit_stops",
                "average_strategy_score",
                "average_confidence_adjusted_score",
                "average_counterfactual_difference",
                "pit_faster_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return summary


# ============================================================
# RACE ANALYSIS
# ============================================================

def analyze_races(df):

    print("\n" + "=" * 70)
    print("RACE STRATEGY ANALYSIS")
    print("=" * 70)

    summary = (
        df.groupby(
            [
                "race_id",
                "race_round"
            ]
        )
        .agg(
            evaluated_pit_stops=(
                "counterfactual_difference",
                "count"
            ),

            average_strategy_score=(
                "strategy_score",
                "mean"
            ),

            average_confidence_adjusted_score=(
                "confidence_adjusted_score",
                "mean"
            ),

            average_counterfactual_difference=(
                "counterfactual_difference",
                "mean"
            ),

            total_counterfactual_difference=(
                "counterfactual_difference",
                "sum"
            ),

            pit_faster_count=(
                "counterfactual_difference",
                lambda x: (x > 0).sum()
            ),

            stay_out_faster_count=(
                "counterfactual_difference",
                lambda x: (x < 0).sum()
            ),

            high_confidence_count=(
                "counterfactual_confidence",
                lambda x: (x == "HIGH").sum()
            ),
        )
        .reset_index()
    )

    summary["pit_faster_rate"] = (
        summary["pit_faster_count"]
        / summary["evaluated_pit_stops"]
        * 100
    )

    summary["high_confidence_rate"] = (
        summary["high_confidence_count"]
        / summary["evaluated_pit_stops"]
        * 100
    )

    summary = summary.sort_values(
        "average_confidence_adjusted_score",
        ascending=False
    )

    print("\nBest races by strategy score:")

    print(
        summary[
            [
                "race_round",
                "evaluated_pit_stops",
                "average_strategy_score",
                "average_confidence_adjusted_score",
                "average_counterfactual_difference",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return summary


# ============================================================
# TYRE ANALYSIS
# ============================================================

def analyze_tyres(df):

    print("\n" + "=" * 70)
    print("TYRE STRATEGY ANALYSIS")
    print("=" * 70)

    summary = (
        df.groupby("tyre_before")
        .agg(
            evaluated_pit_stops=(
                "counterfactual_difference",
                "count"
            ),

            average_strategy_score=(
                "strategy_score",
                "mean"
            ),

            average_confidence_adjusted_score=(
                "confidence_adjusted_score",
                "mean"
            ),

            average_counterfactual_difference=(
                "counterfactual_difference",
                "mean"
            ),

            median_counterfactual_difference=(
                "counterfactual_difference",
                "median"
            ),

            average_degradation=(
                "estimated_degradation_per_lap",
                "mean"
            ),

            pit_faster_count=(
                "counterfactual_difference",
                lambda x: (x > 0).sum()
            ),

            stay_out_faster_count=(
                "counterfactual_difference",
                lambda x: (x < 0).sum()
            ),
        )
        .reset_index()
    )

    summary["pit_faster_rate"] = (
        summary["pit_faster_count"]
        / summary["evaluated_pit_stops"]
        * 100
    )

    summary = summary.sort_values(
        "average_confidence_adjusted_score",
        ascending=False
    )

    print("\nTyre compound results:")

    print(
        summary.to_string(index=False)
    )

    return summary


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def analyze_overall(df):

    print("\n" + "=" * 70)
    print("OVERALL STRATEGY ANALYSIS")
    print("=" * 70)

    total_records = len(df)

    pit_faster = (
        df["counterfactual_difference"] > 0
    ).sum()

    stay_out_faster = (
        df["counterfactual_difference"] < 0
    ).sum()

    neutral = (
        df["counterfactual_difference"].abs() < 2
    ).sum()

    high_confidence = (
        df["counterfactual_confidence"]
        == "HIGH"
    ).sum()

    overall = pd.DataFrame(
        [
            {
                "year": YEAR,

                "evaluated_counterfactuals":
                    total_records,

                "races_evaluated":
                    df["race_id"].nunique(),

                "drivers_evaluated":
                    df["driver_id"].nunique(),

                "pit_faster_count":
                    pit_faster,

                "stay_out_faster_count":
                    stay_out_faster,

                "neutral_count":
                    neutral,

                "pit_faster_rate":
                    pit_faster / total_records * 100,

                "stay_out_faster_rate":
                    stay_out_faster / total_records * 100,

                "high_confidence_count":
                    high_confidence,

                "high_confidence_rate":
                    high_confidence / total_records * 100,

                "average_strategy_score":
                    df["strategy_score"].mean(),

                "average_confidence_adjusted_score":
                    df[
                        "confidence_adjusted_score"
                    ].mean(),

                "average_counterfactual_difference":
                    df[
                        "counterfactual_difference"
                    ].mean(),

                "median_counterfactual_difference":
                    df[
                        "counterfactual_difference"
                    ].median(),

                "total_counterfactual_difference":
                    df[
                        "counterfactual_difference"
                    ].sum(),

                "average_degradation":
                    df[
                        "estimated_degradation_per_lap"
                    ].mean(),
            }
        ]
    )

    print("\nOverall metrics:")

    print(
        overall.T.to_string(
            header=False
        )
    )

    return overall


# ============================================================
# BEST / WORST DECISIONS
# ============================================================

def analyze_extreme_decisions(df):

    print("\n" + "=" * 70)
    print("BEST AND WORST STRATEGIC DECISIONS")
    print("=" * 70)

    columns = [
        "race_round",
        "race_id",
        "driver_id",
        "pit_lap",
        "tyre_before",
        "counterfactual_difference",
        "strategy_assessment",
        "strategy_score",
        "confidence_adjusted_score",
        "counterfactual_confidence",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    best = (
        df.sort_values(
            "counterfactual_difference",
            ascending=False
        )
        [columns]
        .head(10)
    )

    worst = (
        df.sort_values(
            "counterfactual_difference",
            ascending=True
        )
        [columns]
        .head(10)
    )

    print("\nTop 10 estimated beneficial pit stops:")

    print(
        best.to_string(index=False)
    )

    print("\nTop 10 estimated costly pit stops:")

    print(
        worst.to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    driver_summary,
    race_summary,
    tyre_summary,
    overall_summary
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    driver_summary.to_csv(
        DRIVER_OUTPUT,
        index=False
    )

    race_summary.to_csv(
        RACE_OUTPUT,
        index=False
    )

    tyre_summary.to_csv(
        TYRE_OUTPUT,
        index=False
    )

    overall_summary.to_csv(
        OVERALL_OUTPUT,
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS FILES SAVED")
    print("=" * 70)

    print(
        f"\nDriver summary:\n{DRIVER_OUTPUT}"
    )

    print(
        f"\nRace summary:\n{RACE_OUTPUT}"
    )

    print(
        f"\nTyre summary:\n{TYRE_OUTPUT}"
    )

    print(
        f"\nOverall summary:\n{OVERALL_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY GRADER")
    print("2025 STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 70)

    df = load_data()

    df = clean_data(df)

    print(
        f"Usable strategy records: {len(df):,}"
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    driver_summary = analyze_drivers(
        df
    )

    race_summary = analyze_races(
        df
    )

    tyre_summary = analyze_tyres(
        df
    )

    overall_summary = analyze_overall(
        df
    )

    analyze_extreme_decisions(
        df
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        driver_summary,
        race_summary,
        tyre_summary,
        overall_summary
    )

    print("\n" + "=" * 70)
    print("STRATEGY PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()