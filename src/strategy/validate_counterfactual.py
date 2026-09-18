import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

DB_NAME = "f1_strategy_grader_ML"
DB_USER = "postgres"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = "localhost"
DB_PORT = "5433"

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "2025_counterfactual_results.csv"
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY GRADER")
    print("2025 COUNTERFACTUAL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD COUNTERFACTUAL RESULTS
    # --------------------------------------------------------

    print("\nLoading counterfactual results...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Counterfactual file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Counterfactual records loaded: {len(df):,}"
    )

    if df.empty:

        raise RuntimeError(
            "Counterfactual dataset is empty."
        )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASIC VALIDATION")
    print("=" * 70)

    print(
        f"Records                    : {len(df):,}"
    )

    print(
        f"Races represented          : {df['race_id'].nunique()}"
    )

    print(
        f"Drivers represented        : {df['driver_id'].nunique()}"
    )

    # --------------------------------------------------------
    # DEGRADATION VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEGRADATION VALIDATION")
    print("=" * 70)

    zero_degradation = (
        df["estimated_degradation_per_lap"] == 0
    ).sum()

    nonzero_degradation = (
        df["estimated_degradation_per_lap"] > 0
    ).sum()

    print(
        f"Zero degradation           : {zero_degradation:,}"
    )

    print(
        f"Positive degradation       : {nonzero_degradation:,}"
    )

    print(
        f"Zero degradation %         : "
        f"{zero_degradation / len(df) * 100:.1f}%"
    )

    print(
        "\nDegradation statistics:"
    )

    print(
        df[
            "estimated_degradation_per_lap"
        ].describe().to_string()
    )

    # --------------------------------------------------------
    # CONFIDENCE VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 70)

    print(
        df[
            "counterfactual_confidence"
        ].value_counts().to_string()
    )

    # --------------------------------------------------------
    # COUNTERFACTUAL DECISION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COUNTERFACTUAL DECISIONS")
    print("=" * 70)

    pit_faster = (
        df["counterfactual_difference"] > 0
    ).sum()

    stay_out_faster = (
        df["counterfactual_difference"] < 0
    ).sum()

    equal = (
        df["counterfactual_difference"] == 0
    ).sum()

    print(
        f"Pit estimated faster       : {pit_faster:,}"
    )

    print(
        f"Stay-out estimated faster  : {stay_out_faster:,}"
    )

    print(
        f"Approximately equal        : {equal:,}"
    )

    # --------------------------------------------------------
    # DIFFERENCE STATISTICS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COUNTERFACTUAL DIFFERENCE")
    print("=" * 70)

    print(
        f"Average                   : "
        f"{df['counterfactual_difference'].mean():.3f} sec"
    )

    print(
        f"Median                    : "
        f"{df['counterfactual_difference'].median():.3f} sec"
    )

    print(
        f"Standard deviation        : "
        f"{df['counterfactual_difference'].std():.3f} sec"
    )

    print(
        f"Minimum                   : "
        f"{df['counterfactual_difference'].min():.3f} sec"
    )

    print(
        f"Maximum                   : "
        f"{df['counterfactual_difference'].max():.3f} sec"
    )

    # --------------------------------------------------------
    # NEUTRALIZED LAPS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NEUTRALIZED LAP VALIDATION")
    print("=" * 70)

    if "neutralized_laps_removed" in df.columns:

        print(
            df[
                "neutralized_laps_removed"
            ].value_counts().sort_index().to_string()
        )

        affected = (
            df["neutralized_laps_removed"] > 0
        ).sum()

        print(
            f"\nRecords affected by neutralized laps: "
            f"{affected:,}"
        )

    else:

        print(
            "neutralized_laps_removed column not found."
        )

    # --------------------------------------------------------
    # CONFIDENCE VS DECISION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFIDENCE VS COUNTERFACTUAL DECISION")
    print("=" * 70)

    decision_df = df.copy()

    decision_df["decision"] = "EQUAL"

    decision_df.loc[
        decision_df["counterfactual_difference"] > 0,
        "decision"
    ] = "PIT_FASTER"

    decision_df.loc[
        decision_df["counterfactual_difference"] < 0,
        "decision"
    ] = "STAY_OUT_FASTER"

    table = pd.crosstab(
        decision_df["counterfactual_confidence"],
        decision_df["decision"]
    )

    print(
        table.to_string()
    )

    # --------------------------------------------------------
    # TYRE ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TYRE COMPOUND ANALYSIS")
    print("=" * 70)

    tyre_summary = (
        df.groupby("tyre_before")
        .agg(
            decisions=(
                "counterfactual_difference",
                "count"
            ),
            average_difference=(
                "counterfactual_difference",
                "mean"
            ),
            median_difference=(
                "counterfactual_difference",
                "median"
            ),
            average_degradation=(
                "estimated_degradation_per_lap",
                "mean"
            )
        )
        .sort_values(
            "average_difference",
            ascending=False
        )
    )

    print(
        tyre_summary.to_string()
    )

    # --------------------------------------------------------
    # LARGE DIFFERENCE CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXTREME VALUE CHECK")
    print("=" * 70)

    extreme = df[
        df["counterfactual_difference"].abs() > 60
    ]

    print(
        f"Records with |difference| > 60 sec: "
        f"{len(extreme)}"
    )

    if not extreme.empty:

        print(
            "\nLargest absolute differences:"
        )

        print(
            extreme[
                [
                    "race_id",
                    "driver_id",
                    "pit_lap",
                    "tyre_before",
                    "counterfactual_difference",
                    "counterfactual_confidence"
                ]
            ]
            .assign(
                absolute_difference=lambda x:
                    x["counterfactual_difference"].abs()
            )
            .sort_values(
                "absolute_difference",
                ascending=False
            )
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # DATABASE CROSS-CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATABASE CROSS-CHECK")
    print("=" * 70)

    conn = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM pit_stops;
            """
        )

        pit_count = cursor.fetchone()[0]

        print(
            f"Pit stops in database      : {pit_count:,}"
        )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM races
            WHERE season = %s;
            """,
            (YEAR,)
        )

        race_count = cursor.fetchone()[0]

        print(
            f"2025 races in database     : {race_count}"
        )

        cursor.close()

    finally:

        if conn is not None:

            conn.close()

    # --------------------------------------------------------
    # FINAL ASSESSMENT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION ASSESSMENT")
    print("=" * 70)

    coverage_note = ""

    if zero_degradation / len(df) > 0.50:

        coverage_note = (
            "WARNING: More than 50% of counterfactuals "
            "use zero estimated degradation."
        )

    else:

        coverage_note = (
            "Degradation model is active for the majority "
            "of counterfactual records."
        )

    print(
        f"\n{coverage_note}"
    )

    if len(extreme) == 0:

        print(
            "✓ No extreme counterfactual differences "
            "above ±60 seconds."
        )

    else:

        print(
            f"⚠ {len(extreme)} records exceed ±60 seconds "
            "and should be reviewed."
        )

    if "neutralized_laps_removed" in df.columns:

        print(
            "✓ Race-condition filtering is present."
        )

    if "counterfactual_confidence" in df.columns:

        print(
            "✓ Counterfactual confidence classification "
            "is present."
        )

    print(
        "\nCounterfactual validation complete."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()