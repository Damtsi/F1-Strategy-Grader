import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

DB_NAME = "f1_strategy_grader_ML"
DB_USER = "postgres"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = "localhost"
DB_PORT = "5433"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    print("Connecting to PostgreSQL...")

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    print("PostgreSQL connection successful.")

    return conn


# ============================================================
# LOAD STRATEGY DATA
# ============================================================

def load_strategy_data(conn):

    print("\nLoading strategy data...")

    # IMPORTANT:
    # We deliberately use s.* instead of guessing the column
    # names inside strategy_scores.
    #
    # This makes the analysis compatible with the actual
    # database table created by the scoring script.

    query = """
        SELECT
            s.*,

            r.race_name,
            r.season,

            d.driver_code,
            d.driver_name,

            t.team_name

        FROM strategy_scores s

        JOIN races r
            ON s.race_id = r.race_id

        JOIN drivers d
            ON s.driver_id = d.driver_id

        JOIN teams t
            ON d.team_id = t.team_id

        WHERE r.season = %s

        ORDER BY
            r.race_id,
            d.driver_code;
    """

    df = pd.read_sql(
        query,
        conn,
        params=(YEAR,)
    )

    print(
        f"Strategy records loaded: {len(df):,}"
    )

    print(
        f"Columns available: {len(df.columns)}"
    )

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    lower_columns = {
        column.lower(): column
        for column in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_columns:

            return lower_columns[name.lower()]

    return None


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing analysis data...")

    df = df.copy()

    score_column = find_column(
        df,
        [
            "strategy_call_score",
            "strategy_score",
            "score"
        ]
    )

    verdict_column = find_column(
        df,
        [
            "verdict",
            "strategy_verdict"
        ]
    )

    if score_column is None:

        raise ValueError(
            "Could not find the strategy score column."
        )

    if verdict_column is None:

        raise ValueError(
            "Could not find the verdict column."
        )

    print(
        f"Score column   : {score_column}"
    )

    print(
        f"Verdict column : {verdict_column}"
    )

    df["_score"] = pd.to_numeric(
        df[score_column],
        errors="coerce"
    )

    df["_verdict"] = (
        df[verdict_column]
        .astype(str)
    )

    return df


# ============================================================
# OVERALL SUMMARY
# ============================================================

def overall_summary(df):

    print("\n" + "=" * 70)
    print("OVERALL STRATEGY PERFORMANCE")
    print("=" * 70)

    total = len(df)

    average_score = df["_score"].mean()

    print(
        f"Season                 : {YEAR}"
    )

    print(
        f"Strategy decisions     : {total:,}"
    )

    print(
        f"Average strategy score : {average_score:.2f}"
    )

    print("\nVerdict distribution:")

    verdicts = [
        "Excellent",
        "Good",
        "Neutral",
        "Poor",
        "Very Poor"
    ]

    for verdict in verdicts:

        count = (
            df["_verdict"]
            .eq(verdict)
            .sum()
        )

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"  {verdict:<11}: "
            f"{count:>4} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# DRIVER ANALYSIS
# ============================================================

def driver_analysis(df):

    print("\n" + "=" * 70)
    print("DRIVER STRATEGY RANKING")
    print("=" * 70)

    summary = (
        df.groupby(
            [
                "driver_code",
                "driver_name",
                "team_name"
            ],
            dropna=False
        )
        .agg(
            decisions=(
                "_score",
                "count"
            ),
            average_score=(
                "_score",
                "mean"
            ),
            best_score=(
                "_score",
                "max"
            ),
            worst_score=(
                "_score",
                "min"
            )
        )
        .reset_index()
    )

    summary[
        "average_score"
    ] = summary[
        "average_score"
    ].round(2)

    summary[
        "best_score"
    ] = summary[
        "best_score"
    ].round(2)

    summary[
        "worst_score"
    ] = summary[
        "worst_score"
    ].round(2)

    summary = summary.sort_values(
        "average_score",
        ascending=False
    )

    print(
        f"\n{'Rank':<6}"
        f"{'Driver':<28}"
        f"{'Team':<22}"
        f"{'Calls':<8}"
        f"{'Average':<10}"
    )

    print("-" * 74)

    for rank, (_, row) in enumerate(
        summary.iterrows(),
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{row['driver_name']:<28}"
            f"{row['team_name']:<22}"
            f"{row['decisions']:<8}"
            f"{row['average_score']:<10.2f}"
        )

    return summary


# ============================================================
# TEAM ANALYSIS
# ============================================================

def team_analysis(df):

    print("\n" + "=" * 70)
    print("TEAM STRATEGY RANKING")
    print("=" * 70)

    summary = (
        df.groupby(
            "team_name",
            dropna=False
        )
        .agg(
            decisions=(
                "_score",
                "count"
            ),
            average_score=(
                "_score",
                "mean"
            )
        )
        .reset_index()
    )

    summary[
        "average_score"
    ] = summary[
        "average_score"
    ].round(2)

    summary = summary.sort_values(
        "average_score",
        ascending=False
    )

    print(
        f"\n{'Rank':<6}"
        f"{'Team':<25}"
        f"{'Calls':<10}"
        f"{'Average':<10}"
    )

    print("-" * 55)

    for rank, (_, row) in enumerate(
        summary.iterrows(),
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{row['team_name']:<25}"
            f"{row['decisions']:<10}"
            f"{row['average_score']:<10.2f}"
        )

    return summary


# ============================================================
# RACE ANALYSIS
# ============================================================

def race_analysis(df):

    print("\n" + "=" * 70)
    print("RACE STRATEGY RANKING")
    print("=" * 70)

    summary = (
        df.groupby(
            [
                "race_id",
                "race_name"
            ],
            dropna=False
        )
        .agg(
            decisions=(
                "_score",
                "count"
            ),
            average_score=(
                "_score",
                "mean"
            ),
            best_score=(
                "_score",
                "max"
            ),
            worst_score=(
                "_score",
                "min"
            )
        )
        .reset_index()
    )

    summary[
        "average_score"
    ] = summary[
        "average_score"
    ].round(2)

    summary[
        "best_score"
    ] = summary[
        "best_score"
    ].round(2)

    summary[
        "worst_score"
    ] = summary[
        "worst_score"
    ].round(2)

    summary = summary.sort_values(
        "average_score",
        ascending=False
    )

    print(
        f"\n{'Race':<30}"
        f"{'Calls':<8}"
        f"{'Average':<10}"
        f"{'Best':<10}"
        f"{'Worst':<10}"
    )

    print("-" * 68)

    for _, row in summary.iterrows():

        print(
            f"{str(row['race_name'])[:29]:<30}"
            f"{row['decisions']:<8}"
            f"{row['average_score']:<10.2f}"
            f"{row['best_score']:<10.2f}"
            f"{row['worst_score']:<10.2f}"
        )

    return summary


# ============================================================
# TYRE TRANSITION ANALYSIS
# ============================================================

def tyre_transition_analysis(df):

    print("\n" + "=" * 70)
    print("TYRE TRANSITION ANALYSIS")
    print("=" * 70)

    before_column = find_column(
        df,
        [
            "tyre_before",
            "compound_before",
            "old_compound"
        ]
    )

    after_column = find_column(
        df,
        [
            "tyre_after",
            "compound_after",
            "new_compound"
        ]
    )

    if (
        before_column is None
        or after_column is None
    ):

        print(
            "\nTyre transition columns "
            "not available."
        )

        return pd.DataFrame()

    summary = (
        df.groupby(
            [
                before_column,
                after_column
            ],
            dropna=False
        )
        .agg(
            decisions=(
                "_score",
                "count"
            ),
            average_score=(
                "_score",
                "mean"
            )
        )
        .reset_index()
    )

    summary[
        "average_score"
    ] = summary[
        "average_score"
    ].round(2)

    summary = summary.sort_values(
        "average_score",
        ascending=False
    )

    print(
        f"\n{'Transition':<25}"
        f"{'Calls':<10}"
        f"{'Average':<10}"
    )

    print("-" * 50)

    for _, row in summary.iterrows():

        before = row[before_column]
        after = row[after_column]

        transition = (
            f"{before} -> {after}"
        )

        print(
            f"{transition:<25}"
            f"{row['decisions']:<10}"
            f"{row['average_score']:<10.2f}"
        )

    return summary


# ============================================================
# BEST STRATEGY DECISIONS
# ============================================================

def best_decisions(df):

    print("\n" + "=" * 70)
    print("TOP 15 STRATEGY DECISIONS")
    print("=" * 70)

    columns = [
        "driver_code",
        "driver_name",
        "team_name",
        "race_name",
        "_score",
        "_verdict"
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    best = (
        df.sort_values(
            "_score",
            ascending=False
        )
        .head(15)
    )

    print()

    for number, (_, row) in enumerate(
        best.iterrows(),
        start=1
    ):

        print(
            f"{number:>2}. "
            f"{row['driver_code']} - "
            f"{row['driver_name']} | "
            f"{row['race_name']} | "
            f"Score: {row['_score']:.2f} | "
            f"{row['_verdict']}"
        )

    return best


# ============================================================
# WORST STRATEGY DECISIONS
# ============================================================

def worst_decisions(df):

    print("\n" + "=" * 70)
    print("BOTTOM 15 STRATEGY DECISIONS")
    print("=" * 70)

    worst = (
        df.sort_values(
            "_score",
            ascending=True
        )
        .head(15)
    )

    print()

    for number, (_, row) in enumerate(
        worst.iterrows(),
        start=1
    ):

        print(
            f"{number:>2}. "
            f"{row['driver_code']} - "
            f"{row['driver_name']} | "
            f"{row['race_name']} | "
            f"Score: {row['_score']:.2f} | "
            f"{row['_verdict']}"
        )

    return worst


# ============================================================
# INSIGHTS
# ============================================================

def generate_insights(
    df,
    driver_summary,
    team_summary,
    race_summary,
    tyre_summary
):

    print("\n" + "=" * 70)
    print("STRATEGIC INSIGHTS")
    print("=" * 70)

    # --------------------------------------------------------
    # BEST DRIVER
    # --------------------------------------------------------

    if not driver_summary.empty:

        best_driver = driver_summary.iloc[0]

        print("\nBest driver strategy performance:")

        print(
            f"  {best_driver['driver_name']} "
            f"({best_driver['team_name']})"
        )

        print(
            f"  Average score: "
            f"{best_driver['average_score']:.2f}"
        )

    # --------------------------------------------------------
    # WORST DRIVER
    # --------------------------------------------------------

    if not driver_summary.empty:

        worst_driver = driver_summary.iloc[-1]

        print("\nLowest driver strategy performance:")

        print(
            f"  {worst_driver['driver_name']} "
            f"({worst_driver['team_name']})"
        )

        print(
            f"  Average score: "
            f"{worst_driver['average_score']:.2f}"
        )

    # --------------------------------------------------------
    # BEST TEAM
    # --------------------------------------------------------

    if not team_summary.empty:

        best_team = team_summary.iloc[0]

        print("\nBest team strategy performance:")

        print(
            f"  {best_team['team_name']}"
        )

        print(
            f"  Average score: "
            f"{best_team['average_score']:.2f}"
        )

    # --------------------------------------------------------
    # BEST RACE
    # --------------------------------------------------------

    if not race_summary.empty:

        best_race = race_summary.iloc[0]

        print("\nHighest-scoring race:")

        print(
            f"  {best_race['race_name']}"
        )

        print(
            f"  Average score: "
            f"{best_race['average_score']:.2f}"
        )

    # --------------------------------------------------------
    # WORST RACE
    # --------------------------------------------------------

    if not race_summary.empty:

        worst_race = race_summary.iloc[-1]

        print("\nLowest-scoring race:")

        print(
            f"  {worst_race['race_name']}"
        )

        print(
            f"  Average score: "
            f"{worst_race['average_score']:.2f}"
        )

    # --------------------------------------------------------
    # POSITIVE / NEGATIVE CALLS
    # --------------------------------------------------------

    positive = df[
        df["_verdict"].isin(
            [
                "Excellent",
                "Good"
            ]
        )
    ]

    negative = df[
        df["_verdict"].isin(
            [
                "Poor",
                "Very Poor"
            ]
        )
    ]

    positive_pct = (
        len(positive) / len(df) * 100
    )

    negative_pct = (
        len(negative) / len(df) * 100
    )

    print(
        f"\nPositive strategy calls "
        f"(Excellent + Good): "
        f"{positive_pct:.1f}%"
    )

    print(
        f"Negative strategy calls "
        f"(Poor + Very Poor): "
        f"{negative_pct:.1f}%"
    )


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    df,
    driver_summary,
    team_summary,
    race_summary,
    tyre_summary
):

    output_dir = (
        BASE_DIR
        / "data"
        / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove internal helper columns
    clean_df = df.copy()

    for column in [
        "_score",
        "_verdict"
    ]:

        if column in clean_df.columns:

            clean_df = clean_df.drop(
                columns=[column]
            )

    clean_df.to_csv(
        output_dir
        / "strategy_analysis_data.csv",
        index=False
    )

    driver_summary.to_csv(
        output_dir
        / "driver_strategy_ranking.csv",
        index=False
    )

    team_summary.to_csv(
        output_dir
        / "team_strategy_ranking.csv",
        index=False
    )

    race_summary.to_csv(
        output_dir
        / "race_strategy_ranking.csv",
        index=False
    )

    if not tyre_summary.empty:

        tyre_summary.to_csv(
            output_dir
            / "tyre_transition_analysis.csv",
            index=False
        )

    print(
        "\nAnalysis files saved to:"
    )

    print(output_dir)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY GRADER")
    print("2025 STRATEGY ANALYSIS")
    print("=" * 70)

    conn = None

    try:

        conn = get_connection()

        print("\n" + "=" * 70)
        print("LOADING STRATEGY SCORES")
        print("=" * 70)

        df = load_strategy_data(
            conn
        )

        if df.empty:

            print(
                "\nERROR: "
                "strategy_scores contains no records."
            )

            return

        df = prepare_data(df)

        print_overall = overall_summary

        print_overall(df)

        driver_summary = driver_analysis(
            df
        )

        team_summary = team_analysis(
            df
        )

        race_summary = race_analysis(
            df
        )

        tyre_summary = tyre_transition_analysis(
            df
        )

        best = best_decisions(
            df
        )

        worst = worst_decisions(
            df
        )

        generate_insights(
            df,
            driver_summary,
            team_summary,
            race_summary,
            tyre_summary
        )

        save_reports(
            df,
            driver_summary,
            team_summary,
            race_summary,
            tyre_summary
        )

        print("\n" + "=" * 70)
        print("2025 STRATEGY ANALYSIS COMPLETE")
        print("=" * 70)

        print(
            f"Records analysed : {len(df):,}"
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("FATAL ERROR")
        print("=" * 70)

        print(
            f"{type(error).__name__}: {error}"
        )

    finally:

        if conn is not None:

            conn.close()

            print(
                "\nPostgreSQL connection closed."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()