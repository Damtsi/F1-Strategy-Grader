import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
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
# HELPERS
# ============================================================

def safe_float(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

        return float(value)

    except Exception:
        return None


def safe_int(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

        return int(value)

    except Exception:
        return None


def clip_score(value):

    if value is None:
        return 50.0

    return float(
        np.clip(float(value), 0, 100)
    )


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def calculate_counterfactual_score(difference):

    if difference is None:
        return 50.0

    score = 50.0 + (
        50.0 * np.tanh(
            difference / 1.0
        )
    )

    return clip_score(score)


def calculate_tyre_age_score(tyre_age):

    if tyre_age is None:
        return 50.0

    tyre_age = float(tyre_age)

    if tyre_age < 5:
        return 35.0

    elif tyre_age < 8:
        return 60.0

    elif tyre_age < 15:
        return 85.0

    elif tyre_age < 25:
        return 95.0

    elif tyre_age < 35:
        return 80.0

    elif tyre_age < 45:
        return 60.0

    else:
        return 40.0


def calculate_pace_context_score(
    pre_pace,
    field_pace
):

    if (
        pre_pace is None
        or field_pace is None
    ):
        return 50.0

    difference = (
        float(pre_pace)
        -
        float(field_pace)
    )

    score = 70.0 - (
        difference * 35.0
    )

    return clip_score(score)


def calculate_position_context_score(
    position_before,
    position_after
):

    if (
        position_before is None
        or position_after is None
    ):
        return 50.0

    position_change = (
        float(position_before)
        -
        float(position_after)
    )

    score = 60.0 + (
        position_change * 15.0
    )

    return clip_score(score)


def calculate_verdict(score):

    if score >= 85:
        return "Excellent"

    elif score >= 70:
        return "Good"

    elif score >= 55:
        return "Neutral"

    elif score >= 40:
        return "Poor"

    else:
        return "Very Poor"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(conn):

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    races = pd.read_sql(
        """
        SELECT
            race_id,
            season,
            race_name
        FROM races
        WHERE season = 2025
        ORDER BY race_id;
        """,
        conn
    )

    print(
        f"Races loaded       : {len(races)}"
    )


    drivers = pd.read_sql(
        """
        SELECT
            driver_id,
            driver_code,
            driver_name
          
        FROM drivers
        ORDER BY driver_id;
        """,
        conn
    )

    print(
        f"Drivers loaded     : {len(drivers)}"
    )


    laps = pd.read_sql(
        """
        SELECT
            race_id,
            driver_id,
            lap_number,
            lap_time_seconds,
            compound,
            tyre_life,
            stint,
            position
        FROM laps
        WHERE lap_time_seconds IS NOT NULL
        ORDER BY
            race_id,
            driver_id,
            lap_number;
        """,
        conn
    )

    print(
        f"Lap records loaded : {len(laps):,}"
    )


    pits = pd.read_sql(
        """
        SELECT
            race_id,
            driver_id,
            pit_lap,
            tyre_before,
            tyre_after,
            tyre_age_before,
            stint_before,
            stint_after
        FROM pit_stops
        ORDER BY
            race_id,
            driver_id,
            pit_lap;
        """,
        conn
    )

    print(
        f"Pit stops loaded   : {len(pits):,}"
    )

    return races, drivers, laps, pits


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_laps(laps):

    print("\nPreparing lap data...")

    laps = laps.copy()

    numeric_columns = [
        "race_id",
        "driver_id",
        "lap_number",
        "lap_time_seconds",
        "tyre_life",
        "stint",
        "position"
    ]

    for column in numeric_columns:

        if column in laps.columns:

            laps[column] = pd.to_numeric(
                laps[column],
                errors="coerce"
            )

    laps = laps[
        laps["lap_time_seconds"].notna()
    ].copy()

    laps = laps[
        (laps["lap_time_seconds"] > 50)
        &
        (laps["lap_time_seconds"] < 200)
    ].copy()

    laps = laps.sort_values(
        [
            "race_id",
            "driver_id",
            "lap_number"
        ]
    )

    print(
        f"Usable lap records : {len(laps):,}"
    )

    return laps


def prepare_pits(pits):

    print("\nPreparing pit-stop data...")

    pits = pits.copy()

    numeric_columns = [
        "race_id",
        "driver_id",
        "pit_lap",
        "tyre_age_before",
        "stint_before",
        "stint_after"
    ]

    for column in numeric_columns:

        if column in pits.columns:

            pits[column] = pd.to_numeric(
                pits[column],
                errors="coerce"
            )

    pits = pits.dropna(
        subset=[
            "race_id",
            "driver_id",
            "pit_lap"
        ]
    ).copy()

    pits["race_id"] = (
        pits["race_id"].astype(int)
    )

    pits["driver_id"] = (
        pits["driver_id"].astype(int)
    )

    pits["pit_lap"] = (
        pits["pit_lap"].astype(int)
    )

    print(
        f"Usable pit stops   : {len(pits):,}"
    )

    return pits


# ============================================================
# PACE FUNCTIONS
# ============================================================

def calculate_pre_pit_pace(
    driver_laps,
    pit_lap
):

    window = driver_laps[
        (
            driver_laps["lap_number"]
            >= pit_lap - 5
        )
        &
        (
            driver_laps["lap_number"]
            < pit_lap
        )
    ].copy()

    if window.empty:
        return None

    values = (
        window["lap_time_seconds"]
        .dropna()
    )

    if values.empty:
        return None

    return float(
        values.median()
    )


def calculate_post_pit_pace(
    driver_laps,
    pit_lap
):

    window = driver_laps[
        (
            driver_laps["lap_number"]
            > pit_lap
        )
        &
        (
            driver_laps["lap_number"]
            <= pit_lap + 5
        )
    ].copy()

    if window.empty:
        return None

    values = (
        window["lap_time_seconds"]
        .dropna()
    )

    if values.empty:
        return None

    return float(
        values.median()
    )


def calculate_field_pace(
    race_laps,
    pit_lap
):

    window = race_laps[
        (
            race_laps["lap_number"]
            >= pit_lap - 2
        )
        &
        (
            race_laps["lap_number"]
            <= pit_lap
        )
    ].copy()

    if window.empty:
        return None

    values = (
        window["lap_time_seconds"]
        .dropna()
    )

    if values.empty:
        return None

    return float(
        values.median()
    )


# ============================================================
# POSITION FUNCTIONS
# ============================================================

def get_position_before(
    driver_laps,
    pit_lap
):

    window = driver_laps[
        driver_laps["lap_number"]
        <= pit_lap
    ].sort_values(
        "lap_number"
    )

    if window.empty:
        return None

    return safe_float(
        window.iloc[-1]["position"]
    )


def get_position_after(
    driver_laps,
    pit_lap
):

    window = driver_laps[
        (
            driver_laps["lap_number"]
            > pit_lap
        )
        &
        (
            driver_laps["lap_number"]
            <= pit_lap + 3
        )
    ].sort_values(
        "lap_number"
    )

    if window.empty:
        return None

    return safe_float(
        window.iloc[-1]["position"]
    )


# ============================================================
# GENERATE STRATEGY SCORES
# ============================================================

def generate_strategy_scores(
    races,
    drivers,
    laps,
    pits
):

    print("\n" + "=" * 70)
    print("GENERATING STRATEGY SCORES")
    print("=" * 70)

    strategy_records = []

    total = len(pits)

    for index, pit in pits.iterrows():

        try:

            race_id = safe_int(
                pit["race_id"]
            )

            driver_id = safe_int(
                pit["driver_id"]
            )

            pit_lap = safe_int(
                pit["pit_lap"]
            )

            if (
                race_id is None
                or driver_id is None
                or pit_lap is None
            ):
                continue


            # ------------------------------------------------
            # DRIVER LAPS
            # ------------------------------------------------

            driver_laps = laps[
                (
                    laps["race_id"]
                    == race_id
                )
                &
                (
                    laps["driver_id"]
                    == driver_id
                )
            ].copy()

            if driver_laps.empty:
                continue


            # ------------------------------------------------
            # RACE LAPS
            # ------------------------------------------------

            race_laps = laps[
                laps["race_id"]
                == race_id
            ].copy()

            if race_laps.empty:
                continue


            # ------------------------------------------------
            # PRE-PIT PACE
            # ------------------------------------------------

            pre_pit_pace = (
                calculate_pre_pit_pace(
                    driver_laps,
                    pit_lap
                )
            )


            # ------------------------------------------------
            # POST-PIT PACE
            # ------------------------------------------------

            post_pit_pace = (
                calculate_post_pit_pace(
                    driver_laps,
                    pit_lap
                )
            )


            # ------------------------------------------------
            # FIELD PACE
            # ------------------------------------------------

            field_pace = (
                calculate_field_pace(
                    race_laps,
                    pit_lap
                )
            )


            # ------------------------------------------------
            # COUNTERFACTUAL DIFFERENCE
            # ------------------------------------------------

            if (
                pre_pit_pace is not None
                and post_pit_pace is not None
            ):

                difference = (
                    pre_pit_pace
                    -
                    post_pit_pace
                )

            else:

                difference = None


            counterfactual_score = (
                calculate_counterfactual_score(
                    difference
                )
            )


            # ------------------------------------------------
            # TYRE AGE
            # ------------------------------------------------

            tyre_age = safe_float(
                pit["tyre_age_before"]
            )

            tyre_age_score = (
                calculate_tyre_age_score(
                    tyre_age
                )
            )


            # ------------------------------------------------
            # PACE CONTEXT
            # ------------------------------------------------

            pace_context_score = (
                calculate_pace_context_score(
                    pre_pit_pace,
                    field_pace
                )
            )


            # ------------------------------------------------
            # POSITION
            # ------------------------------------------------

            position_before = (
                get_position_before(
                    driver_laps,
                    pit_lap
                )
            )

            position_after = (
                get_position_after(
                    driver_laps,
                    pit_lap
                )
            )

            position_context_score = (
                calculate_position_context_score(
                    position_before,
                    position_after
                )
            )


            # ------------------------------------------------
            # FINAL SCORE
            # ------------------------------------------------

            final_score = (

                counterfactual_score * 0.40

                +

                tyre_age_score * 0.20

                +

                pace_context_score * 0.20

                +

                position_context_score * 0.20

            )

            final_score = clip_score(
                final_score
            )


            # ------------------------------------------------
            # VERDICT
            # ------------------------------------------------

            verdict = calculate_verdict(
                final_score
            )


            # ------------------------------------------------
            # DATABASE RECORD
            # ------------------------------------------------

            record = (

                int(race_id),

                int(driver_id),

                int(pit_lap),

                safe_float(difference),

                safe_float(counterfactual_score),

                safe_float(tyre_age_score),

                safe_float(pace_context_score),

                safe_float(position_context_score),

                safe_float(final_score),

                str(verdict)

            )

            strategy_records.append(
                record
            )


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                (index + 1) % 100 == 0
                or
                (index + 1) == total
            ):

                print(
                    f"Processed "
                    f"{index + 1:,} / "
                    f"{total:,}"
                )


        except Exception as error:

            print(
                f"Warning at pit stop "
                f"{index + 1}: {error}"
            )

            continue


    print("\n" + "-" * 70)

    print(
        f"Strategy records generated: "
        f"{len(strategy_records):,}"
    )

    print("-" * 70)

    return strategy_records


# ============================================================
# INSERT INTO DATABASE
# ============================================================

def insert_strategy_scores(
    conn,
    records
):

    if not records:

        print(
            "\nERROR: ZERO strategy records generated."
        )

        return 0


    print("\n" + "=" * 70)
    print("INSERTING STRATEGY SCORES")
    print("=" * 70)


    cursor = conn.cursor()


    try:

        cursor.execute(
            "DELETE FROM strategy_scores;"
        )


        query = """
            INSERT INTO strategy_scores (
                race_id,
                driver_id,
                decision_lap,
                counterfactual_difference,
                counterfactual_score,
                tyre_age_score,
                pace_context_score,
                position_context_score,
                strategy_call_score,
                verdict
            )
            VALUES %s
        """


        execute_values(
            cursor,
            query,
            records,
            page_size=500
        )


        conn.commit()


        print(
            f"Inserted strategy records: "
            f"{len(records):,}"
        )


        return len(records)


    except Exception as error:

        conn.rollback()

        print(
            "\nDATABASE INSERT ERROR:"
        )

        print(error)

        raise


    finally:

        cursor.close()


# ============================================================
# VERIFY
# ============================================================

def verify_results(conn):

    print("\n" + "=" * 70)
    print("VERIFYING RESULTS")
    print("=" * 70)


    cursor = conn.cursor()


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM strategy_scores;
        """
    )

    total = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT COUNT(DISTINCT race_id)
        FROM strategy_scores;
        """
    )

    races = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT COUNT(DISTINCT driver_id)
        FROM strategy_scores;
        """
    )

    drivers = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT
            ROUND(
                AVG(strategy_call_score)::numeric,
                2
            )
        FROM strategy_scores;
        """
    )

    average_score = cursor.fetchone()[0]


    print(
        f"\nStrategy decisions : {total:,}"
    )

    print(
        f"Races represented  : {races}"
    )

    print(
        f"Drivers represented: {drivers}"
    )

    print(
        f"Average score      : {average_score}"
    )


    print("\nVerdict distribution:")


    cursor.execute(
        """
        SELECT
            verdict,
            COUNT(*)
        FROM strategy_scores
        GROUP BY verdict
        ORDER BY COUNT(*) DESC;
        """
    )


    for verdict, count in cursor.fetchall():

        print(
            f"  {verdict:<12} : {count:,}"
        )


    print("\nTop drivers:")


    cursor.execute(
    """
    SELECT
        d.driver_code,
        d.driver_name,
        t.team_name,
        COUNT(*) AS decisions,
        ROUND(
            AVG(
                s.strategy_call_score
            )::numeric,
            2
        ) AS avg_score

    FROM strategy_scores s

    JOIN drivers d
        ON s.driver_id = d.driver_id

    JOIN teams t
        ON d.team_id = t.team_id

    GROUP BY
        d.driver_code,
        d.driver_name,
        t.team_name

    ORDER BY
        avg_score DESC;
    """
)


    for row in cursor.fetchall():

       print(
        f"  {row[0]:<5}"
        f"{row[1]:<28}"
        f"Decisions: {row[2]:<4}"
        f"Score: {row[3]}"
    )


    cursor.close()

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("F1 STRATEGY GRADER")
    print("2025 FULL SEASON STRATEGY SCORING")
    print("=" * 70)


    conn = None


    try:

        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        conn = get_connection()


        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        (
            races,
            drivers,
            laps,
            pits
        ) = load_data(conn)


        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        laps = prepare_laps(
            laps
        )

        pits = prepare_pits(
            pits
        )


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        records = (
            generate_strategy_scores(
                races,
                drivers,
                laps,
                pits
            )
        )


        # ----------------------------------------------------
        # INSERT
        # ----------------------------------------------------

        inserted = (
            insert_strategy_scores(
                conn,
                records
            )
        )


        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        total = verify_results(
            conn
        )


        # ----------------------------------------------------
        # FINAL MESSAGE
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("2025 STRATEGY SCORING COMPLETE")
        print("=" * 70)

        print(
            f"\nRecords generated : {len(records):,}"
        )

        print(
            f"Records inserted  : {inserted:,}"
        )

        print(
            f"Database total    : {total:,}"
        )


    except Exception as error:

        print("\n" + "=" * 70)
        print("FATAL ERROR")
        print("=" * 70)

        print(error)


    finally:

        if conn is not None:

            conn.close()

            print(
                "\nPostgreSQL connection closed."
            )


# ============================================================
# EXECUTE SCRIPT
# ============================================================

if __name__ == "__main__":

    main()