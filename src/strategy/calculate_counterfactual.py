import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

YEAR = 2025

COUNTERFACTUAL_LAPS = 5

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

DB_NAME = "f1_strategy_grader_ML"
DB_USER = "postgres"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = "localhost"
DB_PORT = "5433"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR /
    "2025_counterfactual_results.csv"
)

CONDITIONS_FILE = (
    OUTPUT_DIR /
    "2025_race_conditions.csv"
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
# LOAD RACE CONDITIONS
# ============================================================

def load_race_conditions():

    if not CONDITIONS_FILE.exists():

        raise FileNotFoundError(
            f"Race conditions file not found:\n"
            f"{CONDITIONS_FILE}"
        )

    conditions = pd.read_csv(
        CONDITIONS_FILE
    )

    required = [
        "race_round",
        "lap",
        "condition"
    ]

    missing = [
        c for c in required
        if c not in conditions.columns
    ]

    if missing:

        raise ValueError(
            f"Race conditions file is missing columns: "
            f"{missing}"
        )

    conditions["race_round"] = pd.to_numeric(
        conditions["race_round"],
        errors="coerce"
    )

    conditions["lap"] = pd.to_numeric(
        conditions["lap"],
        errors="coerce"
    )

    conditions["condition"] = (
        conditions["condition"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    neutral_conditions = {
        "VSC",
        "SAFETY_CAR",
        "RED_FLAG"
    }

    conditions = conditions[
        conditions["condition"].isin(
            neutral_conditions
        )
    ].copy()

    print(
        f"Race-condition records loaded: "
        f"{len(conditions):,}"
    )

    print("\nNeutralized-condition records:")

    print(
        conditions["condition"]
        .value_counts()
        .to_string()
    )

    return conditions


# ============================================================
# BUILD NEUTRALIZED LAP MAP
# ============================================================

def build_neutralized_laps(
    conditions
):

    neutralized = {}

    for race_round, group in conditions.groupby(
        "race_round"
    ):

        race_laps = set()

        active_condition = None
        start_lap = None

        events = (
            group
            .sort_values("lap")
            .drop_duplicates(
                subset=["lap"],
                keep="first"
            )
        )

        for _, row in events.iterrows():

            lap = row["lap"]

            condition = row["condition"]

            if pd.isna(lap):
                continue

            lap = int(lap)

            # ------------------------------------------------
            # START
            # ------------------------------------------------

            if condition in {
                "VSC",
                "SAFETY_CAR"
            }:

                if active_condition is None:

                    active_condition = condition
                    start_lap = lap

            # ------------------------------------------------
            # RED FLAG
            # ------------------------------------------------

            elif condition == "RED_FLAG":

                if active_condition is not None:

                    for x in range(
                        start_lap,
                        lap + 1
                    ):
                        race_laps.add(x)

                    active_condition = None
                    start_lap = None

        # ----------------------------------------------------
        # OPEN NEUTRALIZATION
        # ----------------------------------------------------

        if (
            active_condition is not None
            and
            start_lap is not None
        ):

            valid_laps = events["lap"].dropna()

            if not valid_laps.empty:

                max_lap = int(
                    valid_laps.max()
                )

                for x in range(
                    start_lap,
                    max_lap + 1
                ):
                    race_laps.add(x)

        neutralized[int(race_round)] = race_laps

    return neutralized


# ============================================================
# CLEAN LAP DATA
# ============================================================

def clean_laps(laps):

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

    laps = laps.dropna(
        subset=[
            "race_id",
            "driver_id",
            "lap_number",
            "lap_time_seconds"
        ]
    ).copy()

    # Remove impossible / obviously abnormal lap times.
    laps = laps[
        (laps["lap_time_seconds"] >= 50)
        &
        (laps["lap_time_seconds"] <= 160)
    ].copy()

    return laps


# ============================================================
# REMOVE NEUTRALIZED LAPS
# ============================================================

def remove_neutralized_laps(
    driver_laps,
    race_round,
    neutralized_laps
):

    blocked = neutralized_laps.get(
        int(race_round),
        set()
    )

    if not blocked:

        return driver_laps.copy()

    result = driver_laps[
        ~driver_laps["lap_number"]
        .astype(int)
        .isin(blocked)
    ].copy()

    return result


# ============================================================
# PRE-PIT BASELINE
# ============================================================

def get_baseline_pace(
    driver_laps,
    pit_lap
):

    pre = driver_laps[
        driver_laps["lap_number"] < pit_lap
    ].sort_values(
        "lap_number"
    )

    if pre.empty:

        return None

    pre = pre.tail(5)

    values = pre[
        "lap_time_seconds"
    ].dropna()

    if len(values) < 2:

        return None

    return float(
        values.median()
    )


# ============================================================
# TYRE DEGRADATION
# ============================================================

def estimate_degradation(
    driver_laps,
    pit_lap
):

    # --------------------------------------------------------
    # PRE-PIT LAPS
    # --------------------------------------------------------

    pre = driver_laps[
        driver_laps["lap_number"] < pit_lap
    ].copy()

    if pre.empty:

        return 0.0

    pre = pre.sort_values(
        "lap_number"
    )

    model_base = pre.dropna(
        subset=[
            "tyre_life",
            "lap_time_seconds"
        ]
    ).copy()

    if model_base.empty:

        return 0.0

    # --------------------------------------------------------
    # REMOVE EXTREME LAP-TIME OUTLIERS
    # --------------------------------------------------------

    q1 = model_base[
        "lap_time_seconds"
    ].quantile(0.25)

    q3 = model_base[
        "lap_time_seconds"
    ].quantile(0.75)

    iqr = q3 - q1

    if iqr > 0:

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        model_base = model_base[
            (model_base["lap_time_seconds"] >= lower)
            &
            (model_base["lap_time_seconds"] <= upper)
        ].copy()

    if len(model_base) < 4:

        return 0.0

    # --------------------------------------------------------
    # 1. CURRENT STINT MODEL
    # --------------------------------------------------------

    if "stint" in model_base.columns:

        current_stint = (
            model_base
            .sort_values("lap_number")
            .iloc[-1]["stint"]
        )

        if pd.notna(current_stint):

            stint_data = model_base[
                model_base["stint"] == current_stint
            ].copy()

            if (
                len(stint_data) >= 4
                and
                stint_data["tyre_life"].nunique() >= 3
            ):

                x = stint_data[
                    "tyre_life"
                ].to_numpy(dtype=float)

                y = stint_data[
                    "lap_time_seconds"
                ].to_numpy(dtype=float)

                try:

                    slope = np.polyfit(
                        x,
                        y,
                        1
                    )[0]

                    slope = max(
                        0.0,
                        float(slope)
                    )

                    return min(
                        slope,
                        0.15
                    )

                except Exception:

                    pass

    # --------------------------------------------------------
    # 2. DRIVER + COMPOUND MODEL
    # --------------------------------------------------------

    if "compound" in model_base.columns:

        latest_compound = (
            model_base
            .sort_values("lap_number")
            .iloc[-1]["compound"]
        )

        if pd.notna(latest_compound):

            compound_data = model_base[
                model_base["compound"]
                == latest_compound
            ].copy()

            if (
                len(compound_data) >= 6
                and
                compound_data["tyre_life"].nunique() >= 3
            ):

                x = compound_data[
                    "tyre_life"
                ].to_numpy(dtype=float)

                y = compound_data[
                    "lap_time_seconds"
                ].to_numpy(dtype=float)

                try:

                    slope = np.polyfit(
                        x,
                        y,
                        1
                    )[0]

                    slope = max(
                        0.0,
                        float(slope)
                    )

                    return min(
                        slope,
                        0.15
                    )

                except Exception:

                    pass

    # --------------------------------------------------------
    # 3. RECENT-LAP DEGRADATION
    # --------------------------------------------------------

    recent = model_base.tail(8).copy()

    if (
        len(recent) >= 4
        and
        recent["tyre_life"].nunique() >= 3
    ):

        x = recent[
            "tyre_life"
        ].to_numpy(dtype=float)

        y = recent[
            "lap_time_seconds"
        ].to_numpy(dtype=float)

        try:

            slope = np.polyfit(
                x,
                y,
                1
            )[0]

            slope = max(
                0.0,
                float(slope)
            )

            return min(
                slope,
                0.15
            )

        except Exception:

            pass

    return 0.0


# ============================================================
# ACTUAL POST-PIT PACE
# ============================================================

def get_actual_post_pit_laps(
    driver_laps,
    pit_lap
):

    post = driver_laps[
        (driver_laps["lap_number"] > pit_lap)
        &
        (
            driver_laps["lap_number"]
            <= pit_lap + COUNTERFACTUAL_LAPS
        )
    ].sort_values(
        "lap_number"
    )

    post = post.dropna(
        subset=["lap_time_seconds"]
    )

    return post.head(
        COUNTERFACTUAL_LAPS
    )


# ============================================================
# COUNTERFACTUAL
# ============================================================

def calculate_counterfactual(
    driver_laps,
    clean_driver_laps,
    pit_lap,
    tyre_age_before,
    race_round,
    neutralized_laps
):

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = get_baseline_pace(
        clean_driver_laps,
        pit_lap
    )

    if baseline is None:

        return {
            "skip_reason": "NO_BASELINE"
        }

    # --------------------------------------------------------
    # ACTUAL POST-PIT LAPS
    # --------------------------------------------------------

    actual_laps = get_actual_post_pit_laps(
        driver_laps,
        pit_lap
    )

    if len(actual_laps) < 3:

        return {
            "skip_reason":
                "INSUFFICIENT_POST_PIT_LAPS"
        }

    # --------------------------------------------------------
    # REMOVE NEUTRALIZED LAPS
    # --------------------------------------------------------

    actual_laps_clean = (
        remove_neutralized_laps(
            actual_laps,
            race_round,
            neutralized_laps
        )
    )

    if len(actual_laps_clean) < 3:

        return {
            "skip_reason":
                "INSUFFICIENT_CLEAN_LAPS"
        }

    actual_laps_clean = (
        actual_laps_clean
        .sort_values("lap_number")
        .head(COUNTERFACTUAL_LAPS)
    )

    # --------------------------------------------------------
    # DEGRADATION
    # --------------------------------------------------------

    degradation = estimate_degradation(
        clean_driver_laps,
        pit_lap
    )

    # --------------------------------------------------------
    # STARTING TYRE AGE
    # --------------------------------------------------------

    if pd.isna(tyre_age_before):

        starting_age = 1.0

    else:

        starting_age = max(
            1.0,
            float(tyre_age_before)
        )

    # --------------------------------------------------------
    # HYPOTHETICAL STAY-OUT
    # --------------------------------------------------------

    hypothetical = []

    for lap_index in range(
        1,
        len(actual_laps_clean) + 1
    ):

        hypothetical_age = (
            starting_age
            + lap_index
        )

        estimated_time = (
            baseline
            +
            degradation
            * hypothetical_age
        )

        hypothetical.append(
            estimated_time
        )

    hypothetical = np.array(
        hypothetical,
        dtype=float
    )

    # --------------------------------------------------------
    # ACTUAL TIME
    # --------------------------------------------------------

    actual = (
        actual_laps_clean[
            "lap_time_seconds"
        ]
        .to_numpy(
            dtype=float
        )
    )

    stay_out_total = float(
        hypothetical.sum()
    )

    actual_total = float(
        actual.sum()
    )

    # Positive:
    # staying out was estimated slower,
    # therefore pit was estimated beneficial.
    #
    # Negative:
    # staying out was estimated faster.

    difference = (
        stay_out_total
        -
        actual_total
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if (
        len(actual_laps_clean) >= 4
        and
        degradation > 0
    ):

        confidence = "HIGH"

    elif len(actual_laps_clean) >= 3:

        confidence = "MEDIUM"

    else:

        confidence = "LOW"

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "laps_compared":
            len(actual),

        "baseline_pre_pit_pace":
            baseline,

        "estimated_degradation_per_lap":
            degradation,

        "actual_post_pit_time":
            actual_total,

        "estimated_stay_out_time":
            stay_out_total,

        "counterfactual_difference":
            difference,

        "actual_average_lap":
            actual_total / len(actual),

        "estimated_stay_out_average_lap":
            stay_out_total / len(actual),

        "neutralized_laps_removed":
            len(actual_laps)
            -
            len(actual_laps_clean),

        "counterfactual_confidence":
            confidence
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY GRADER")
    print("2025 COUNTERFACTUAL ANALYSIS")
    print("=" * 70)

    conn = None

    try:

        # ----------------------------------------------------
        # CONDITIONS
        # ----------------------------------------------------

        print(
            "\nLoading race conditions..."
        )

        conditions = load_race_conditions()

        neutralized_laps = (
            build_neutralized_laps(
                conditions
            )
        )

        print(
            f"Races with condition data: "
            f"{len(neutralized_laps)}"
        )

        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        print(
            "\nConnecting to PostgreSQL..."
        )

        conn = get_connection()

        print(
            "PostgreSQL connection successful."
        )

        # ----------------------------------------------------
        # LOAD RACES
        # ----------------------------------------------------

        print(
            "\nLoading race information..."
        )

        races = pd.read_sql(
            """
            SELECT
                race_id,
                season,
                round,
                race_name
            FROM races
            WHERE season = %s
            ORDER BY round;
            """,
            conn,
            params=(YEAR,)
        )

        print(
            f"Races loaded: {len(races)}"
        )

        # ----------------------------------------------------
        # LOAD LAPS
        # ----------------------------------------------------

        print(
            "\nLoading laps..."
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
            f"Laps loaded: {len(laps):,}"
        )

        # ----------------------------------------------------
        # LOAD PIT STOPS
        # ----------------------------------------------------

        print(
            "\nLoading pit stops..."
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
            f"Pit stops loaded: {len(pits):,}"
        )

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        laps = clean_laps(
            laps
        )

        print(
            f"Usable laps: {len(laps):,}"
        )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        records = []

        skipped = 0

        skip_reasons = {}

        print(
            "\nCalculating counterfactuals..."
        )

        for index, pit in pits.iterrows():

            race_id = int(
                pit["race_id"]
            )

            driver_id = int(
                pit["driver_id"]
            )

            pit_lap = int(
                pit["pit_lap"]
            )

            race_row = races[
                races["race_id"]
                == race_id
            ]

            if race_row.empty:

                skipped += 1

                reason = "RACE_NOT_FOUND"

                skip_reasons[reason] = (
                    skip_reasons.get(reason, 0)
                    + 1
                )

                continue

            race_round = int(
                race_row.iloc[0]["round"]
            )

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

                skipped += 1

                reason = "NO_DRIVER_LAPS"

                skip_reasons[reason] = (
                    skip_reasons.get(reason, 0)
                    + 1
                )

                continue

            # ------------------------------------------------
            # REMOVE NEUTRALIZED LAPS BEFORE
            # BASELINE / DEGRADATION
            # ------------------------------------------------

            clean_driver_laps = (
                remove_neutralized_laps(
                    driver_laps,
                    race_round,
                    neutralized_laps
                )
            )

            result = calculate_counterfactual(
                driver_laps,
                clean_driver_laps,
                pit_lap,
                pit["tyre_age_before"],
                race_round,
                neutralized_laps
            )

            # ------------------------------------------------
            # SKIP INVALID RESULT
            # ------------------------------------------------

            if result is None:

                skipped += 1

                reason = "UNKNOWN"

                skip_reasons[reason] = (
                    skip_reasons.get(reason, 0)
                    + 1
                )

                continue

            if (
                isinstance(result, dict)
                and
                "skip_reason" in result
            ):

                skipped += 1

                reason = result[
                    "skip_reason"
                ]

                skip_reasons[reason] = (
                    skip_reasons.get(reason, 0)
                    + 1
                )

                continue

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            records.append({

                "race_id":
                    race_id,

                "race_round":
                    race_round,

                "driver_id":
                    driver_id,

                "pit_lap":
                    pit_lap,

                "tyre_before":
                    pit["tyre_before"],

                "tyre_after":
                    pit["tyre_after"],

                "tyre_age_before":
                    pit["tyre_age_before"],

                "stint_before":
                    pit["stint_before"],

                "stint_after":
                    pit["stint_after"],

                **result

            })

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                (index + 1) % 100 == 0
                or
                index + 1 == len(pits)
            ):

                print(
                    f"Processed "
                    f"{index + 1} / "
                    f"{len(pits)}"
                )

        df = pd.DataFrame(
            records
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "COUNTERFACTUAL VALIDATION"
        )

        print(
            "=" * 70
        )

        print(
            f"Pit stops available       : "
            f"{len(pits)}"
        )

        print(
            f"Counterfactuals generated : "
            f"{len(df)}"
        )

        coverage = (
            len(df)
            /
            len(pits)
            *
            100
        )

        print(
            f"Coverage                   : "
            f"{coverage:.1f}%"
        )

        print(
            f"Skipped                   : "
            f"{skipped}"
        )

        # ----------------------------------------------------
        # SKIP REASONS
        # ----------------------------------------------------

        if skip_reasons:

            print(
                "\nSkip reasons:"
            )

            for reason, count in sorted(
                skip_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            ):

                print(
                    f"{reason:<40}: {count}"
                )

        if df.empty:

            raise RuntimeError(
                "ZERO counterfactual records generated."
            )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        print(
            f"\nAverage difference : "
            f"{df['counterfactual_difference'].mean():.3f} sec"
        )

        print(
            f"Median difference  : "
            f"{df['counterfactual_difference'].median():.3f} sec"
        )

        print(
            f"Min difference     : "
            f"{df['counterfactual_difference'].min():.3f} sec"
        )

        print(
            f"Max difference     : "
            f"{df['counterfactual_difference'].max():.3f} sec"
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        print(
            "\nConfidence distribution:"
        )

        print(
            df[
                "counterfactual_confidence"
            ]
            .value_counts()
            .to_string()
        )

        # ----------------------------------------------------
        # NEUTRALIZED LAPS
        # ----------------------------------------------------

        print(
            "\nNeutralized laps removed:"
        )

        print(
            df[
                "neutralized_laps_removed"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        # ----------------------------------------------------
        # DEGRADATION
        # ----------------------------------------------------

        print(
            "\nDegradation validation:"
        )

        zero_degradation = (
            df[
                "estimated_degradation_per_lap"
            ]
            .eq(0)
            .sum()
        )

        positive_degradation = (
            df[
                "estimated_degradation_per_lap"
            ]
            .gt(0)
            .sum()
        )

        print(
            f"Zero degradation       : "
            f"{zero_degradation}"
        )

        print(
            f"Positive degradation   : "
            f"{positive_degradation}"
        )

        print(
            f"Zero degradation %     : "
            f"{zero_degradation / len(df) * 100:.1f}%"
        )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        print(
            "\nInterpretation:"
        )

        print(
            "Positive difference = pit was estimated faster."
        )

        print(
            "Negative difference = staying out was estimated faster."
        )

        # ----------------------------------------------------
        # SAMPLE RESULTS
        # ----------------------------------------------------

        print(
            "\nSample results:"
        )

        sample_columns = [

            "race_round",

            "race_id",

            "driver_id",

            "pit_lap",

            "tyre_before",

            "tyre_age_before",

            "baseline_pre_pit_pace",

            "estimated_degradation_per_lap",

            "actual_post_pit_time",

            "estimated_stay_out_time",

            "counterfactual_difference",

            "neutralized_laps_removed",

            "counterfactual_confidence"

        ]

        sample_columns = [
            column
            for column in sample_columns
            if column in df.columns
        ]

        print(
            df[
                sample_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            "\nSaved:"
        )

        print(
            OUTPUT_FILE
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "COUNTERFACTUAL ENGINE COMPLETE"
        )

        print(
            "=" * 70
        )

    finally:

        if conn is not None:

            conn.close()

            print(
                "\nPostgreSQL connection closed."
            )


if __name__ == "__main__":

    main()