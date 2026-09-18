import os
from pathlib import Path

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

def python_value(value):
    """
    Convert NumPy/Pandas values into native Python values
    that psycopg2 can safely send to PostgreSQL.
    """

    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"

load_dotenv(BASE_DIR / ".env")


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_NAME = "f1_strategy_grader_ML"
DB_USER = "postgres"
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5433"


if not DB_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD was not found. "
        "Make sure your .env file contains DB_PASSWORD=..."
    )


# ============================================================
# FILES
# ============================================================

LAPS_FILE = RAW_DIR / "2025_all_race_laps.csv"
RESULTS_FILE = RAW_DIR / "2025_race_results.csv"
SCHEDULE_FILE = RAW_DIR / "2025_race_schedule.csv"


print("=" * 70)
print("F1 STRATEGY GRADER")
print("2025 DATABASE LOADER")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking raw data files...")

for file in [LAPS_FILE, RESULTS_FILE, SCHEDULE_FILE]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )

    print(f"FOUND: {file.name}")


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\nLoading CSV files...")

laps_df = pd.read_csv(LAPS_FILE)

results_df = pd.read_csv(RESULTS_FILE)

schedule_df = pd.read_csv(SCHEDULE_FILE)


print(
    f"Laps loaded     : {len(laps_df):,}"
)

print(
    f"Results loaded  : {len(results_df):,}"
)

print(
    f"Schedule loaded : {len(schedule_df):,}"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return None

    return value


def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


# ============================================================
# IDENTIFY IMPORTANT COLUMNS
# ============================================================

driver_col = find_column(
    laps_df,
    ["Driver"]
)

team_col = find_column(
    laps_df,
    ["Team"]
)

race_col = find_column(
    laps_df,
    ["RaceName"]
)

round_col = find_column(
    laps_df,
    ["Round"]
)

lap_number_col = find_column(
    laps_df,
    ["LapNumber"]
)

lap_time_col = find_column(
    laps_df,
    ["LapTimeSeconds"]
)

compound_col = find_column(
    laps_df,
    ["Compound"]
)

tyre_life_col = find_column(
    laps_df,
    ["TyreLife"]
)

fresh_tyre_col = find_column(
    laps_df,
    ["FreshTyre"]
)

stint_col = find_column(
    laps_df,
    ["Stint"]
)

position_col = find_column(
    laps_df,
    ["Position"]
)

pit_in_col = find_column(
    laps_df,
    ["PitIn"]
)

pit_out_col = find_column(
    laps_df,
    ["PitOut"]
)

track_status_col = find_column(
    laps_df,
    ["TrackStatus"]
)

accurate_col = find_column(
    laps_df,
    ["IsAccurate"]
)

deleted_col = find_column(
    laps_df,
    ["Deleted"]
)


required = {
    "Driver": driver_col,
    "RaceName": race_col,
    "Round": round_col,
    "LapNumber": lap_number_col,
}


for name, column in required.items():

    if column is None:

        raise RuntimeError(
            f"Required column '{name}' "
            f"was not found in the lap CSV."
        )


# ============================================================
# DRIVER INFORMATION
# ============================================================

print("\nPreparing driver information...")


# FastF1 results normally contains richer driver information.
result_driver_col = find_column(
    results_df,
    [
        "Abbreviation",
        "Driver",
        "DriverCode"
    ]
)

result_name_col = find_column(
    results_df,
    [
        "FullName",
        "BroadcastName"
    ]
)

result_team_col = find_column(
    results_df,
    [
        "TeamName",
        "Team"
    ]
)


driver_records = {}


# First use race results
if result_driver_col:

    for _, row in results_df.iterrows():

        code = clean_value(
            row[result_driver_col]
        )

        if not code:
            continue

        name = None
        team = None

        if result_name_col:
            name = clean_value(
                row[result_name_col]
            )

        if result_team_col:
            team = clean_value(
                row[result_team_col]
            )

        if code not in driver_records:

            driver_records[code] = {
                "name": name or code,
                "team": team or "Unknown"
            }


# Fill missing information from laps
for _, row in laps_df.iterrows():

    code = clean_value(
        row[driver_col]
    )

    if not code:
        continue

    team = None

    if team_col:
        team = clean_value(
            row[team_col]
        )

    if code not in driver_records:

        driver_records[code] = {
            "name": code,
            "team": team or "Unknown"
        }

    elif (
        driver_records[code]["team"] == "Unknown"
        and team
    ):

        driver_records[code]["team"] = team


print(
    f"Drivers identified: {len(driver_records)}"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

print("\nConnecting to PostgreSQL...")

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

cursor = conn.cursor()

print("PostgreSQL connection successful.")


# ============================================================
# CLEAR DATA
# ============================================================

print("\nClearing existing data...")

cursor.execute(
    "DELETE FROM strategy_scores;"
)

cursor.execute(
    "DELETE FROM pit_stops;"
)

cursor.execute(
    "DELETE FROM laps;"
)

cursor.execute(
    "DELETE FROM drivers;"
)

cursor.execute(
    "DELETE FROM teams;"
)

cursor.execute(
    "DELETE FROM races;"
)

conn.commit()

print("Existing data cleared.")


# ============================================================
# INSERT TEAMS
# ============================================================

print("\nInserting teams...")

teams = sorted(
    {
        info["team"]
        for info in driver_records.values()
        if info["team"]
    }
)

team_values = [
    (team,)
    for team in teams
]

execute_values(
    cursor,
    """
    INSERT INTO teams (team_name)
    VALUES %s
    ON CONFLICT (team_name)
    DO NOTHING
    """,
    team_values
)

conn.commit()

print(
    f"Teams inserted: {len(teams)}"
)


# ============================================================
# GET TEAM IDS
# ============================================================

cursor.execute(
    """
    SELECT team_id, team_name
    FROM teams
    """
)

team_map = {
    name: team_id
    for team_id, name in cursor.fetchall()
}


# ============================================================
# INSERT DRIVERS
# ============================================================

print("\nInserting drivers...")

driver_values = []

for code, info in sorted(
    driver_records.items()
):

    team_id = team_map.get(
        info["team"]
    )

    driver_values.append(
        (
            code,
            info["name"],
            team_id
        )
    )


execute_values(
    cursor,
    """
    INSERT INTO drivers
    (
        driver_code,
        driver_name,
        team_id
    )
    VALUES %s
    ON CONFLICT (driver_code)
    DO UPDATE SET
        driver_name = EXCLUDED.driver_name,
        team_id = EXCLUDED.team_id
    """,
    driver_values
)

conn.commit()

print(
    f"Drivers inserted: {len(driver_values)}"
)


# ============================================================
# GET DRIVER IDS
# ============================================================

cursor.execute(
    """
    SELECT
        driver_id,
        driver_code
    FROM drivers
    """
)

driver_map = {
    code: driver_id
    for driver_id, code in cursor.fetchall()
}


# ============================================================
# INSERT RACES
# ============================================================

print("\nInserting races...")


race_records = []


# Prefer schedule data
if not schedule_df.empty:

    for _, row in schedule_df.iterrows():

        season = clean_value(
            row.get("season")
        )

        round_number = clean_value(
            row.get("round")
        )

        race_name = clean_value(
            row.get("race_name")
        )

        country = clean_value(
            row.get("country")
        )

        event_date = clean_value(
            row.get("event_date")
        )

        if (
            season is None
            or round_number is None
            or race_name is None
        ):
            continue

        race_records.append(
            (
                int(season),
                int(round_number),
                race_name,
                country,
                event_date
            )
        )


# If schedule somehow lacks races, derive from laps
if not race_records:

    derived = (
        laps_df[
            [
                "Season",
                "Round",
                "RaceName"
            ]
        ]
        .drop_duplicates()
    )

    for _, row in derived.iterrows():

        race_records.append(
            (
                int(row["Season"]),
                int(row["Round"]),
                row["RaceName"],
                None,
                None
            )
        )


execute_values(
    cursor,
    """
    INSERT INTO races
    (
        season,
        round,
        race_name,
        country,
        event_date
    )
    VALUES %s
    ON CONFLICT (season, round)
    DO UPDATE SET
        race_name = EXCLUDED.race_name,
        country = EXCLUDED.country,
        event_date = EXCLUDED.event_date
    """,
    race_records
)

conn.commit()

print(
    f"Races inserted: {len(race_records)}"
)


# ============================================================
# GET RACE IDS
# ============================================================

cursor.execute(
    """
    SELECT
        race_id,
        season,
        round
    FROM races
    WHERE season = 2025
    """
)

race_map = {
    (season, round_number): race_id
    for race_id, season, round_number
    in cursor.fetchall()
}


# ============================================================
# INSERT LAPS
# ============================================================

print("\nPreparing lap records...")

lap_values = []


for _, row in laps_df.iterrows():

    driver_code = clean_value(
        row[driver_col]
    )

    season = clean_value(
        row.get("Season", YEAR)
    )

    round_number = clean_value(
        row[round_col]
    )

    if (
        not driver_code
        or season is None
        or round_number is None
    ):
        continue

    driver_id = driver_map.get(
        driver_code
    )

    race_id = race_map.get(
        (
            int(season),
            int(round_number)
        )
    )

    if driver_id is None or race_id is None:
        continue


    lap_number = clean_value(
        row[lap_number_col]
    )

    if lap_number is None:
        continue


    lap_time = (
        clean_value(
            row[lap_time_col]
        )
        if lap_time_col
        else None
    )

    compound = (
        clean_value(
            row[compound_col]
        )
        if compound_col
        else None
    )

    tyre_life = (
        clean_value(
            row[tyre_life_col]
        )
        if tyre_life_col
        else None
    )

    fresh_tyre = (
        clean_value(
            row[fresh_tyre_col]
        )
        if fresh_tyre_col
        else None
    )

    stint = (
        clean_value(
            row[stint_col]
        )
        if stint_col
        else None
    )

    position = (
        clean_value(
            row[position_col]
        )
        if position_col
        else None
    )

    pit_in = (
        bool(row[pit_in_col])
        if pit_in_col
        and not pd.isna(row[pit_in_col])
        else False
    )

    pit_out = (
        bool(row[pit_out_col])
        if pit_out_col
        and not pd.isna(row[pit_out_col])
        else False
    )

    track_status = (
        clean_value(
            row[track_status_col]
        )
        if track_status_col
        else None
    )

    is_accurate = (
        bool(row[accurate_col])
        if accurate_col
        and not pd.isna(row[accurate_col])
        else None
    )

    deleted = (
        bool(row[deleted_col])
        if deleted_col
        and not pd.isna(row[deleted_col])
        else None
    )


    lap_values.append(
        (
            race_id,
            driver_id,
            int(lap_number),
            lap_time,
            compound,
            tyre_life,
            fresh_tyre,
            stint,
            position,
            pit_in,
            pit_out,
            track_status,
            is_accurate,
            deleted
        )
    )


print(
    f"Prepared {len(lap_values):,} lap records."
)


# ============================================================
# INSERT LAPS IN BATCHES
# ============================================================

print("\nInserting laps...")

BATCH_SIZE = 5000

for start in range(
    0,
    len(lap_values),
    BATCH_SIZE
):

    batch = lap_values[
        start:start + BATCH_SIZE
    ]

    execute_values(
        cursor,
        """
        INSERT INTO laps
        (
            race_id,
            driver_id,
            lap_number,
            lap_time_seconds,
            compound,
            tyre_life,
            fresh_tyre,
            stint,
            position,
            pit_in,
            pit_out,
            track_status,
            is_accurate,
            deleted
        )
        VALUES %s
        """,
        batch
    )

    conn.commit()

    print(
        f"Inserted "
        f"{min(start + BATCH_SIZE, len(lap_values)):,}"
        f" / {len(lap_values):,}"
    )


# ============================================================
# CREATE PIT STOP DATA FROM LAP DATA
# ============================================================

print("\nCreating pit-stop records...")


pit_values = []


# Sort original data
pit_source = laps_df.copy()

sort_columns = [
    c
    for c in [
        race_col,
        driver_col,
        lap_number_col
    ]
    if c
]

pit_source = pit_source.sort_values(
    sort_columns
)


for (
    (race_name, driver_code),
    group
) in pit_source.groupby(
    [race_col, driver_col]
):

    group = group.sort_values(
        lap_number_col
    ).reset_index(drop=True)


    for i in range(
        len(group)
    ):

        row = group.iloc[i]


        # A pit-in marks the lap before the new stint
        if (
            pit_in_col
            and bool(
                row[pit_in_col]
            )
        ):

            current_stint = (
                clean_value(
                    row[stint_col]
                )
                if stint_col
                else None
            )

            current_compound = (
                clean_value(
                    row[compound_col]
                )
                if compound_col
                else None
            )

            tyre_age_before = (
                clean_value(
                    row[tyre_life_col]
                )
                if tyre_life_col
                else None
            )

            pit_lap = clean_value(
                row[lap_number_col]
            )


            # Look ahead for the next available lap
            next_row = None

            for j in range(
                i + 1,
                len(group)
            ):

                candidate = group.iloc[j]

                if (
                    candidate[lap_number_col]
                    > row[lap_number_col]
                ):

                    next_row = candidate
                    break


            tyre_after = None
            stint_after = None

            if next_row is not None:

                if compound_col:

                    tyre_after = clean_value(
                        next_row[compound_col]
                    )

                if stint_col:

                    stint_after = clean_value(
                        next_row[stint_col]
                    )


            season = YEAR

            round_number = clean_value(
                row[round_col]
            )

            race_id = race_map.get(
                (
                    int(season),
                    int(round_number)
                )
            )

            driver_id = driver_map.get(
                driver_code
            )


            if (
                race_id is not None
                and driver_id is not None
                and pit_lap is not None
            ):

                pit_values.append(
                    (
                        race_id,
                        driver_id,
                        int(pit_lap),
                        current_compound,
                        tyre_after,
                        tyre_age_before,
                        current_stint,
                        stint_after
                    )
                )


print(
    f"Prepared "
    f"{len(pit_values):,} pit-stop records."
)




print(
    f"Pit stops inserted: "
    f"{len(pit_values):,}"
)


# ============================================================
# FINAL DATABASE VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("DATABASE VERIFICATION")
print("=" * 70)


verification_queries = {

    "Teams":
        "SELECT COUNT(*) FROM teams",

    "Drivers":
        "SELECT COUNT(*) FROM drivers",

    "Races":
        "SELECT COUNT(*) FROM races WHERE season = 2025",

    "Laps":
        "SELECT COUNT(*) FROM laps",

    "Pit Stops":
        "SELECT COUNT(*) FROM pit_stops",

    "Strategy Scores":
        "SELECT COUNT(*) FROM strategy_scores"
}


for label, query in verification_queries.items():

    cursor.execute(query)

    count = cursor.fetchone()[0]

    print(
        f"{label:<20}: {count:,}"
    )

    print("Inserting pit stops...")

if pit_values:

    # Convert all NumPy values to native Python values
    clean_pit_values = []

    for row in pit_values:

        clean_row = []

        for value in row:

            if pd.isna(value):
                clean_row.append(None)

            elif isinstance(value, np.integer):
                clean_row.append(int(value))

            elif isinstance(value, np.floating):
                clean_row.append(float(value))

            elif isinstance(value, np.bool_):
                clean_row.append(bool(value))

            else:
                clean_row.append(value)

        clean_pit_values.append(
            tuple(clean_row)
        )

    execute_values(
        cursor,
        """
        INSERT INTO pit_stops (
            race_id,
            driver_id,
            pit_lap,
            tyre_before,
            tyre_after,
            tyre_age_before,
            stint_before,
            stint_after
        )
        VALUES %s
        """,
        clean_pit_values,
        page_size=500
    )

    print(
        f"Inserted {len(clean_pit_values):,} pit-stop records."
    )
    conn.commit()

    print("Pit-stop transaction committed.")

else:

    print("No pit-stop records to insert.")


# ============================================================
# TEAM VERIFICATION
# ============================================================

print("\nTeams in database:")

cursor.execute(
    """
    SELECT
        team_name
    FROM teams
    ORDER BY team_name
    """
)

for (team_name,) in cursor.fetchall():

    print(
        f"  - {team_name}"
    )


# ============================================================
# DRIVER VERIFICATION
# ============================================================

print("\nDrivers in database:")

cursor.execute(
    """
    SELECT
        d.driver_code,
        d.driver_name,
        t.team_name
    FROM drivers d
    LEFT JOIN teams t
        ON d.team_id = t.team_id
    ORDER BY t.team_name, d.driver_code
    """
)

for code, name, team in cursor.fetchall():

    print(
        f"  {code:<5} "
        f"{name:<30} "
        f"{team}"
    )


# ============================================================
# CLOSE
# ============================================================

cursor.close()

conn.close()


print("\n" + "=" * 70)
print("DATABASE LOAD COMPLETE")
print("=" * 70)