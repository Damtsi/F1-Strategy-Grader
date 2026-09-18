import os
from pathlib import Path

import fastf1
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

CACHE_DIR = BASE_DIR / "fastf1_cache"
RAW_DIR = BASE_DIR / "data" / "raw"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTF1 CONFIGURATION
# ============================================================

fastf1.Cache.enable_cache(str(CACHE_DIR))
fastf1.set_log_level("WARNING")


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("F1 STRATEGY GRADER")
print("2025 FULL GRID DATA COLLECTION")
print("=" * 75)


# ============================================================
# LOAD 2025 EVENT SCHEDULE
# ============================================================

print("\nLoading 2025 Formula 1 race schedule...")

schedule = fastf1.get_event_schedule(YEAR)

# Keep actual championship rounds only
schedule = schedule[
    schedule["RoundNumber"].notna()
].copy()

schedule = schedule[
    schedule["RoundNumber"] > 0
].copy()

schedule["RoundNumber"] = schedule["RoundNumber"].astype(int)

print(f"Race events found: {len(schedule)}")


# ============================================================
# STORAGE
# ============================================================

all_laps = []
all_results = []
all_events = []
all_pits = []


# ============================================================
# PROCESS EVERY RACE
# ============================================================

for _, event in schedule.iterrows():

    round_number = int(event["RoundNumber"])
    event_name = str(event["EventName"])

    print("\n" + "=" * 75)
    print(f"ROUND {round_number}: {event_name}")
    print("=" * 75)

    try:

        # ----------------------------------------------------
        # LOAD RACE SESSION
        # ----------------------------------------------------

        session = fastf1.get_session(
            YEAR,
            round_number,
            "R"
        )

        print("Loading race session...")

        session.load(
            laps=True,
            telemetry=False,
            weather=False,
            messages=False
        )

        # ----------------------------------------------------
        # EVENT INFORMATION
        # ----------------------------------------------------

        event_record = {
            "season": YEAR,
            "round": round_number,
            "race_name": event_name,
            "country": event.get("Country"),
            "event_date": event.get("EventDate")
        }

        all_events.append(event_record)

        # ----------------------------------------------------
        # LAP DATA
        # ----------------------------------------------------

        laps = session.laps.copy()

        if laps.empty:

            print("No lap data found.")

            continue

        # ----------------------------------------------------
        # ADD SEASON / RACE INFORMATION
        # ----------------------------------------------------

        laps["Season"] = YEAR
        laps["Round"] = round_number
        laps["RaceName"] = event_name

        # ----------------------------------------------------
        # KEEP RELEVANT COLUMNS
        # ----------------------------------------------------

        required_columns = [

            "Driver",
            "DriverNumber",
            "Team",

            "LapNumber",
            "LapTime",

            "Compound",
            "TyreLife",
            "FreshTyre",
            "Stint",

            "Position",

            "PitInTime",
            "PitOutTime",

            "TrackStatus",

            "IsAccurate",
            "Deleted"
        ]

        available_columns = [
            column
            for column in required_columns
            if column in laps.columns
        ]

        metadata_columns = [
            "Season",
            "Round",
            "RaceName"
        ]

        laps = laps[
            available_columns + metadata_columns
        ]

        # ----------------------------------------------------
        # LAP TIME IN SECONDS
        # ----------------------------------------------------

        if "LapTime" in laps.columns:

            laps["LapTimeSeconds"] = (
                laps["LapTime"]
                .dt.total_seconds()
            )

        else:

            laps["LapTimeSeconds"] = None

        # ----------------------------------------------------
        # PIT-IN FLAG
        # ----------------------------------------------------

        if "PitInTime" in laps.columns:

            laps["PitIn"] = (
                laps["PitInTime"]
                .notna()
            )

        else:

            laps["PitIn"] = False

        # ----------------------------------------------------
        # PIT-OUT FLAG
        # ----------------------------------------------------

        if "PitOutTime" in laps.columns:

            laps["PitOut"] = (
                laps["PitOutTime"]
                .notna()
            )

        else:

            laps["PitOut"] = False

        # ----------------------------------------------------
        # SAVE LAP DATA
        # ----------------------------------------------------

        all_laps.append(laps)

        # ----------------------------------------------------
        # CREATE PIT STOP DATA
        # ----------------------------------------------------

        pit_laps = laps[
            laps["PitIn"] == True
        ].copy()

        if not pit_laps.empty:

            pit_laps["PitLap"] = (
                pit_laps["LapNumber"]
            )

            pit_laps["TyreBefore"] = (
                pit_laps["Compound"]
            )

            pit_laps["TyreAgeBefore"] = (
                pit_laps["TyreLife"]
            )

            pit_laps["StintBefore"] = (
                pit_laps["Stint"]
            )

            pit_columns = [
                "Driver",
                "DriverNumber",
                "Team",
                "PitLap",
                "TyreBefore",
                "TyreAgeBefore",
                "StintBefore",
                "Season",
                "Round",
                "RaceName"
            ]

            pit_columns = [
                column
                for column in pit_columns
                if column in pit_laps.columns
            ]

            pit_laps = pit_laps[
                pit_columns
            ].copy()

            all_pits.append(
                pit_laps
            )

        # ----------------------------------------------------
        # RACE RESULTS
        # ----------------------------------------------------

        try:

            results = session.results.copy()

            if not results.empty:

                results["Season"] = YEAR
                results["Round"] = round_number
                results["RaceName"] = event_name

                all_results.append(
                    results
                )

        except Exception as result_error:

            print(
                "Could not load race results:",
                result_error
            )

        # ----------------------------------------------------
        # SAVE INDIVIDUAL RACE LAP FILE
        # ----------------------------------------------------

        safe_name = (
            event_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        output_file = (
            RAW_DIR /
            f"2025_R{round_number:02d}_{safe_name}_laps.csv"
        )

        laps.to_csv(
            output_file,
            index=False
        )

        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print(
            f"Lap records : {len(laps):,}"
        )

        print(
            f"Drivers     : {laps['Driver'].nunique()}"
        )

        print(
            f"Teams       : {laps['Team'].nunique()}"
        )

        print(
            f"Pit laps    : {len(pit_laps):,}"
        )

        print(
            f"Saved       : {output_file.name}"
        )

    except Exception as error:

        print(
            f"ERROR loading {event_name}"
        )

        print(
            str(error)
        )

        continue


# ============================================================
# COMBINE LAP DATA
# ============================================================

print("\n" + "=" * 75)
print("COMBINING LAP DATA")
print("=" * 75)

if all_laps:

    combined_laps = pd.concat(
        all_laps,
        ignore_index=True
    )

    combined_laps_file = (
        RAW_DIR /
        "2025_all_race_laps.csv"
    )

    combined_laps.to_csv(
        combined_laps_file,
        index=False
    )

    print(
        f"Total lap records: "
        f"{len(combined_laps):,}"
    )

    print(
        f"Total races: "
        f"{combined_laps['RaceName'].nunique()}"
    )

    print(
        f"Total drivers: "
        f"{combined_laps['Driver'].nunique()}"
    )

    print(
        f"Total teams: "
        f"{combined_laps['Team'].nunique()}"
    )

else:

    print("No lap data collected.")


# ============================================================
# COMBINE RACE RESULTS
# ============================================================

print("\n" + "=" * 75)
print("COMBINING RACE RESULTS")
print("=" * 75)

if all_results:

    combined_results = pd.concat(
        all_results,
        ignore_index=True
    )

    results_file = (
        RAW_DIR /
        "2025_race_results.csv"
    )

    combined_results.to_csv(
        results_file,
        index=False
    )

    print(
        f"Race result rows: "
        f"{len(combined_results):,}"
    )

else:

    print("No race results collected.")


# ============================================================
# COMBINE PIT STOP DATA
# ============================================================

print("\n" + "=" * 75)
print("COMBINING PIT STOP DATA")
print("=" * 75)

if all_pits:

    combined_pits = pd.concat(
        all_pits,
        ignore_index=True
    )

    pits_file = (
        RAW_DIR /
        "2025_pit_stops.csv"
    )

    combined_pits.to_csv(
        pits_file,
        index=False
    )

    print(
        f"Pit stop records: "
        f"{len(combined_pits):,}"
    )

else:

    print("No pit stop data collected.")


# ============================================================
# CREATE DRIVER MASTER DATA
# ============================================================

print("\n" + "=" * 75)
print("CREATING DRIVER MASTER DATA")
print("=" * 75)

if all_laps:

    driver_data = (
        combined_laps[
            [
                "Driver",
                "DriverNumber",
                "Team"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["Team", "Driver"]
        )
        .reset_index(drop=True)
    )

    driver_data["DriverCode"] = (
        driver_data["Driver"]
    )

    driver_data = driver_data[
        [
            "DriverCode",
            "DriverNumber",
            "Team"
        ]
    ]

    driver_file = (
        RAW_DIR /
        "2025_drivers.csv"
    )

    driver_data.to_csv(
        driver_file,
        index=False
    )

    print(
        f"Driver records: "
        f"{len(driver_data)}"
    )


# ============================================================
# CREATE TEAM MASTER DATA
# ============================================================

if all_laps:

    team_data = (
        combined_laps[
            ["Team"]
        ]
        .drop_duplicates()
        .sort_values("Team")
        .reset_index(drop=True)
    )

    team_data["TeamID"] = (
        range(
            1,
            len(team_data) + 1
        )
    )

    team_data = team_data[
        [
            "TeamID",
            "Team"
        ]
    ]

    team_file = (
        RAW_DIR /
        "2025_teams.csv"
    )

    team_data.to_csv(
        team_file,
        index=False
    )

    print(
        f"Team records: "
        f"{len(team_data)}"
    )


# ============================================================
# SAVE RACE SCHEDULE
# ============================================================

print("\n" + "=" * 75)
print("SAVING RACE SCHEDULE")
print("=" * 75)

events_df = pd.DataFrame(
    all_events
)

events_file = (
    RAW_DIR /
    "2025_race_schedule.csv"
)

events_df.to_csv(
    events_file,
    index=False
)

print(
    f"Schedule records: "
    f"{len(events_df)}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("2025 DATA COLLECTION COMPLETE")
print("=" * 75)

if all_laps:

    print()
    print(
        f"Races collected    : "
        f"{combined_laps['RaceName'].nunique()}"
    )

    print(
        f"Drivers collected  : "
        f"{combined_laps['Driver'].nunique()}"
    )

    print(
        f"Teams collected    : "
        f"{combined_laps['Team'].nunique()}"
    )

    print(
        f"Lap records        : "
        f"{len(combined_laps):,}"
    )

if all_pits:

    print(
        f"Pit-stop records   : "
        f"{len(combined_pits):,}"
    )

print()
print(
    "Raw data location:"
)

print(
    RAW_DIR
)

print()
print("=" * 75)