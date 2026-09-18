import os
from pathlib import Path

import pandas as pd
import fastf1
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "2025_race_conditions.csv"

load_dotenv(BASE_DIR / ".env")


# ============================================================
# FASTF1 CACHE
# ============================================================

CACHE_DIR = BASE_DIR / "data" / "fastf1_cache"
CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fastf1.Cache.enable_cache(
    str(CACHE_DIR)
)


# ============================================================
# RACE SCHEDULE
# ============================================================

def get_schedule():

    schedule = fastf1.get_event_schedule(
        YEAR
    )

    schedule = schedule[
        schedule["RoundNumber"] > 0
    ].copy()

    return schedule


# ============================================================
# CLASSIFY TRACK STATUS
# ============================================================

def classify_status(status):

    if pd.isna(status):
        return "UNKNOWN"

    status = str(status).strip()

    # FastF1 track-status encoding:
    #
    # 1 = All clear
    # 2 = Yellow
    # 3 = Unknown
    # 4 = Safety Car
    # 5 = Red Flag
    # 6 = Virtual Safety Car deployed
    # 7 = Virtual Safety Car ending
    #
    # Keep the original code as well.

    if status == "1":
        return "GREEN"

    elif status == "2":
        return "YELLOW"

    elif status == "3":
        return "UNKNOWN"

    elif status == "4":
        return "SAFETY_CAR"

    elif status == "5":
        return "RED_FLAG"

    elif status == "6":
        return "VSC"

    elif status == "7":
        return "VSC_ENDING"

    return "UNKNOWN"


# ============================================================
# COLLECT ONE RACE
# ============================================================

def collect_race(event):

    round_number = int(
        event["RoundNumber"]
    )

    event_name = str(
        event["EventName"]
    )

    print(
        f"\nLoading Round {round_number}: "
        f"{event_name}"
    )

    try:

        session = fastf1.get_session(
            YEAR,
            round_number,
            "R"
        )

        session.load(
            laps=False,
            telemetry=False,
            weather=False,
            messages=True
        )

    except Exception as error:

        print(
            f"ERROR loading {event_name}: {error}"
        )

        return []

    records = []

    # --------------------------------------------------------
    # FASTF1 RACE CONTROL MESSAGES
    # --------------------------------------------------------

    messages = getattr(
        session,
        "race_control_messages",
        None
    )

    if messages is None:

        print(
            "No race-control messages available."
        )

        return records

    if messages.empty:

        print(
            "Race-control message table is empty."
        )

        return records

    messages = messages.copy()

    print(
        f"Race-control messages: {len(messages)}"
    )

    # --------------------------------------------------------
    # SAVE RACE-CONTROL EVENTS
    # --------------------------------------------------------

    for _, row in messages.iterrows():

        records.append({

            "season":
                YEAR,

            "race_round":
                round_number,

            "race_name":
                event_name,

            "date":
                row.get("Time"),

            "category":
                row.get("Category"),

            "message":
                row.get("Message"),

            "flag":
                row.get("Flag"),

            "scope":
                row.get("Scope"),

            "sector":
                row.get("Sector"),

            "status":
                row.get("Status"),

            "lap":
                row.get("Lap")

        })

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("F1 STRATEGY GRADER")
    print("2025 RACE CONDITION COLLECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    schedule = get_schedule()

    print(
        f"\n2025 races found: {len(schedule)}"
    )

    all_records = []

    for _, event in schedule.iterrows():

        records = collect_race(
            event
        )

        all_records.extend(
            records
        )

    if not all_records:

        raise RuntimeError(
            "No race-control data was collected."
        )

    df = pd.DataFrame(
        all_records
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    if "lap" in df.columns:

        df["lap"] = pd.to_numeric(
            df["lap"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # DETECT VSC / SC EVENTS
    # --------------------------------------------------------

    text = (
        df[
            [
                "message",
                "category",
                "flag",
                "scope"
            ]
        ]
        .fillna("")
        .astype(str)
        .agg(
            " ".join,
            axis=1
        )
        .str.upper()
    )

    df["condition"] = "OTHER"

    df.loc[
        text.str.contains(
            "VIRTUAL SAFETY CAR|VSC"
        ),
        "condition"
    ] = "VSC"

    df.loc[
        text.str.contains(
            "SAFETY CAR"
        )
        &
        ~text.str.contains(
            "VIRTUAL SAFETY CAR|VSC"
        ),
        "condition"
    ] = "SAFETY_CAR"

    df.loc[
        text.str.contains(
            "RED FLAG"
        ),
        "condition"
    ] = "RED_FLAG"

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RACE CONDITION SUMMARY")
    print("=" * 70)

    print(
        df["condition"]
        .value_counts()
        .to_string()
    )

    print("\nVSC / SC EVENTS:")

    important = df[
        df["condition"].isin(
            [
                "VSC",
                "SAFETY_CAR",
                "RED_FLAG"
            ]
        )
    ]

    if important.empty:

        print(
            "No VSC/SC/Red Flag messages detected."
        )

    else:

        print(
            important[
                [
                    "race_round",
                    "race_name",
                    "lap",
                    "condition",
                    "message"
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "RACE CONDITION COLLECTION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Records collected : {len(df):,}"
    )

    print(
        f"Output file       : {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()