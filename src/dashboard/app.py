import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="F1 Strategy Grader",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DB_NAME = "f1_strategy_grader_ML"
DB_USER = "postgres"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = "localhost"
DB_PORT = "5433"

STRATEGY_FILE = BASE_DIR / "data" / "processed" / "2025_strategy_grades.csv"

# ============================================================
# DATABASE
# ============================================================

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


@st.cache_data(ttl=300)
def load_driver_lookup():
    conn = get_connection()

    # Preferred schema: drivers -> teams
    try:
        return pd.read_sql(
            """
            SELECT
                d.driver_id,
                d.driver_code,
                d.driver_name,
                t.team_name
            FROM drivers d
            LEFT JOIN teams t ON d.team_id = t.team_id
            ORDER BY d.driver_name;
            """,
            conn,
        )
    except Exception:
        # Fallback for schema where team is stored directly in drivers
        return pd.read_sql(
            """
            SELECT
                driver_id,
                driver_code,
                driver_name,
                team AS team_name
            FROM drivers
            ORDER BY driver_name;
            """,
            conn,
        )


@st.cache_data(ttl=300)
def load_race_lookup():
    return pd.read_sql(
        """
        SELECT race_id, season, round, race_name
        FROM races
        WHERE season = 2025
        ORDER BY round;
        """,
        get_connection(),
    )


@st.cache_data(ttl=300)
def load_strategy():
    if not STRATEGY_FILE.exists():
        raise FileNotFoundError(
            f"Strategy grades file not found:\n{STRATEGY_FILE}"
        )

    df = pd.read_csv(STRATEGY_FILE)

    required = [
        "race_id",
        "driver_id",
        "pit_lap",
        "counterfactual_difference",
        "strategy_assessment",
        "strategy_score",
        "confidence_adjusted_score",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_columns = [
        "race_id",
        "driver_id",
        "pit_lap",
        "tyre_age_before",
        "counterfactual_difference",
        "counterfactual_difference_per_lap",
        "strategy_score",
        "confidence_adjusted_score",
        "laps_compared",
        "baseline_pre_pit_pace",
        "estimated_degradation_per_lap",
        "actual_post_pit_time",
        "estimated_stay_out_time",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# LOAD + ENRICH
# ============================================================

try:
    strategy = load_strategy()
    drivers = load_driver_lookup()
    races = load_race_lookup()

    strategy = strategy.merge(
        drivers[["driver_id", "driver_code", "driver_name", "team_name"]],
        on="driver_id",
        how="left",
    )

    strategy = strategy.merge(
        races[["race_id", "round", "race_name"]],
        on="race_id",
        how="left",
    )

except Exception as e:
    st.error("Dashboard data could not be loaded.")
    st.code(str(e))
    st.stop()

strategy["driver_code"] = strategy["driver_code"].fillna(
    strategy["driver_id"].map(
        lambda x: f"Driver {int(x)}" if pd.notna(x) else "Unknown"
    )
)
strategy["driver_name"] = strategy["driver_name"].fillna(strategy["driver_code"])
strategy["team_name"] = strategy["team_name"].fillna("Unknown Team")
strategy["race_name"] = strategy["race_name"].fillna("Unknown Race")

# ============================================================
# HEADER
# ============================================================

st.title("🏎️ 2025 F1 Strategy Grader")
st.markdown(
    "Evaluate Formula 1 pit-stop decisions using counterfactual "
    "stay-out analysis, tyre context and confidence-adjusted scoring."
)
st.caption(
    "FastF1 → PostgreSQL → Counterfactual Analysis → Strategy Grading → Streamlit"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Strategy Filters")

driver_search = st.sidebar.text_input(
    "Search Driver",
    placeholder="e.g. Norris, Verstappen, Leclerc",
)

team_search = st.sidebar.text_input(
    "Search Team",
    placeholder="e.g. McLaren, Ferrari",
)

driver_options = ["All Drivers"] + sorted(strategy["driver_name"].unique())
team_options = ["All Teams"] + sorted(strategy["team_name"].unique())
race_options = ["All Races"] + sorted(strategy["race_name"].unique())

assessment_options = [
    "All Assessments",
    "EXCELLENT_PIT",
    "VERY_GOOD_PIT",
    "GOOD_PIT",
    "NEUTRAL",
    "SLIGHTLY_BETTER_STAY_OUT",
    "POOR_PIT",
    "VERY_POOR_PIT",
]

selected_driver = st.sidebar.selectbox("Driver", driver_options)
selected_team = st.sidebar.selectbox("Team", team_options)
selected_race = st.sidebar.selectbox("Race", race_options)
selected_assessment = st.sidebar.selectbox(
    "Strategy Assessment",
    assessment_options,
)

score_range = st.sidebar.slider(
    "Strategy Score",
    0,
    100,
    (0, 100),
)

filtered = strategy.copy()

if driver_search.strip():
    q = driver_search.strip().lower()
    filtered = filtered[
        filtered["driver_name"].str.lower().str.contains(q, na=False)
        | filtered["driver_code"].str.lower().str.contains(q, na=False)
    ]

if team_search.strip():
    q = team_search.strip().lower()
    filtered = filtered[
        filtered["team_name"].str.lower().str.contains(q, na=False)
    ]

if selected_driver != "All Drivers":
    filtered = filtered[filtered["driver_name"] == selected_driver]

if selected_team != "All Teams":
    filtered = filtered[filtered["team_name"] == selected_team]

if selected_race != "All Races":
    filtered = filtered[filtered["race_name"] == selected_race]

if selected_assessment != "All Assessments":
    filtered = filtered[
        filtered["strategy_assessment"] == selected_assessment
    ]

filtered = filtered[
    filtered["strategy_score"].between(
        score_range[0], score_range[1]
    )
]

# Do not show fake 0/nan metrics.
if filtered.empty:
    st.warning(
        "No strategy decisions match the current filters. "
        "Clear the search boxes or choose an 'All' option."
    )
    st.stop()

# ============================================================
# KPIs
# ============================================================

decisions = len(filtered)
pit_faster = int((filtered["counterfactual_difference"] > 0).sum())
stay_out_faster = int((filtered["counterfactual_difference"] < 0).sum())
neutral = int((filtered["counterfactual_difference"] == 0).sum())

avg_score = filtered["strategy_score"].mean()
avg_adjusted = filtered["confidence_adjusted_score"].mean()
avg_difference = filtered["counterfactual_difference"].mean()
# ============================================================
# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

st.markdown("## 📊 Performance Overview")

# Use the already-filtered strategy dataset
evaluated = len(filtered)

pit_faster = int(
    (filtered["counterfactual_difference"] > 0).sum()
)

stay_out_faster = int(
    (filtered["counterfactual_difference"] < 0).sum()
)

avg_score = filtered["confidence_adjusted_score"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Evaluated Pit Stops",
        f"{evaluated:,}"
    )

with col2:
    st.metric(
        "Avg Adjusted Score",
        f"{avg_score:.1f}/100"
    )

with col3:
    st.metric(
        "Pit Faster",
        f"{pit_faster:,}"
    )

with col4:
    st.metric(
        "Stay-Out Faster",
        f"{stay_out_faster:,}"
    )

st.divider()

# ============================================================
# DECISION CHARTS
# ============================================================

st.divider()
st.subheader("⚖️ Counterfactual Strategy Outcomes")

left, right = st.columns(2)

with left:
    decision_df = pd.DataFrame(
        {
            "Decision": [
                "Pit Estimated Faster",
                "Stay-Out Estimated Faster",
                "Approximately Equal",
            ],
            "Count": [pit_faster, stay_out_faster, neutral],
        }
    )

    fig = px.bar(
        decision_df,
        x="Decision",
        y="Count",
        text="Count",
        title="Pit vs Stay-Out Outcome",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with right:
    order = [
        "EXCELLENT_PIT",
        "VERY_GOOD_PIT",
        "GOOD_PIT",
        "NEUTRAL",
        "SLIGHTLY_BETTER_STAY_OUT",
        "POOR_PIT",
        "VERY_POOR_PIT",
    ]

    assessment_df = (
        filtered["strategy_assessment"]
        .value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
    )
    assessment_df.columns = ["Assessment", "Decisions"]

    fig = px.bar(
        assessment_df,
        x="Assessment",
        y="Decisions",
        text="Decisions",
        title="Strategy Assessment Distribution",
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DRIVER RANKING
# ============================================================

st.divider()
st.subheader("🏆 Driver Strategy Ranking")

driver_summary = (
    filtered.groupby(
        ["driver_code", "driver_name", "team_name"],
        as_index=False,
    )
    .agg(
        Decisions=("strategy_score", "count"),
        AvgScore=("strategy_score", "mean"),
        AvgAdjusted=("confidence_adjusted_score", "mean"),
        AvgCounterfactual=("counterfactual_difference", "mean"),
        PitFasterRate=(
            "counterfactual_difference",
            lambda x: (x > 0).mean() * 100,
        ),
    )
    .sort_values("AvgAdjusted", ascending=False)
)

driver_summary = driver_summary.rename(
    columns={
        "driver_code": "Code",
        "driver_name": "Driver",
        "team_name": "Team",
        "AvgScore": "Avg Score",
        "AvgAdjusted": "Avg Adjusted Score",
        "AvgCounterfactual": "Avg Counterfactual (s)",
        "PitFasterRate": "Pit Faster %",
    }
)

for col in [
    "Avg Score",
    "Avg Adjusted Score",
    "Avg Counterfactual (s)",
    "Pit Faster %",
]:
    driver_summary[col] = driver_summary[col].round(2)

st.dataframe(
    driver_summary,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# DRIVER DETAIL
# ============================================================

st.divider()
st.subheader("👤 Driver Detail")

detail_driver = st.selectbox(
    "Select a driver",
    sorted(filtered["driver_name"].unique()),
)

driver_df = filtered[filtered["driver_name"] == detail_driver]

d1, d2, d3, d4 = st.columns(4)

d1.metric("Decisions", len(driver_df))
d2.metric("Average Score", f"{driver_df['strategy_score'].mean():.1f}")
d3.metric("Best Decision", f"{driver_df['strategy_score'].max():.0f}")
d4.metric(
    "Pit Faster Rate",
    f"{(driver_df['counterfactual_difference'] > 0).mean() * 100:.1f}%",
)

driver_race = (
    driver_df.groupby(
        ["round", "race_name"],
        as_index=False,
    )["strategy_score"]
    .mean()
    .sort_values("round")
)

fig = px.line(
    driver_race,
    x="round",
    y="strategy_score",
    markers=True,
    hover_data=["race_name"],
    labels={
        "round": "Race Round",
        "strategy_score": "Average Strategy Score",
    },
    title=f"{detail_driver} — Strategy Score Across 2025",
)
fig.update_yaxes(range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# BEST / WORST
# ============================================================

st.divider()
st.subheader("🎯 Best & Worst Strategy Decisions")

best_col, worst_col = st.columns(2)

decision_columns = [
    "race_name",
    "driver_name",
    "team_name",
    "pit_lap",
    "tyre_before",
    "tyre_age_before",
    "counterfactual_difference",
    "strategy_assessment",
    "strategy_score",
    "counterfactual_confidence",
]

available = [c for c in decision_columns if c in filtered.columns]

best_df = filtered.nlargest(10, "counterfactual_difference")[available].copy()
worst_df = filtered.nsmallest(10, "counterfactual_difference")[available].copy()

rename = {
    "race_name": "Race",
    "driver_name": "Driver",
    "team_name": "Team",
    "pit_lap": "Pit Lap",
    "tyre_before": "Tyre Before",
    "tyre_age_before": "Tyre Age",
    "counterfactual_difference": "Counterfactual Diff (s)",
    "strategy_assessment": "Assessment",
    "strategy_score": "Score",
    "counterfactual_confidence": "Confidence",
}

best_df = best_df.rename(columns=rename)
worst_df = worst_df.rename(columns=rename)

for df in [best_df, worst_df]:
    if "Counterfactual Diff (s)" in df.columns:
        df["Counterfactual Diff (s)"] = df["Counterfactual Diff (s)"].round(2)

with best_col:
    st.markdown("#### 🟢 Best Pit Decisions")
    st.dataframe(best_df, use_container_width=True, hide_index=True)

with worst_col:
    st.markdown("#### 🔴 Most Costly Pit Decisions")
    st.dataframe(worst_df, use_container_width=True, hide_index=True)

# ============================================================
# DECISION EXPLORER
# ============================================================

st.divider()
st.subheader("🔎 Strategy Decision Explorer")

explorer_columns = [
    "race_name",
    "driver_code",
    "driver_name",
    "team_name",
    "pit_lap",
    "tyre_before",
    "tyre_age_before",
    "counterfactual_difference",
    "counterfactual_difference_per_lap",
    "strategy_assessment",
    "strategy_score",
    "confidence_adjusted_score",
    "counterfactual_confidence",
]

explorer_columns = [c for c in explorer_columns if c in filtered.columns]
explorer = filtered[explorer_columns].copy()

explorer = explorer.rename(
    columns={
        "race_name": "Race",
        "driver_code": "Code",
        "driver_name": "Driver",
        "team_name": "Team",
        "pit_lap": "Pit Lap",
        "tyre_before": "Tyre Before",
        "tyre_age_before": "Tyre Age",
        "counterfactual_difference": "Counterfactual Diff (s)",
        "counterfactual_difference_per_lap": "Diff / Lap (s)",
        "strategy_assessment": "Assessment",
        "strategy_score": "Score",
        "confidence_adjusted_score": "Confidence-Adjusted Score",
        "counterfactual_confidence": "Confidence",
    }
)

for col in [
    "Tyre Age",
    "Counterfactual Diff (s)",
    "Diff / Lap (s)",
    "Score",
    "Confidence-Adjusted Score",
]:
    if col in explorer.columns:
        explorer[col] = explorer[col].round(2)

st.dataframe(
    explorer.sort_values("Score", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

with st.expander("📘 How the Strategy Grader Works"):
    st.markdown(
        """
### Counterfactual

**Counterfactual Difference = Estimated Stay-Out Time − Actual Post-Pit Time**

- Positive → the pit stop was estimated to be beneficial.
- Negative → staying out was estimated to be faster.

### Strategy grading

| Difference | Assessment | Score |
|---:|---|---:|
| ≥ +10 s | EXCELLENT_PIT | 100 |
| +5 to < +10 s | VERY_GOOD_PIT | 85 |
| +2 to < +5 s | GOOD_PIT | 70 |
| −2 to < +2 s | NEUTRAL | 50 |
| −5 to < −2 s | SLIGHTLY_BETTER_STAY_OUT | 35 |
| −10 to < −5 s | POOR_PIT | 20 |
| < −10 s | VERY_POOR_PIT | 0 |

The confidence-adjusted score discounts medium-confidence evaluations.

This is an analytical counterfactual model, not proof of a team's internal strategy quality.
"""
    )

st.caption(
    "F1 Strategy Grader · 2025 Season · FastF1 + Python + PostgreSQL + Streamlit"
)
