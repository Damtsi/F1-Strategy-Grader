# 🏎️ 2025 F1 Strategy Grader

A Formula 1 race-strategy analytics system that evaluates pit-stop decisions using **counterfactual analysis**.

The system estimates what may have happened if a driver had **stayed out instead of pitting**, compares that scenario with the observed post-pit performance, and converts the estimated strategic impact into a standardized strategy score.

The project combines **FastF1, Python, PostgreSQL, statistical modelling and Streamlit** to create an end-to-end F1 strategy analysis pipeline.

---

## 📌 Project Overview

Formula 1 strategy decisions are highly dependent on tyre degradation, race conditions, pit timing and track circumstances.

A simple analysis of lap times does not answer an important question:

> **Was the pit stop actually the better strategic decision?**

This project attempts to answer that question using a counterfactual approach.

For each usable pit stop, the system:

1. Identifies the driver's pre-pit race state.
2. Establishes a baseline pre-pit pace.
3. Estimates tyre degradation.
4. Constructs a hypothetical stay-out scenario.
5. Compares the estimated stay-out performance with the actual post-pit performance.
6. Calculates a counterfactual difference.
7. Classifies the pit-stop decision.
8. Applies a confidence adjustment.
9. Aggregates results by driver, race and tyre compound.
10. Presents the results through an interactive Streamlit dashboard.

---

# 🎯 Objectives

The project was designed to answer several strategy questions:

- Was a particular pit stop beneficial?
- Would staying out have been faster under the model assumptions?
- Which drivers made better pit-stop decisions?
- Which races contained stronger strategic decisions?
- How does strategy performance differ by tyre compound?
- How confident is the counterfactual estimate?
- Which individual pit stops were estimated to be the most beneficial or costly?

---

# 🧠 Core Concept: Counterfactual Strategy Analysis

The central idea is to compare two scenarios.

### Observed scenario

The driver actually pits and produces a sequence of laps after the pit stop.

### Counterfactual scenario

The model estimates how the driver could have performed if they had **continued without pitting**.

The core calculation is:

```text
Counterfactual Difference
=
Estimated Stay-Out Time
-
Actual Post-Pit Time

Interpretation
Difference	Interpretation
Positive	Pit stop estimated to be beneficial
Negative	Staying out estimated to be faster
Approximately zero	Similar performance

For example:

Estimated stay-out time = 500.0 s
Actual post-pit time    = 492.0 s

Difference = 500.0 - 492.0
           = +8.0 s

The model therefore estimates that the pit stop produced an approximately 8-second advantage over the modelled stay-out scenario.

🏁 Strategy Scoring

Counterfactual differences are converted into strategy assessments.

Counterfactual Difference	Assessment	Score
≥ +10 s	EXCELLENT_PIT	100
+5 to < +10 s	VERY_GOOD_PIT	85
+2 to < +5 s	GOOD_PIT	70
−2 to < +2 s	NEUTRAL	50
−5 to < −2 s	SLIGHTLY_BETTER_STAY_OUT	35
−10 to < −5 s	POOR_PIT	20
< −10 s	VERY_POOR_PIT	0

The system then applies a confidence adjustment to account for the reliability of the counterfactual estimate.

📊 Confidence Classification

Counterfactual results are assigned confidence levels.

The current pipeline contains:

HIGH confidence
MEDIUM confidence

The confidence-adjusted score reduces the impact of medium-confidence evaluations.

This prevents uncertain counterfactual estimates from being treated exactly the same as stronger evaluations.

🛞 Tyre Context

The analysis incorporates tyre information available around the pit stop.

Important variables include:

Tyre compound
Tyre age
Pre-pit pace
Estimated degradation
Post-pit performance
Number of laps compared

This allows the strategy evaluation to consider tyre state rather than treating every pit stop as identical.

📈 Current Results

The current counterfactual pipeline produced:

Pit stops available       : 841
Counterfactuals generated : 278
Coverage                  : 33.1%

The remaining pit stops were excluded when the available data did not satisfy the counterfactual model's requirements.

Skip reasons
NO_BASELINE
INSUFFICIENT_CLEAN_LAPS
INSUFFICIENT_POST_PIT_LAPS
NO_DRIVER_LAPS
Counterfactual Results

From the 278 evaluated decisions:

Pit estimated faster       : 35
Stay-out estimated faster  : 243
Approximately equal       : 0

Therefore:

Pit faster rate      : 12.59%
Stay-out faster rate : 87.41%

The average counterfactual difference was:

Average : -9.634 s
Median  : -10.677 s
Minimum : -43.620 s
Maximum : +25.830 s

These values describe the output of the current counterfactual model and should not be interpreted as direct measurements of actual strategic time loss.

📊 Strategy Grades

The current strategy grading pipeline produced:

EXCELLENT_PIT              5
VERY_GOOD_PIT              7
GOOD_PIT                  11
NEUTRAL                   21
SLIGHTLY_BETTER_STAY_OUT  16
POOR_PIT                  66
VERY_POOR_PIT            152
Score statistics
Average raw strategy score       : 17.25
Average confidence-adjusted score: 16.19
Maximum score                    : 100

The confidence-adjusted score is used as the primary performance metric when comparing strategy decisions.

👨‍🏎️ Driver Strategy Analysis

The system aggregates individual pit-stop evaluations by driver.

Driver-level metrics include:

Number of evaluated pit stops
Average strategy score
Average confidence-adjusted score
Average counterfactual difference
Pit-faster rate

Example output from the current dataset:

Driver ID	Decisions	Avg Score	Avg Adjusted Score	Avg Counterfactual	Pit Faster
105	15	29.00	28.67	-5.55 s	20.0%
101	15	24.00	23.17	-8.35 s	6.7%
86	13	23.08	21.44	-10.97 s	15.4%
102	15	23.67	21.08	-8.10 s	20.0%
85	12	21.25	20.83	-7.51 s	25.0%

Driver IDs are used internally by the data pipeline and are mapped to driver names in the dashboard.

🏆 Best Estimated Pit Decisions

The model identified several pit stops with strongly positive counterfactual differences.

Examples include:

Race	Driver ID	Pit Lap	Tyre	Difference	Assessment
Round 18	92	51	HARD	+25.83 s	EXCELLENT_PIT
Round 10	85	23	MEDIUM	+20.87 s	EXCELLENT_PIT
Round 10	86	15	MEDIUM	+19.83 s	EXCELLENT_PIT
Round 10	88	18	MEDIUM	+11.25 s	EXCELLENT_PIT
Round 10	103	51	HARD	+10.71 s	EXCELLENT_PIT
⚠️ Most Costly Estimated Pit Decisions

The model also identified pit stops where the counterfactual stay-out scenario was substantially faster.

Examples:

Race	Driver ID	Pit Lap	Tyre	Difference	Assessment
Round 16	86	20	MEDIUM	-43.62 s	VERY_POOR_PIT
Round 2	104	46	HARD	-35.07 s	VERY_POOR_PIT
Round 16	103	49	HARD	-32.11 s	VERY_POOR_PIT
Round 18	86	27	SOFT	-22.91 s	VERY_POOR_PIT
Round 7	96	10	MEDIUM	-22.78 s	VERY_POOR_PIT
🛞 Tyre Strategy Analysis

The system also evaluates strategy performance by tyre compound.

Current results:

Tyre	Evaluated Stops	Avg Score	Avg Adjusted Score	Avg Counterfactual	Pit Faster Rate
HARD	58	28.36	24.72	-6.46 s	25.86%
SOFT	69	15.29	15.22	-9.52 s	7.25%
MEDIUM	151	13.87	13.35	-10.91 s	9.93%

This allows the project to investigate whether strategic outcomes differ depending on the tyre compound being used before the pit stop.

🏎️ Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard provides:

Strategy Filters

Users can filter results by:

Driver
Team
Race
Strategy assessment
Strategy score
Driver search
Team search
Performance Overview

The dashboard displays:

Evaluated pit stops
Average confidence-adjusted strategy score
Pit-faster decisions
Stay-out-faster decisions
Counterfactual Outcomes

Visualizes:

Pit estimated faster
Stay-out estimated faster
Approximately equal
Strategy Assessment Distribution

Shows the distribution of:

Excellent pit
Very good pit
Good pit
Neutral
Slightly better stay-out
Poor pit
Very poor pit
Driver Strategy Ranking

Drivers can be compared using:

Average score
Confidence-adjusted score
Counterfactual difference
Pit-faster percentage
Driver Detail

Selecting a driver provides:

Number of evaluated decisions
Average score
Best decision
Pit-faster rate
Strategy performance across the season
Best & Worst Decisions

The dashboard highlights:

Most beneficial estimated pit stops
Most costly estimated pit stops
Strategy Decision Explorer

The complete evaluated decision dataset can be explored interactively.

🏗️ Data Pipeline
                Formula 1 Race Data
                       │
                       ▼
                    FastF1
                       │
                       ▼
                Data Processing
                       │
                       ▼
                  PostgreSQL
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Race Conditions       Lap Data
             │                   │
             └─────────┬─────────┘
                       ▼
                 Pit Stop Data
                       │
                       ▼
            Counterfactual Engine
                       │
                       ▼
              Counterfactual Results
                       │
                       ▼
               Strategy Grading
                       │
                       ▼
               Strategy Analysis
                       │
                       ▼
                Streamlit Dashboard
🧮 Counterfactual Methodology

For each usable pit stop, the system attempts to establish a clean comparison window.

The process includes:

1. Identify the pit stop

The driver, race and pit lap are identified.

2. Establish pre-pit baseline

Clean laps before the pit are used to estimate the driver's baseline pace.

3. Estimate degradation

The system estimates how lap time changes with tyre age.

A linear degradation model is used when sufficient data is available.

4. Construct stay-out scenario

The model estimates what the driver's next laps could have looked like if they had remained on the pre-pit tyre.

5. Compare scenarios

The estimated stay-out time is compared with the observed post-pit performance.

6. Assign confidence

The counterfactual is classified based on the quality and availability of supporting data.

7. Grade the strategy

The counterfactual difference is converted into a strategy score.

🗄️ Database

PostgreSQL is used to store and manage the underlying Formula 1 data.

The project uses relational data including:

Drivers
Teams
Races
Laps
Pit stops
Race conditions

The dashboard joins the strategy results with driver, team and race information so that users can search using meaningful Formula 1 entities rather than only internal IDs.

📁 Project Structure
F1 Strategy Grader/
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       ├── 2025_counterfactual_results.csv
│       ├── 2025_strategy_grades.csv
│       ├── 2025_driver_strategy_summary.csv
│       ├── 2025_race_strategy_summary.csv
│       ├── 2025_tyre_strategy_summary.csv
│       └── 2025_overall_strategy_summary.csv
│
├── src/
│   ├── strategy/
│   │   ├── calculate_counterfactual.py
│   │   ├── validate_counterfactual.py
│   │   ├── grade_strategy.py
│   │   └── analyze_strategy.py
│   │
│   └── dashboard/
│       └── app.py
│
├── .env
├── requirements.txt
└── README.md
⚙️ Technologies Used
Programming
Python
Pandas
NumPy
Formula 1 Data
FastF1
Database
PostgreSQL
psycopg2
Statistical Analysis
Linear regression
Tyre degradation estimation
Counterfactual modelling
Confidence classification
Visualization
Plotly
Streamlit
Development
Git
GitHub
VS Code
🚀 Installation

Clone the repository:

git clone https://github.com/Damtsi/F1-Strategy-Grader.git
cd F1-Strategy-Grader

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
🔐 Environment Configuration

Create a .env file in the project root.

Example:

DB_PASSWORD=your_postgresql_password

The application connects to PostgreSQL using the configured database settings.

Do not commit .env or database credentials to GitHub.

▶️ Running the Pipeline
1. Calculate counterfactuals
.\.venv\Scripts\python.exe src\strategy\calculate_counterfactual.py

This produces:

data/processed/2025_counterfactual_results.csv
2. Validate counterfactuals
.\.venv\Scripts\python.exe src\strategy\validate_counterfactual.py

This checks:

Record counts
Degradation behaviour
Confidence distribution
Counterfactual decisions
Extreme values
Database consistency
3. Grade strategies
.\.venv\Scripts\python.exe src\strategy\grade_strategy.py

This produces:

data/processed/2025_strategy_grades.csv
4. Analyse strategy performance
.\.venv\Scripts\python.exe src\strategy\analyze_strategy.py

This produces:

2025_driver_strategy_summary.csv
2025_race_strategy_summary.csv
2025_tyre_strategy_summary.csv
2025_overall_strategy_summary.csv
5. Launch the dashboard
.\.venv\Scripts\python.exe -m streamlit run src\dashboard\app.py


✅ Validation

The counterfactual validation pipeline currently reports:

Records                    : 278
Races represented          : 12
Drivers represented        : 21

High confidence            : 138
Medium confidence          : 140

Extreme differences > ±60s : 0

The validation also confirms that:

Race-condition filtering is present.
Confidence classification is present.
The degradation model is active.
No extreme counterfactual differences above ±60 seconds were found.


⚠️ Limitations

This project is a counterfactual analytical model, not a reconstruction of a team's actual strategy decision process.

The model does not have access to all information available to real F1 strategy teams, such as:

Real-time competitor strategy information
Traffic predictions
Team simulations
Fuel-load uncertainty
Weather forecasts available at the exact decision time
Driver-specific instructions
Live tyre temperature information
Detailed track evolution models
Internal pit-wall calculations
Full race-control decision context

Therefore:

A high strategy score does not prove that a team made the objectively correct decision, and a low score does not prove that a team made a strategically incorrect decision.

The results should be interpreted as:

Model-based estimates under the assumptions of the counterfactual framework.

🔬 Future Improvements

Potential improvements include:

Incorporating fuel-load estimation
Modelling traffic and clean-air effects
Incorporating track evolution
Adding weather conditions
Modelling Safety Car/VSC strategy more explicitly
Comparing alternative pit windows
Adding competitor strategy interactions
Improving tyre degradation modelling
Expanding counterfactual coverage
Adding race-level strategy timelines
Adding team-level strategy comparisons
Training a machine-learning model for pit-stop decision quality


💡 Why This Project Matters

The project demonstrates an important analytics concept:

evaluating decisions by comparing what actually happened with a plausible alternative scenario.

Instead of simply asking:

"What happened after the pit stop?"

the project asks:

"What might have happened if the driver had not pitted?"

This transforms raw Formula 1 telemetry into a decision-analysis problem.

The same analytical framework can be applied beyond motorsport to areas such as:

Business decision analysis
Operations
Supply-chain optimization
Pricing decisions
Marketing experiments
Risk analysis
Resource allocation


👨‍💻 Author

Damtsi Drema

B.Tech — Electronics & Communication Engineering

Interests:

Formula 1 Analytics
Data Analytics
Business Intelligence
Machine Learning
Decision Analysis


⭐ Project Summary
FastF1
   ↓
PostgreSQL
   ↓
Race + Lap + Pit Data
   ↓
Counterfactual Stay-Out Model
   ↓
Strategy Grading
   ↓
Confidence Adjustment
   ↓
Driver / Race / Tyre Analysis
   ↓
Interactive Streamlit Dashboard

2025 F1 Strategy Grader — turning Formula 1 race data into measurable strategic insights.