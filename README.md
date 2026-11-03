# Football transfer data pipeline

Participant 1's data preparation code for the football transfer-decision project.
The first source adapter uses the pinned
[Top-5 European Leagues dataset](https://github.com/m-mahadi/top5-football-dataset)
revision `8940299c2c56c2c0932d3caf292336f5e1689893`.

The adapter creates separate season-level player statistics and financial tables.
The 2025/26 scouting table contains sporting statistics only; financial data is not
silently joined into historical scouting analysis.

## Data sources and limitations

- The master CSV combines FBref, Understat, SofaScore, and SoFIFA fields. Source
  prefixes are retained in the upstream dataset; the adapter maps selected columns
  into documented project names.
- The source contains 2024/25 and 2025/26 seasons for the five European leagues.
- The source calls Manchester United `Manchester Utd`; the adapter normalizes this
  team name to `Manchester United`.
- Goals and assists are mapped from FBref; xG, non-penalty xG, and xA from Understat;
  passing, defending, duels, and physical metrics from SofaScore.
- SoFIFA/EA FC 26 values, wages, release clauses, and contract years are a current
  snapshot without a source-provided as-of date. They are EA estimates, not verified
  transfer fees or real club wages, and are **not** historical values as of the
  project's decision date, 2026-06-01. They are kept separately with `ea_` column
  names and must not be used as if they were known on that date.
- The dataset maintainer's license covers the repository code, not the underlying
  FBref, Understat, SofaScore, or SoFIFA data. Raw and processed data are git-ignored.
  Verify source terms before redistributing data or using it commercially.
- Progressive passes are not present in the selected master table and are not
  fabricated. Missing source metrics remain missing.
- Name-based player ID fallbacks are marked in `player_id_source`; validate these
  before using them to compare seasons or join external datasets.
- The proposed [Kaggle transfer dataset](https://www.kaggle.com/datasets/otaviofbrito/soccer-player-transfer-dataset)
  was updated on 2024-09-03 and is built from scraped Transfermarkt data. It is not
  yet imported: its published update predates the 2026-06-01 decision cutoff, so
  it cannot substantiate a complete transfer history through that date. Its transfer
  table is about 120 MB; only add it after checking the file schema, cutoff coverage,
  source terms, and player-matching strategy.

## Generated files

```text
data/
├── raw/
│   └── player_seasons.csv
└── processed/
    ├── player_stats.csv
    ├── player_stats_2024_25.csv
    ├── player_stats_2025_26.csv
    ├── player_financials.csv
    └── scouting_dataset.csv
```

`scouting_dataset.csv` is the 2025/26 sporting-statistics table. It intentionally
does not contain financial columns. `player_financials.csv` contains the separate EA FC 26 snapshot with a warning field.

## Run

Install dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Download the pinned source CSV and prepare the tables:

```powershell
python -m src.prepare_top5
```

Or use a locally obtained master CSV:

```powershell
python -m src.prepare_top5 --source data/raw/player_seasons.csv
```

Run the generic example pipeline (the sample contains synthetic data only):

```powershell
python -m src.pipeline data/sample_players.csv --output data/processed/sample_players.csv --min-minutes 450
```

Run tests:

```powershell
python -m pytest
```

The per-90 threshold defaults to 450 minutes. It excludes low-minute rows from
per-90 values but does not make the remaining statistics automatically reliable.

## Run the app

Participant 2's Streamlit app turns the processed tables into a transfer-window
decision tool for one club (default: Manchester United, season 2025/26).

```powershell
python -m pip install -r requirements.txt
python -m src.prepare_top5      # once, creates data/processed/*.csv
streamlit run app.py
```

Inputs: `data/processed/scouting_dataset.csv` joined with
`data/processed/player_financials.csv` on `record_id` (`ea_value_eur` ->
`market_value`, `ea_wage_eur` -> `salary`, `ea_contract_end_year` ->
`contract_end`), plus `role_requirements.csv` with the metric weights per position.
An empty value stays unknown; it is never treated as zero.

Code map:

- `src/app_data.py` loads and joins the tables and reads the role model.
- `src/ranking.py` scores players: every metric becomes a percentile inside the
  position pool (DF / MF / FW, minimum minutes configurable), the score is the
  weighted average of percentiles on a 0-100 scale, and `contrib_<metric>` columns
  show how many points each metric added. A goals-plus-assists percentile is kept
  as the naive baseline; a "hidden gem" is in the top quarter of the role model but
  below the baseline median. Players transferred mid-season keep only their
  longest stint when listed as candidates.
- `src/scenarios.py` audits the club's players for the position (heuristic flags
  for veterans below the median and prospects with little playing time), builds a
  greedy budget-constrained reinforcement plan, and recomputes it at -20% / +20%
  budget. Candidates without a known value are listed for valuation, not bought.
- `app.py` is the UI: squad audit, ranked candidates with score breakdown, radar
  comparison, and the reinforcement scenario. Role weights can be changed live.

Goalkeepers are not supported because the scouting table has no goalkeeper
metrics. Alternatives not implemented on purpose: z-scores instead of percentiles
(sensitive to outliers), a learned model for weights (no outcome labels in the
data), and knapsack optimisation instead of the greedy plan (harder to explain
for a one-to-three-signing window).

Tests for the ranking and the scenarios live in `tests/test_ranking.py` and
`tests/test_scenarios.py`.
