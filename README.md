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
