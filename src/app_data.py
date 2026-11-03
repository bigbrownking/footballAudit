"""Load participant 1's processed tables in the shape the Streamlit app needs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCOUTING_FILE = Path("data/processed/scouting_dataset.csv")
FINANCIALS_FILE = Path("data/processed/player_financials.csv")
ROLE_FILE = Path("role_requirements.csv")

# The EA FC 26 snapshot is kept in a separate table by the data pipeline.
# The app renames it to generic names but keeps the caveat visible.
FINANCIAL_COLUMNS = {
    "ea_value_eur": "market_value",
    "ea_wage_eur": "salary",
    "ea_contract_end_year": "contract_end",
}
FINANCIAL_DISCLAIMER = (
    "Стоимость и зарплата — оценки EA FC 26 (снимок без даты), а не реальные "
    "трансферные суммы. Пустое значение означает «неизвестно», а не ноль."
)
SUPPORTED_POSITIONS = ("DF", "MF", "FW")
ROLE_COLUMNS = ("position", "metric", "weight", "label")


def plays_position(position: pd.Series, token: str) -> pd.Series:
    """True when `token` is one of the comma-separated FBref labels, e.g. "MF,FW"."""
    return position.fillna("").astype(str).str.split(",").map(lambda parts: token in parts)


def load_players(
    scouting_path: str | Path = SCOUTING_FILE,
    financials_path: str | Path = FINANCIALS_FILE,
) -> pd.DataFrame:
    """Join 2025/26 sporting statistics with the EA financial snapshot by record_id."""
    for path in (scouting_path, financials_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Processed table does not exist: {path}")

    players = pd.read_csv(scouting_path, encoding="utf-8-sig")
    financials = pd.read_csv(financials_path, encoding="utf-8-sig")
    financials = financials[["record_id", *FINANCIAL_COLUMNS]].rename(columns=FINANCIAL_COLUMNS)
    return players.merge(financials, on="record_id", how="left")


def load_role_requirements(path: str | Path = ROLE_FILE) -> pd.DataFrame:
    """Read the agreed role model: one row per (position, metric) with a weight and UI label."""
    roles = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in ROLE_COLUMNS if column not in roles]
    if missing:
        raise ValueError("role_requirements.csv is missing columns: " + ", ".join(missing))
    return roles


def role_weights(roles: pd.DataFrame, position: str) -> dict[str, float]:
    """Default metric weights for one position, e.g. {"xa_per90": 0.2, ...}."""
    subset = roles.loc[roles["position"].eq(position)]
    if subset.empty:
        raise ValueError(f"No role requirements defined for position {position}.")
    return dict(zip(subset["metric"], subset["weight"].astype(float)))


def metric_labels(roles: pd.DataFrame) -> dict[str, str]:
    """Human-readable label for every metric used in the role model."""
    return dict(zip(roles["metric"], roles["label"]))
