import pandas as pd

from src.validation import (
    NON_STAT_COLUMNS,
    REQUIRED_TEXT_COLUMNS,
    validate_players,
)


def prepare_players(
    players: pd.DataFrame,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Validate the input and add eligibility and per-90 statistical columns."""
    if min_minutes < 0:
        raise ValueError("min_minutes must be zero or greater.")

    report = validate_players(players)
    if not report.is_valid:
        details = "; ".join(issue.message for issue in report.errors)
        raise ValueError(f"Player data validation failed: {details}")

    prepared = players.copy()
    numeric_columns = [
        column for column in prepared.columns if column not in REQUIRED_TEXT_COLUMNS
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    minutes = prepared["minutes"]
    prepared["minutes_eligible"] = minutes.ge(min_minutes).fillna(False)
    stat_columns = [
        column
        for column in numeric_columns
        if column not in NON_STAT_COLUMNS and not column.endswith("_per90")
    ]
    enough_minutes = minutes.ge(min_minutes) & minutes.gt(0)
    for column in stat_columns:
        prepared[f"{column}_per90"] = (
            prepared[column].div(minutes).mul(90).where(enough_minutes)
        )

    return prepared
