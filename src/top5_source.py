from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

SOURCE_COMMIT = "8940299c2c56c2c0932d3caf292336f5e1689893"
SOURCE_URL = (
    "https://raw.githubusercontent.com/m-mahadi/top5-football-dataset/"
    f"{SOURCE_COMMIT}/data/master/player_seasons.csv"
)

SOURCE_REQUIRED_COLUMNS = (
    "league",
    "season_label",
    "team",
    "player",
    "nation",
    "pos",
    "age",
    "nineties",
    "playing_time_starts",
    "playing_time_min",
    "ss_player_id",
)

STAT_COLUMNS = {
    "goals": "fb_gls",
    "assists": "fb_ast",
    "shots": "fb_sh",
    "xg": "us_xg",
    "npxg": "us_np_xg",
    "xa": "us_xa",
    "key_passes": "ss_keyPasses",
    "dribbles_completed": "ss_successfulDribbles",
    "tackles": "ss_tackles",
    "interceptions": "ss_interceptions",
    "aerial_duels_won": "ss_aerialDuelsWon",
    "clearances": "ss_clearances",
    "top_speed_kmh": "ss_topSpeed",
    "distance_covered_km": "ss_kilometersCovered",
    "sprints": "ss_numberOfSprints",
}

FINANCIAL_COLUMNS = {
    "ea_value_eur": "fifa_value_eur",
    "ea_wage_eur": "fifa_wage_eur",
    "ea_release_clause_eur": "fifa_release_clause_eur",
    "ea_contract_start_year": "fifa_contract_start",
    "ea_contract_end_year": "fifa_contract_end",
}

SEASON_LABELS = {"2024/25": "2024/25", "2025/26": "2025/26"}
TEAM_ALIASES = {"Manchester Utd": "Manchester United"}
MINUTES_THRESHOLD = 450
FINANCIAL_SNAPSHOT_LABEL = (
    "EA FC 26 current snapshot; source does not provide an as-of date; "
    "not historical financial data for 2026-06-01"
)


def load_source_csv(path: str | Path) -> pd.DataFrame:
    """Load the pinned m-mahadi master CSV and check its required schema."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {file_path}")

    source = pd.read_csv(file_path, encoding="utf-8-sig")
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in source]
    if missing:
        raise ValueError("Source CSV is missing required columns: " + ", ".join(missing))
    return source


def _record_id(row: pd.Series) -> str:
    key = "|".join(
        str(row[column]).strip().casefold()
        for column in ("league", "season_label", "team", "player")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _player_id(row: pd.Series) -> tuple[str | None, str]:
    sofascore_id = row["ss_player_id"]
    if pd.notna(sofascore_id) and str(sofascore_id).strip():
        return str(int(sofascore_id)), "sofascore"

    name = unicodedata.normalize("NFKD", str(row["player"]))
    name = "".join(char for char in name if not unicodedata.combining(char))
    name = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    nation = str(row["nation"]).strip().casefold()
    if name and nation and nation != "nan":
        return f"name:{name}|nation:{nation}", "name_nation_fallback"
    return None, "unmatched"


def _per90(values: pd.Series, minutes: pd.Series) -> pd.Series:
    eligible = minutes.ge(MINUTES_THRESHOLD) & minutes.gt(0)
    return values.div(minutes).mul(90).where(eligible)


def prepare_player_stats(source: pd.DataFrame) -> pd.DataFrame:
    """Map source rows to the project's season-level sporting-statistics table."""
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in source]
    if missing:
        raise ValueError("Source data is missing required columns: " + ", ".join(missing))

    frame = source.loc[source["season_label"].isin(SEASON_LABELS)].copy()
    if frame.empty:
        raise ValueError("Source contains no rows for seasons 2024/25 or 2025/26.")

    player_ids = frame.apply(_player_id, axis=1)
    frame["player_id"] = [identity[0] for identity in player_ids]
    frame["player_id_source"] = [identity[1] for identity in player_ids]

    result = pd.DataFrame(
        {
            "record_id": frame.apply(_record_id, axis=1),
            "player_id": frame["player_id"],
            "player_id_source": frame["player_id_source"],
            "player_name": frame["player"],
            "age": pd.to_numeric(frame["age"], errors="coerce"),
            "team": frame["team"].replace(TEAM_ALIASES),
            "league": frame["league"],
            "season": frame["season_label"],
            "position": frame["pos"],
            "nation": frame["nation"],
            "minutes": pd.to_numeric(frame["playing_time_min"], errors="coerce"),
            "starts": pd.to_numeric(frame["playing_time_starts"], errors="coerce"),
            "nineties": pd.to_numeric(frame["nineties"], errors="coerce"),
        },
        index=frame.index,
    )

    for output_column, source_column in STAT_COLUMNS.items():
        if source_column in frame:
            result[output_column] = pd.to_numeric(frame[source_column], errors="coerce")

    for output_column, source_column in (
        ("pass_accuracy_pct", "ss_accuratePassesPercentage"),
        ("aerial_duels_won_pct", "ss_aerialDuelsWonPercentage"),
        ("ground_duels_won_pct", "ss_groundDuelsWonPercentage"),
    ):
        if source_column in frame:
            result[output_column] = pd.to_numeric(frame[source_column], errors="coerce")

    per90_columns = (
        "goals",
        "assists",
        "shots",
        "xg",
        "npxg",
        "xa",
        "key_passes",
        "dribbles_completed",
        "tackles",
        "interceptions",
        "aerial_duels_won",
        "clearances",
        "distance_covered_km",
        "sprints",
    )
    for column in per90_columns:
        if column in result:
            result[f"{column}_per90"] = _per90(result[column], result["minutes"])

    result["minutes_eligible"] = result["minutes"].ge(MINUTES_THRESHOLD).fillna(False)
    result["source_record_key"] = (
        frame["league"].astype("string")
        + "|"
        + frame["season_label"].astype("string")
        + "|"
        + frame["team"].astype("string")
        + "|"
        + frame["player"].astype("string")
    )

    duplicate_ids = result["record_id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise ValueError(
            f"Source mapping generated {int(duplicate_ids.sum())} duplicate record IDs."
        )
    return result.reset_index(drop=True)


def prepare_player_financials(source: pd.DataFrame) -> pd.DataFrame:
    """Extract the EA FC 26 financial snapshot separately from historical stats."""
    missing = [column for column in SOURCE_REQUIRED_COLUMNS if column not in source]
    if missing:
        raise ValueError("Source data is missing required columns: " + ", ".join(missing))

    frame = source.loc[source["season_label"].eq("2025/26")].copy()
    frame["player_id"] = frame["ss_player_id"].map(
        lambda value: str(int(value)) if pd.notna(value) and str(value).strip() else pd.NA
    )
    result = pd.DataFrame(
        {
            "record_id": frame.apply(_record_id, axis=1),
            "player_id": frame["player_id"],
            "player_name": frame["player"],
            "team": frame["team"].replace(TEAM_ALIASES),
            "league": frame["league"],
            "season": frame["season_label"],
            "financial_snapshot": FINANCIAL_SNAPSHOT_LABEL,
        },
        index=frame.index,
    )

    for output_column, source_column in FINANCIAL_COLUMNS.items():
        if source_column in frame:
            result[output_column] = frame[source_column]
        else:
            result[output_column] = pd.NA

    result = result.drop_duplicates(subset=["record_id"]).reset_index(drop=True)
    return result


def prepare_scouting_dataset(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Build the 2025/26 scouting table from sporting statistics only."""
    scouting = player_stats.loc[player_stats["season"].eq("2025/26")].copy()
    if scouting.empty:
        raise ValueError("No 2025/26 player statistics are available.")
    return scouting.reset_index(drop=True)
