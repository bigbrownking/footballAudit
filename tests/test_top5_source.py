import pandas as pd
import pytest

from src.top5_source import (
    FINANCIAL_SNAPSHOT_LABEL,
    prepare_player_financials,
    prepare_player_stats,
    prepare_scouting_dataset,
)


def source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "ENG-Premier League",
                "season_label": "2024/25",
                "team": "Manchester Utd",
                "player": "Example Player",
                "nation": "ENG",
                "pos": "FW",
                "age": 23,
                "nineties": 10,
                "playing_time_starts": 9,
                "playing_time_min": 900,
                "ss_player_id": 123.0,
                "fb_gls": 5,
                "fb_ast": 2,
                "us_xg": 4.5,
                "ss_successfulDribbles": 20,
                "ss_accuratePassesPercentage": 81.5,
                "fifa_value_eur": 1_000_000,
                "fifa_wage_eur": 10_000,
                "fifa_contract_end": 2028,
            },
            {
                "league": "ENG-Premier League",
                "season_label": "2025/26",
                "team": "Manchester Utd",
                "player": "Example Player",
                "nation": "ENG",
                "pos": "FW",
                "age": 24,
                "nineties": 3,
                "playing_time_starts": 2,
                "playing_time_min": 270,
                "ss_player_id": 123.0,
                "fb_gls": 1,
                "fb_ast": 0,
                "us_xg": 0.8,
                "ss_successfulDribbles": 2,
                "ss_accuratePassesPercentage": 78.0,
                "fifa_value_eur": 1_500_000,
                "fifa_wage_eur": 12_000,
                "fifa_contract_end": 2029,
            },
        ]
    )


def test_stats_are_mapped_and_per90_respects_minutes_floor() -> None:
    stats = prepare_player_stats(source_rows())

    assert stats.loc[0, "team"] == "Manchester United"
    assert stats.loc[0, "goals_per90"] == pytest.approx(0.5)
    assert stats.loc[0, "xg_per90"] == pytest.approx(0.45)
    assert bool(stats.loc[0, "minutes_eligible"])
    assert pd.isna(stats.loc[1, "goals_per90"])
    assert not bool(stats.loc[1, "minutes_eligible"])
    assert stats.loc[0, "player_id"] == stats.loc[1, "player_id"] == "123"


def test_scouting_dataset_uses_only_2025_26_sporting_rows() -> None:
    stats = prepare_player_stats(source_rows())

    scouting = prepare_scouting_dataset(stats)

    assert scouting["season"].unique().tolist() == ["2025/26"]
    assert "ea_value_eur" not in scouting.columns


def test_financial_data_is_separate_and_discloses_snapshot_limitation() -> None:
    financials = prepare_player_financials(source_rows())

    assert len(financials) == 1
    assert financials.loc[0, "ea_value_eur"] == 1_500_000
    assert financials.loc[0, "ea_contract_end_year"] == 2029
    assert financials.loc[0, "financial_snapshot"] == FINANCIAL_SNAPSHOT_LABEL


def test_missing_source_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_player_stats(pd.DataFrame({"player": ["Example"]}))
