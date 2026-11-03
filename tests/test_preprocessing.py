import pandas as pd
import pytest

from src.preprocessing import prepare_players


def test_per90_metrics_and_minutes_eligibility() -> None:
    players = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "Player One",
                "team": "Club A",
                "league": "League A",
                "season": "2024",
                "position": "FW",
                "age": 24,
                "minutes": 900,
                "goals": 5,
                "market_value": 1_000_000,
            },
            {
                "player_id": "p2",
                "player_name": "Player Two",
                "team": "Club A",
                "league": "League A",
                "season": "2024",
                "position": "FW",
                "age": 20,
                "minutes": 300,
                "goals": 2,
                "market_value": None,
            },
        ]
    )

    result = prepare_players(players, min_minutes=450)

    assert result.loc[0, "goals_per90"] == pytest.approx(0.5)
    assert bool(result.loc[0, "minutes_eligible"])
    assert pd.isna(result.loc[1, "goals_per90"])
    assert not bool(result.loc[1, "minutes_eligible"])
    assert result.loc[0, "market_value"] == 1_000_000
    assert pd.isna(result.loc[1, "market_value"])


def test_zero_minimum_is_valid_but_negative_minimum_is_not() -> None:
    players = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "Player One",
                "team": "Club A",
                "league": "League A",
                "season": "2024",
                "position": "FW",
                "age": 24,
                "minutes": 0,
                "goals": 0,
            }
        ]
    )

    result = prepare_players(players, min_minutes=0)

    assert pd.isna(result.loc[0, "goals_per90"])
    with pytest.raises(ValueError, match="min_minutes"):
        prepare_players(players, min_minutes=-1)
