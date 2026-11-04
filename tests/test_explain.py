import pandas as pd
import pytest

from src.explain import candidate_funnel, pool_quantiles, score_gap, weight_table
from src.ranking import score_players

WEIGHTS = {"xa_per90": 0.5, "tackles_per90": 0.5}


def scored_midfielders() -> pd.DataFrame:
    players = pd.DataFrame(
        [
            {"record_id": "a", "player_id": "1", "player_name": "Creator", "team": "Club A", "league": "L1",
             "age": 24, "position": "MF", "minutes": 2000, "xa_per90": 0.40, "tackles_per90": 1.0,
             "goals_per90": 0.1, "assists_per90": 0.4, "market_value": 20e6},
            {"record_id": "b", "player_id": "2", "player_name": "Destroyer", "team": "Club B", "league": "L1",
             "age": 29, "position": "MF", "minutes": 2000, "xa_per90": 0.05, "tackles_per90": 4.0,
             "goals_per90": 0.0, "assists_per90": 0.0, "market_value": None},
            {"record_id": "c", "player_id": "3", "player_name": "Hybrid", "team": "Club C", "league": "L2",
             "age": 21, "position": "MF,FW", "minutes": 1500, "xa_per90": 0.20, "tackles_per90": 2.0,
             "goals_per90": 0.5, "assists_per90": 0.2, "market_value": 5e6},
            {"record_id": "d", "player_id": "4", "player_name": "Weak", "team": "Club D", "league": "L2",
             "age": 31, "position": "MF", "minutes": 1000, "xa_per90": 0.01, "tackles_per90": 0.5,
             "goals_per90": 0.0, "assists_per90": 0.0, "market_value": 1e6},
        ]
    )
    return score_players(players, "MF", WEIGHTS)


def test_weight_table_normalises_to_100_points() -> None:
    table = weight_table({"xa_per90": 2.0, "tackles_per90": 1.0, "shots_per90": 1.0})

    assert table["share"].sum() == pytest.approx(1.0)
    assert table["max_points"].tolist() == pytest.approx([50.0, 25.0, 25.0])


def test_funnel_counts_never_increase_and_end_at_candidate_count() -> None:
    funnel = candidate_funnel(scored_midfielders(), "Club A", (20, 30), ["L1", "L2"], 10e6, False)

    assert funnel["players"].is_monotonic_decreasing
    assert funnel["players"].tolist() == [4, 4, 3, 2, 2, 1]


def test_score_gap_adds_up_to_the_score_difference() -> None:
    scored = scored_midfielders().set_index("player_name")
    compared = scored.loc[["Creator", "Destroyer"]]

    gap = score_gap(compared, WEIGHTS, ["Creator", "Destroyer"])

    assert gap["points:Creator"].sum() == pytest.approx(scored.loc["Creator", "score"])
    assert gap["gap:Destroyer"].sum() == pytest.approx(
        scored.loc["Destroyer", "score"] - scored.loc["Creator", "score"]
    )


def test_pool_quantiles_reports_requested_percentiles() -> None:
    quantiles = pool_quantiles(scored_midfielders(), ["xa_per90"])

    assert quantiles.columns.tolist() == ["p25", "p50", "p75", "p90"]
    assert quantiles.loc["xa_per90", "p50"] == pytest.approx(pd.Series([0.4, 0.05, 0.2, 0.01]).median())
