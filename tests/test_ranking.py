import pandas as pd
import pytest

from src.ranking import filter_candidates, score_players

WEIGHTS = {"xa_per90": 0.5, "tackles_per90": 0.5}


def players() -> pd.DataFrame:
    """Four midfielders plus one forward; `low` has too few minutes to be scored."""
    return pd.DataFrame(
        [
            {"record_id": "a", "player_name": "Creator", "team": "Club A", "league": "L1", "age": 24,
             "position": "MF", "minutes": 2000, "xa_per90": 0.40, "tackles_per90": 1.0,
             "goals_per90": 0.1, "assists_per90": 0.4, "market_value": 20e6},
            {"record_id": "b", "player_name": "Destroyer", "team": "Club B", "league": "L1", "age": 29,
             "position": "MF", "minutes": 2000, "xa_per90": 0.05, "tackles_per90": 4.0,
             "goals_per90": 0.0, "assists_per90": 0.0, "market_value": None},
            {"record_id": "c", "player_name": "Hybrid", "team": "Club C", "league": "L2", "age": 21,
             "position": "MF,FW", "minutes": 1500, "xa_per90": 0.20, "tackles_per90": 2.0,
             "goals_per90": 0.5, "assists_per90": 0.2, "market_value": 5e6},
            {"record_id": "d", "player_name": "Weak", "team": "Club D", "league": "L2", "age": 31,
             "position": "MF", "minutes": 1000, "xa_per90": 0.01, "tackles_per90": 0.5,
             "goals_per90": 0.0, "assists_per90": 0.0, "market_value": 1e6},
            {"record_id": "e", "player_name": "Low Minutes", "team": "Club A", "league": "L1", "age": 19,
             "position": "MF", "minutes": 100, "xa_per90": None, "tackles_per90": None,
             "goals_per90": None, "assists_per90": None, "market_value": 2e6},
        ]
    )


def test_scores_are_bounded_and_low_minute_players_are_excluded() -> None:
    scored = score_players(players(), "MF", WEIGHTS, min_minutes=450)

    assert scored["score"].between(0, 100).all()
    assert "Low Minutes" not in scored["player_name"].tolist()
    assert scored["player_name"].iloc[-1] == "Weak"


def test_weights_change_the_order_and_contributions_add_up() -> None:
    attacking = score_players(players(), "MF", {"xa_per90": 1.0, "tackles_per90": 0.0})
    defensive = score_players(players(), "MF", {"xa_per90": 0.0, "tackles_per90": 1.0})

    assert attacking["player_name"].iloc[0] == "Creator"
    assert defensive["player_name"].iloc[0] == "Destroyer"
    balanced = score_players(players(), "MF", WEIGHTS)
    contributions = balanced[["contrib_xa_per90", "contrib_tackles_per90"]].sum(axis=1)
    assert contributions.tolist() == pytest.approx(balanced["score"].tolist())


def test_multi_position_player_is_scored_in_each_position_pool() -> None:
    assert "Hybrid" in score_players(players(), "MF", WEIGHTS)["player_name"].tolist()
    assert score_players(players(), "FW", WEIGHTS)["player_name"].tolist() == ["Hybrid"]


def test_hidden_gem_flags_player_the_naive_benchmark_misses() -> None:
    scored = score_players(players(), "MF", {"tackles_per90": 1.0}).set_index("player_name")

    assert bool(scored.loc["Destroyer", "hidden_gem"])
    assert not bool(scored.loc["Hybrid", "hidden_gem"])


def test_unknown_value_is_not_treated_as_zero() -> None:
    scored = score_players(players(), "MF", WEIGHTS).set_index("player_name")

    assert pd.isna(scored.loc["Destroyer", "value_for_money"])
    assert scored.loc["Hybrid", "value_for_money"] > scored.loc["Creator", "value_for_money"]


def test_filters_respect_budget_and_unknown_value_switch() -> None:
    scored = score_players(players(), "MF", WEIGHTS)

    cheap = filter_candidates(scored, max_value=10e6, include_unknown_value=False)
    assert set(cheap["player_name"]) == {"Hybrid", "Weak"}
    with_unknown = filter_candidates(scored, max_value=10e6, include_unknown_value=True)
    assert set(with_unknown["player_name"]) == {"Hybrid", "Weak", "Destroyer"}
    narrowed = filter_candidates(scored, exclude_team="Club A", age_range=(20, 30), leagues=["L1"])
    assert narrowed["player_name"].tolist() == ["Destroyer"]


def test_mid_season_transfer_keeps_only_the_longest_stint() -> None:
    frame = players().assign(player_id=["1", "2", "3", "4", "5"])
    second_stint = frame.iloc[[0]].assign(record_id="a2", team="Club Z", minutes=600)
    scored = score_players(pd.concat([frame, second_stint], ignore_index=True), "MF", WEIGHTS)

    candidates = filter_candidates(scored)

    assert candidates["player_id"].is_unique
    assert candidates.loc[candidates["player_id"].eq("1"), "team"].item() == "Club A"


def test_missing_metric_or_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing metrics"):
        score_players(players(), "MF", {"progressive_passes_per90": 1.0})
    with pytest.raises(ValueError, match="positive weight"):
        score_players(players(), "MF", {"xa_per90": 0.0})
