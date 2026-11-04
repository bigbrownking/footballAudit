import pandas as pd
import pytest

from src.scenarios import (
    budget_sensitivity,
    plan_steps,
    reinforcement_plan,
    squad_audit,
    team_core_quality,
)


def scored_pool() -> pd.DataFrame:
    """Already-scored defenders: two from Our Club, four external candidates."""
    return pd.DataFrame(
        [
            {"record_id": "o1", "player_name": "Own Veteran", "team": "Our Club", "age": 32,
             "position": "DF", "minutes": 2500, "score": 40.0, "market_value": 10e6},
            {"record_id": "o2", "player_name": "Own Starter", "team": "Our Club", "age": 26,
             "position": "DF", "minutes": 2800, "score": 60.0, "market_value": 30e6},
            {"record_id": "c1", "player_name": "Star", "team": "Rival", "age": 27,
             "position": "DF", "minutes": 3000, "score": 95.0, "market_value": 70e6},
            {"record_id": "c2", "player_name": "Solid", "team": "Rival", "age": 25,
             "position": "DF", "minutes": 2700, "score": 80.0, "market_value": 25e6},
            {"record_id": "c3", "player_name": "Bargain", "team": "Other", "age": 23,
             "position": "DF", "minutes": 2000, "score": 70.0, "market_value": 8e6},
            {"record_id": "c4", "player_name": "Unpriced", "team": "Other", "age": 24,
             "position": "DF", "minutes": 2200, "score": 90.0, "market_value": None},
        ]
    )


def split_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = scored_pool()
    return pool[pool["team"].ne("Our Club")], pool[pool["team"].eq("Our Club")]


def test_plan_stays_within_budget_and_improves_core_quality() -> None:
    candidates, squad = split_pool()

    plan = reinforcement_plan(candidates, squad, budget=40e6, n_signings=2)

    assert plan.signings["player_name"].tolist() == ["Solid", "Bargain"]
    assert plan.spent == pytest.approx(33e6)
    assert plan.remaining == pytest.approx(7e6)
    assert plan.quality_before == pytest.approx(50.0)
    assert plan.quality_after == pytest.approx((80 + 70 + 60) / 3)


def test_plan_steps_record_every_greedy_decision() -> None:
    candidates, _ = split_pool()

    steps = plan_steps(candidates, budget=40e6, n_signings=2)

    assert steps["player_name"].tolist() == ["Star", "Unpriced", "Solid", "Bargain"]
    assert steps["decision"].tolist() == ["too_expensive", "no_value", "signed", "signed"]
    assert steps["remaining_after"].tolist() == pytest.approx([40e6, 40e6, 15e6, 7e6])


def test_unpriced_players_are_flagged_instead_of_bought() -> None:
    candidates, squad = split_pool()

    plan = reinforcement_plan(candidates, squad, budget=500e6, n_signings=3)

    assert "Unpriced" not in plan.signings["player_name"].tolist()
    assert plan.needs_valuation["player_name"].tolist() == ["Unpriced"]


def test_budget_cut_of_20_percent_changes_the_signing() -> None:
    candidates, squad = split_pool()

    table = budget_sensitivity(candidates, squad, budget=80e6, n_signings=1)

    assert table["budget_change"].tolist() == [-0.2, 0.0, 0.2]
    assert table.loc[table["budget_change"].eq(0.0), "signings"].item() == "Star"
    assert table.loc[table["budget_change"].eq(-0.2), "signings"].item() == "Solid"
    assert table["spent"].le(table["budget"]).all()


def test_squad_audit_keeps_low_minute_players_and_flags_heuristics() -> None:
    pool = scored_pool()
    players = pd.concat(
        [
            pool.drop(columns="score"),
            pd.DataFrame([{"record_id": "o3", "player_name": "Own Prospect", "team": "Our Club",
                           "age": 19, "position": "DF,MF", "minutes": 200, "market_value": 1e6}]),
        ],
        ignore_index=True,
    )

    audit = squad_audit(players, pool, team="Our Club", position="DF")

    assert audit["player_name"].tolist() == ["Own Starter", "Own Veteran", "Own Prospect"]
    assert pd.isna(audit.loc[2, "score"])
    assert audit.set_index("player_name")["flag"].to_dict() == {
        "Own Starter": "",
        "Own Veteran": "veteran_below_median",
        "Own Prospect": "prospect_low_minutes",
    }


def test_team_core_quality_ranks_teams_by_their_best_players() -> None:
    ranking = team_core_quality(scored_pool(), core_size=2)

    assert ranking.index.tolist() == ["Rival", "Other", "Our Club"]
    assert ranking["Rival"] == pytest.approx(87.5)
