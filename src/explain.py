"""Intermediate tables that show how the app's numbers are produced."""

from __future__ import annotations

import pandas as pd

from src.app_data import SUPPORTED_POSITIONS, role_weights
from src.ranking import filter_candidates, score_players
from src.scenarios import team_core_quality

QUANTILES = (0.25, 0.5, 0.75, 0.9)


def weight_table(weights: dict[str, float]) -> pd.DataFrame:
    """Weight, share of the total and the maximum points each metric can add."""
    total = sum(weights.values())
    table = pd.DataFrame({"metric": list(weights), "weight": list(weights.values())})
    table["share"] = table["weight"] / total
    table["max_points"] = 100 * table["share"]
    return table


def pool_quantiles(scored: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Metric values at selected percentiles of the pool: what a percentile means in units."""
    table = scored[metrics].quantile(list(QUANTILES)).T
    table.columns = [f"p{int(q * 100)}" for q in QUANTILES]
    return table


def candidate_funnel(
    scored: pd.DataFrame,
    exclude_team: str,
    age_range: tuple[int, int],
    leagues: list[str],
    max_value: float,
    include_unknown_value: bool,
) -> pd.DataFrame:
    """How many players survive each candidate filter when filters are applied cumulatively."""
    rows = [("pool", len(scored))]
    kwargs: dict = {}
    for step, extra in (
        ("one_stint", {}),
        ("exclude_team", {"exclude_team": exclude_team}),
        ("age", {"age_range": age_range}),
        ("leagues", {"leagues": leagues}),
        ("value", {"max_value": max_value, "include_unknown_value": include_unknown_value}),
    ):
        kwargs.update(extra)
        rows.append((step, len(filter_candidates(scored, **kwargs))))
    return pd.DataFrame(rows, columns=["step", "players"])


def score_gap(compared: pd.DataFrame, weights: dict[str, float], names: list[str]) -> pd.DataFrame:
    """Points per metric for each compared player and the gap of every player to the first."""
    table = weight_table(weights).set_index("metric")[["max_points"]]
    for name, (_, player) in zip(names, compared.iterrows()):
        table[f"points:{name}"] = [player[f"contrib_{metric}"] for metric in weights]
    for name in names[1:]:
        table[f"gap:{name}"] = table[f"points:{name}"] - table[f"points:{names[0]}"]
    return table


def positions_overview(
    players: pd.DataFrame,
    roles: pd.DataFrame,
    team: str,
    min_minutes: int,
    weights_by_position: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Core quality and league rank of `team` for every supported position.

    Positions missing from `weights_by_position` use the defaults from the role file.
    """
    rows = []
    for position in SUPPORTED_POSITIONS:
        weights = weights_by_position.get(position) or role_weights(roles, position)
        table = team_core_quality(score_players(players, position, weights, min_minutes))
        rows.append(
            {
                "position": position,
                "core_quality": table.get(team),
                "rank": int(table.index.get_loc(team)) + 1 if team in table.index else None,
                "teams": len(table),
            }
        )
    return pd.DataFrame(rows)
