"""Transparent percentile-based ranking of players for one position."""

from __future__ import annotations

import pandas as pd

from src.app_data import plays_position

# The naive benchmark the role model is compared against: goals + assists per 90.
BASELINE_METRICS = ("goals_per90", "assists_per90")
# A "hidden gem" is in the top quarter of the role model but below the median of
# the naive benchmark, i.e. a player a goals-and-assists scan would skip.
HIDDEN_GEM_TOP_SHARE = 0.25


def score_players(
    players: pd.DataFrame,
    position: str,
    weights: dict[str, float],
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Score every player of `position` with at least `min_minutes` on a 0-100 scale.

    Each metric is turned into a percentile inside the position pool, so 0.8 means
    "better than 80% of players in this role" regardless of units. The score is
    the weighted average of those percentiles times 100. `pct_<metric>` keeps the
    percentile and `contrib_<metric>` the points it added, so every score can be
    explained metric by metric. A missing metric counts as a neutral 0.5 and is
    tallied in `n_missing_metrics`.
    """
    weights = {metric: weight for metric, weight in weights.items() if weight > 0}
    if not weights:
        raise ValueError("weights must contain at least one positive weight.")
    missing = [metric for metric in weights if metric not in players.columns]
    if missing:
        raise ValueError("players table is missing metrics: " + ", ".join(missing))

    in_pool = plays_position(players["position"], position) & players["minutes"].ge(min_minutes)
    pool = players.loc[in_pool].copy()
    total_weight = sum(weights.values())

    pool["score"] = 0.0
    pool["n_missing_metrics"] = 0
    for metric, weight in weights.items():
        percentile = pool[metric].rank(pct=True)
        pool["n_missing_metrics"] += percentile.isna().astype(int)
        pool[f"pct_{metric}"] = percentile.fillna(0.5)
        pool[f"contrib_{metric}"] = 100 * weight / total_weight * pool[f"pct_{metric}"]
        pool["score"] += pool[f"contrib_{metric}"]

    naive = pool[list(BASELINE_METRICS)].sum(axis=1, min_count=1)
    pool["baseline_score"] = 100 * naive.rank(pct=True).fillna(0.5)
    top_by_model = pool["score"].rank(pct=True).ge(1 - HIDDEN_GEM_TOP_SHARE)
    pool["hidden_gem"] = top_by_model & pool["baseline_score"].lt(50)

    if "market_value" in pool:
        value_millions = pd.to_numeric(pool["market_value"], errors="coerce").div(1e6)
        # Unknown value stays unknown: no value-for-money instead of a misleading number.
        pool["value_for_money"] = pool["score"].div(value_millions).where(value_millions.gt(0))

    return pool.sort_values("score", ascending=False).reset_index(drop=True)


def filter_candidates(
    scored: pd.DataFrame,
    exclude_team: str | None = None,
    age_range: tuple[int, int] | None = None,
    max_value: float | None = None,
    include_unknown_value: bool = True,
    leagues: list[str] | None = None,
) -> pd.DataFrame:
    """Narrow a scored pool to transfer candidates without changing anyone's score.

    A player transferred mid-season has one row per club; only the stint with the
    most minutes is kept so each candidate appears once.
    """
    if "player_id" in scored:
        scored = scored.sort_values("minutes", ascending=False).drop_duplicates("player_id")
        scored = scored.sort_values("score", ascending=False)
    keep = pd.Series(True, index=scored.index)
    if exclude_team:
        keep &= scored["team"].ne(exclude_team)
    if age_range:
        keep &= scored["age"].between(*age_range)
    if leagues:
        keep &= scored["league"].isin(leagues)
    if max_value is not None:
        value = pd.to_numeric(scored["market_value"], errors="coerce")
        affordable = value.le(max_value)
        unknown = value.isna()
        keep &= affordable | (unknown & include_unknown_value)
    return scored.loc[keep]
