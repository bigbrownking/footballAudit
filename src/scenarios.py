"""Squad audit and budget-constrained reinforcement scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.app_data import plays_position

CORE_SIZE = 3  # players per position whose mean score defines "position quality"
VETERAN_AGE = 30
PROSPECT_AGE = 21
PROSPECT_MINUTES = 900
SCORE_COLUMNS = ("record_id", "score", "baseline_score", "hidden_gem", "value_for_money")


def squad_audit(
    players: pd.DataFrame, scored: pd.DataFrame, team: str, position: str
) -> pd.DataFrame:
    """Every club player covering `position`, with a score where one exists.

    Players below the minute threshold keep an empty score instead of being hidden.
    `flag` is a heuristic talking point for the sporting director, not a decision:
    - "veteran_below_median": age >= 30 and score below the position median;
    - "prospect_low_minutes": age <= 21 with fewer than 900 minutes.
    """
    in_squad = players["team"].eq(team) & plays_position(players["position"], position)
    score_columns = [column for column in SCORE_COLUMNS if column in scored]
    squad = players.loc[in_squad].merge(scored[score_columns], on="record_id", how="left")

    squad["flag"] = ""
    prospect = squad["age"].le(PROSPECT_AGE) & squad["minutes"].lt(PROSPECT_MINUTES)
    veteran = squad["age"].ge(VETERAN_AGE) & squad["score"].lt(scored["score"].median())
    squad.loc[prospect, "flag"] = "prospect_low_minutes"
    squad.loc[veteran, "flag"] = "veteran_below_median"
    return squad.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


def core_quality(group: pd.DataFrame, core_size: int = CORE_SIZE) -> float:
    """Mean score of the best `core_size` players; NaN for an empty group."""
    return float(group["score"].nlargest(core_size).mean()) if len(group) else float("nan")


def team_core_quality(scored: pd.DataFrame, core_size: int = CORE_SIZE) -> pd.Series:
    """Core quality of every team in the pool, strongest first (used for league context)."""
    by_team = scored.groupby("team")["score"].apply(lambda scores: scores.nlargest(core_size).mean())
    return by_team.sort_values(ascending=False)


@dataclass
class ReinforcementPlan:
    budget: float
    signings: pd.DataFrame
    quality_before: float
    quality_after: float
    needs_valuation: pd.DataFrame  # strong candidates without a known value

    @property
    def spent(self) -> float:
        return float(self.signings["market_value"].sum())

    @property
    def remaining(self) -> float:
        return self.budget - self.spent


def reinforcement_plan(
    candidates: pd.DataFrame,
    squad: pd.DataFrame,
    budget: float,
    n_signings: int = 1,
    core_size: int = CORE_SIZE,
) -> ReinforcementPlan:
    """Greedy plan: sign the best-scoring affordable candidates until budget or slots run out.

    Only candidates with a known value can be bought. The strongest unpriced ones are
    returned separately so a scout can price them, instead of the model silently
    treating an unknown value as free.
    """
    priced = candidates.dropna(subset=["market_value"]).sort_values("score", ascending=False)
    chosen: list = []
    remaining = budget
    for index, candidate in priced.iterrows():
        if len(chosen) == n_signings:
            break
        if candidate["market_value"] <= remaining:
            chosen.append(index)
            remaining -= candidate["market_value"]

    signings = priced.loc[chosen]
    unpriced = candidates.loc[candidates["market_value"].isna()].nlargest(5, "score")
    return ReinforcementPlan(
        budget=budget,
        signings=signings,
        quality_before=core_quality(squad, core_size),
        quality_after=core_quality(pd.concat([squad, signings]), core_size),
        needs_valuation=unpriced,
    )


def budget_sensitivity(
    candidates: pd.DataFrame,
    squad: pd.DataFrame,
    budget: float,
    deltas: tuple[float, ...] = (-0.2, 0.0, 0.2),
    **plan_kwargs,
) -> pd.DataFrame:
    """Re-run the plan for budget * (1 + delta) and tabulate what changes."""
    rows = []
    for delta in deltas:
        plan = reinforcement_plan(candidates, squad, budget * (1 + delta), **plan_kwargs)
        rows.append(
            {
                "budget_change": delta,
                "budget": plan.budget,
                "signings": ", ".join(plan.signings["player_name"]),
                "spent": plan.spent,
                "quality_after": plan.quality_after,
            }
        )
    return pd.DataFrame(rows)
