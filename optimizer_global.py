"""Complete global branch-and-bound over fixed legal Survivor portfolios.

All legal weekly choices remain in the search. Independent maximum-product
assignments bound each entry's possible terminal survival. A concave payoff
relaxation bounds joint prize share without assuming independent entry outcomes.
The exact physical-game state transition is used in every visited prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from optimizer_portfolio import (
    EntryPath, Pick, PortfolioSearchResult, evaluate_portfolio,
    _advance_one_week, _validate_week_games, _validate_other_survivors,
)


@dataclass(frozen=True, slots=True)
class GlobalPortfolioResult(PortfolioSearchResult):
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    absolute_gap: float = 1.0
    relative_gap: float | None = None
    termination: str = "not_started"
    visited_nodes: int = 0
    completed_portfolios: int = 0
    elapsed_seconds: float = 0.0
    absolute_gap_tolerance: float = 1e-10
    numerical_bound_slack: float = 1e-12


@dataclass(slots=True)
class _Frame:
    position: int
    paths: tuple[tuple[Pick, ...], ...]
    used: tuple[frozenset[str], ...]
    state: tuple[float, ...]
    upper: float
    children: object


def search_global_portfolios(
    entry_ids: Sequence[str],
    weekly_probabilities: Mapping[int, Mapping[str, float]],
    game_ids: Mapping[int, Mapping[str, str]],
    opponents: Mapping[int, Mapping[str, str]],
    *,
    used_teams_by_entry: Mapping[str, Sequence[str]] | None = None,
    fixed_other_survivors: int,
    sensitivity_other_survivors: Sequence[int] = (),
    max_nodes: int | None = 100_000,
    max_seconds: float | None = 30.0,
    absolute_gap_tolerance: float = 1e-10,
) -> GlobalPortfolioResult:
    """Search every legal fixed path combination for one to three entries.

    Limits return a feasible incumbent and a valid global upper bound. A
    certificate is reported only when the global absolute gap closes within
    the requested tolerance (plus the disclosed floating-point bound slack).
    Node-budget runs are deterministic; wall-clock-limited runs need not be.
    Memory for the DFS frontier grows with horizon, not the search space.
    """
    started = perf_counter()
    ids = tuple(entry_ids)
    if not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
        raise ValueError("One to three distinct nonempty entry IDs are required")
    _validate_other_survivors(fixed_other_survivors)
    for f in sensitivity_other_survivors:
        _validate_other_survivors(f)
    if max_nodes is not None and (isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1):
        raise ValueError("max_nodes must be positive or None")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite or None")
    if not math.isfinite(absolute_gap_tolerance) or absolute_gap_tolerance < 0:
        raise ValueError("absolute_gap_tolerance must be nonnegative and finite")
    weeks = tuple(sorted(weekly_probabilities))
    if not weeks or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in weeks):
        raise ValueError("Weeks must be positive integers")
    if weeks != tuple(range(weeks[0], weeks[-1] + 1)):
        raise ValueError("Weeks must be contiguous")
    choices_by_week = {}
    for week in weeks:
        probs = weekly_probabilities[week]
        if not probs or set(probs) != set(game_ids.get(week, {})) or set(probs) != set(opponents.get(week, {})):
            raise ValueError("Every probability needs a physical game ID and opponent")
        picks = {}
        for team, probability in probs.items():
            opponent = opponents[week][team]
            if opponent not in probs or opponents[week].get(opponent) != team or game_ids[week][opponent] != game_ids[week][team]:
                raise ValueError("Each physical game needs exactly two consistent opposing sides")
            picks[team] = Pick(week, str(game_ids[week][team]), team, opponent, float(probability))
        _validate_week_games(tuple(picks.values()))
        choices_by_week[week] = picks
    used_map = used_teams_by_entry or {}
    if set(used_map) - set(ids):
        raise ValueError("Used-team history contains an unknown entry")
    initial_used = tuple(frozenset(used_map.get(entry, ())) for entry in ids)
    slack = 1e-12

    @lru_cache(maxsize=4096)
    def completion(position, used):
        remaining = weeks[position:]
        if not remaining:
            return 1.0, ()
        teams = sorted({team for week in remaining for team in choices_by_week[week]} - used)
        if len(teams) < len(remaining):
            return None
        cost = np.full((len(remaining), len(teams)), np.inf)
        for row, week in enumerate(remaining):
            for col, team in enumerate(teams):
                if team in choices_by_week[week]:
                    cost[row, col] = -math.log(choices_by_week[week][team].win_probability)
        try:
            rows, cols = linear_sum_assignment(cost)
        except ValueError:
            return None
        if len(rows) != len(remaining) or not np.isfinite(cost[rows, cols]).all():
            return None
        path = tuple(choices_by_week[remaining[row]][teams[col]] for row, col in zip(rows, cols))
        probability = math.prod(pick.win_probability for pick in path)
        return min(1.0, probability + slack), path

    def payoff(k):
        return k / (k + fixed_other_survivors) if k else 0.0

    def bound(position, used, state):
        completions = [completion(position, u) for u in used]
        if any(value is None for value in completions):
            return None
        q = [value[0] for value in completions]
        total = 0.0
        for mask, mass in enumerate(state):
            if not mass or not mask:
                continue
            count = mask.bit_count()
            mean = min(float(count), sum(q[i] for i in range(len(ids)) if mask & (1 << i)))
            low = int(math.floor(mean))
            high = min(count, low + 1)
            # Piecewise-linear concave envelope is tighter than ordinary Jensen.
            upper = payoff(low) + (mean - low) * (payoff(high) - payoff(low))
            total += mass * upper
        return min(1.0, total + slack)

    seed_paths = []
    for entry, used in zip(ids, initial_used):
        value = completion(0, used)
        if value is None:
            raise ValueError(f"No complete legal no-repeat path exists for {entry}")
        seed_paths.append(EntryPath(entry, value[1]))
    sensitivities = tuple(dict.fromkeys((*sensitivity_other_survivors, fixed_other_survivors)))
    best_paths = tuple(seed_paths)
    best_metrics = evaluate_portfolio(best_paths, other_survivor_counts=sensitivities)
    best = best_metrics.expected_prize_share(fixed_other_survivors)
    initial_state = tuple([0.0] * ((1 << len(ids)) - 1) + [1.0])

    def children(position, used):
        week = weeks[position]
        choices = [tuple(sorted((pick for team, pick in choices_by_week[week].items() if team not in unavailable),
                                key=lambda pick: (-pick.win_probability, pick.team))) for unavailable in used]
        return iter(product(*choices))

    root_upper = bound(0, initial_used, initial_state)
    stack = [_Frame(0, tuple(() for _ in ids), initial_used, initial_state, root_upper, children(0, initial_used))]
    visited, leaves = 1, 1
    pruned_upper = best
    reason = "search_exhausted"
    while stack:
        frame = stack[-1]
        if frame.upper <= best + absolute_gap_tolerance:
            pruned_upper = max(pruned_upper, frame.upper)
            stack.pop()
            continue
        # Drain exhausted iterators before checking budgets. Iterator cleanup
        # visits no new search node and must not invalidate a completed proof.
        try:
            picks = next(frame.children)
        except StopIteration:
            stack.pop()
            continue
        if max_nodes is not None and visited >= max_nodes:
            reason = "node_limit"
            break
        if max_seconds is not None and perf_counter() - started >= max_seconds:
            reason = "time_limit"
            break
        visited += 1
        paths = tuple(path + (pick,) for path, pick in zip(frame.paths, picks))
        used = tuple(unavailable | {pick.team} for unavailable, pick in zip(frame.used, picks))
        state = tuple(_advance_one_week(frame.state, picks))
        position = frame.position + 1
        if position == len(weeks):
            leaves += 1
            value = sum(mass * payoff(mask.bit_count()) for mask, mass in enumerate(state))
            if value > best:
                proposed = tuple(EntryPath(entry, path) for entry, path in zip(ids, paths))
                metrics = evaluate_portfolio(proposed, other_survivor_counts=sensitivities)
                actual = metrics.expected_prize_share(fixed_other_survivors)
                if actual > best:
                    best_paths, best_metrics, best = proposed, metrics, actual
            continue
        upper = bound(position, used, state)
        if upper is None:
            continue
        if upper <= best + absolute_gap_tolerance:
            pruned_upper = max(pruned_upper, upper)
            continue
        stack.append(_Frame(position, paths, used, state, upper, children(position, used)))
    upper = max(best, pruned_upper, *(frame.upper for frame in stack))
    gap = max(0.0, upper - best)
    certified = gap <= absolute_gap_tolerance + 2 * slack
    termination = "optimal_within_tolerance" if certified else reason
    return GlobalPortfolioResult(
        paths=best_paths, metrics=best_metrics, objective="expected_prize_share", objective_value=best,
        evaluated_portfolios=leaves, method="global_branch_and_bound", exact_within_candidate_sets=False,
        globally_optimal_over_all_legal_paths=certified, fixed_other_survivors=fixed_other_survivors,
        expected_share_sensitivity={f: best_metrics.expected_prize_share(f) for f in sensitivities},
        assumptions=("Search spans all legal fixed paths for the supplied entries and horizon.",
                     "Different physical games and weeks are independent; identical/opposing picks share game outcomes.",
                     "Other terminal finalists F are fixed and exogenous; explicit game ties are omitted.",
                     "Bounds use independent no-repeat assignment optima and a concave terminal-payoff relaxation.",
                     "Certificates are numerical within the disclosed absolute tolerance and bound slack."),
        total_candidate_portfolios=0, seed=None, lower_bound=best, upper_bound=upper, absolute_gap=gap,
        relative_gap=gap / best if best > 0 else None, termination=termination,
        visited_nodes=visited, completed_portfolios=leaves, elapsed_seconds=perf_counter() - started,
        absolute_gap_tolerance=absolute_gap_tolerance, numerical_bound_slack=slack,
    )
