"""Exact evaluation and candidate search for up to three Survivor entries.

The evaluator is exact for a fixed set of active paths. It propagates probability
through every possible alive/dead state and samples no outcomes. Entries
that select the same team share an outcome, while entries on opposite sides of
the same game have mutually exclusive outcomes.

The search helper operates over caller-supplied candidate paths.  Exhaustive
search is exact only within those candidate sets; it is not a global certificate
over every legal NFL path.  When the Cartesian product is too large, the helper
uses a seeded random search followed by coordinate improvement and reports that
fact explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product
import math
import random
from typing import Iterable, Mapping, Sequence


_MAX_ENTRY_COUNT = 3
_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class Pick:
    """One entry's selection for one week.

    ``game_id`` must identify the physical game, and must therefore be identical
    for both sides of that game.  ``win_probability`` is the fair probability
    for ``team`` and should already have vig removed.
    """

    week: int
    game_id: str
    team: str
    opponent: str
    win_probability: float

    def __post_init__(self) -> None:
        if self.week < 1:
            raise ValueError("week must be a positive integer")
        if not self.game_id.strip():
            raise ValueError("game_id must not be empty")
        if not self.team.strip() or not self.opponent.strip():
            raise ValueError("team and opponent must not be empty")
        if self.team == self.opponent:
            raise ValueError("team and opponent must be different")
        if not math.isfinite(self.win_probability):
            raise ValueError("win_probability must be finite")
        if not 0.0 < self.win_probability <= 1.0:
            raise ValueError("win_probability must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class EntryPath:
    """A legal provisional path for one Survivor entry."""

    entry_id: str
    picks: tuple[Pick, ...]

    def __post_init__(self) -> None:
        if not self.entry_id.strip():
            raise ValueError("entry_id must not be empty")
        if not self.picks:
            raise ValueError("a path must contain at least one pick")

        weeks = [pick.week for pick in self.picks]
        if weeks != sorted(weeks) or len(weeks) != len(set(weeks)):
            raise ValueError("path weeks must be unique and in ascending order")

        teams = [pick.team for pick in self.picks]
        if len(teams) != len(set(teams)):
            raise ValueError("a team cannot be used twice within one entry")


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Exact metrics for one fixed portfolio of one to three active entries."""

    weeks: tuple[int, ...]
    # Index 0 is the pre-horizon state.  Later rows are after each listed week.
    state_probabilities_by_week: tuple[tuple[float, ...], ...]
    alive_count_distribution_by_week: tuple[tuple[float, ...], ...]
    probability_any_survives_by_week: tuple[float, ...]
    expected_entries_alive_by_week: tuple[float, ...]
    expected_longest_survival: float
    final_survivor_distribution: tuple[float, ...]
    expected_prize_share_by_other_survivors: Mapping[int, float]

    @property
    def probability_any_survives(self) -> float:
        """Probability at least one owned entry survives the full horizon."""

        return self.probability_any_survives_by_week[-1]

    @property
    def expected_final_survivors(self) -> float:
        """Expected number of owned entries surviving the full horizon."""

        return self.expected_entries_alive_by_week[-1]

    def expected_prize_share(self, other_survivors: int) -> float:
        """Return E[K/(K+F)] for a fixed number F of other survivors.

        This treats ``other_survivors`` as exogenous and fixed.  It does not model
        field picks or correlation between owned entries and the field.
        """

        _validate_other_survivors(other_survivors)
        return _expected_share(self.final_survivor_distribution, other_survivors)


@dataclass(frozen=True, slots=True)
class PortfolioSearchResult:
    """Result of searching combinations of supplied candidate paths."""

    paths: tuple[EntryPath, ...]
    metrics: PortfolioMetrics
    objective: str
    objective_value: float
    evaluated_portfolios: int
    method: str
    exact_within_candidate_sets: bool
    globally_optimal_over_all_legal_paths: bool
    fixed_other_survivors: int | None
    expected_share_sensitivity: Mapping[int, float]
    assumptions: tuple[str, ...]
    total_candidate_portfolios: int = 0
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class _BeamState:
    teams: tuple[str, ...]
    used_teams: frozenset[str]
    survival_probability: float
    expected_weeks: float


def generate_conditional_candidate_paths(
    entry_id: str,
    weekly_probabilities: Mapping[int, Mapping[str, float]],
    game_ids: Mapping[int, Mapping[str, str]],
    opponents: Mapping[int, Mapping[str, str]],
    *,
    used_teams: Iterable[str] = (),
    beam_width: int = 300,
    candidate_limit: int | None = None,
    paths_per_current_team: int = 1,
) -> tuple[EntryPath, ...]:
    """Generate one strong legal path conditional on each current-week team.

    The beam retains partial paths with the largest optimistic expected-survival
    value.  Completed paths are evaluated with the exact cumulative-product
    objective.  The resulting candidates cover distinct current-week choices,
    but beam pruning means they are not globally certified future paths.
    """

    if not entry_id.strip():
        raise ValueError("entry_id must not be empty")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate_limit must be positive or None")
    if paths_per_current_team < 1:
        raise ValueError("paths_per_current_team must be positive")
    if not weekly_probabilities:
        raise ValueError("weekly_probabilities must not be empty")

    weeks = tuple(sorted(weekly_probabilities))
    unavailable = frozenset(used_teams)
    probabilities: dict[int, dict[str, float]] = {}
    for week in weeks:
        eligible: dict[str, float] = {}
        for team, raw_probability in weekly_probabilities[week].items():
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
                raise ValueError(
                    f"probability for {team!r} in week {week} must be in (0, 1]"
                )
            if team not in unavailable:
                eligible[team] = probability
        if not eligible:
            raise ValueError(f"week {week} has no eligible teams")
        probabilities[week] = eligible

    continuation_upper = [0.0] * (len(weeks) + 1)
    for position in range(len(weeks) - 1, -1, -1):
        week = weeks[position]
        continuation_upper[position] = max(probabilities[week].values()) * (
            1.0 + continuation_upper[position + 1]
        )

    first_week = weeks[0]
    candidates_by_current_team: list[tuple[float, str, tuple[EntryPath, ...]]] = []
    for first_team in sorted(probabilities[first_week]):
        first_probability = probabilities[first_week][first_team]
        states = [
            _BeamState(
                teams=(first_team,),
                used_teams=unavailable | {first_team},
                survival_probability=first_probability,
                expected_weeks=first_probability,
            )
        ]

        for position, week in enumerate(weeks[1:], start=1):
            expanded: list[_BeamState] = []
            for state in states:
                for team, probability in probabilities[week].items():
                    if team in state.used_teams:
                        continue
                    survival = state.survival_probability * probability
                    expanded.append(
                        _BeamState(
                            teams=state.teams + (team,),
                            used_teams=state.used_teams | {team},
                            survival_probability=survival,
                            expected_weeks=state.expected_weeks + survival,
                        )
                    )
            if not expanded:
                states = []
                break
            remaining_upper = continuation_upper[position + 1]
            states = heapq.nlargest(
                beam_width,
                expanded,
                key=lambda state: (
                    state.expected_weeks
                    + state.survival_probability * remaining_upper,
                    state.expected_weeks,
                    state.survival_probability,
                ),
            )

        if not states:
            continue
        best_states = sorted(
            states,
            key=lambda state: (
                -state.expected_weeks,
                -state.survival_probability,
                state.teams,
            ),
        )[:paths_per_current_team]
        current_paths = tuple(
            build_entry_path(
                entry_id,
                dict(zip(weeks, state.teams, strict=True)),
                probabilities,
                game_ids,
                opponents,
                used_teams=unavailable,
            )
            for state in best_states
        )
        candidates_by_current_team.append(
            (best_states[0].expected_weeks, first_team, current_paths)
        )

    candidates_by_current_team.sort(key=lambda item: (-item[0], item[1]))
    if candidate_limit is not None:
        candidates_by_current_team = candidates_by_current_team[:candidate_limit]
    return tuple(
        path
        for _, _, current_paths in candidates_by_current_team
        for path in current_paths
    )


def build_entry_path(
    entry_id: str,
    selections: Mapping[int, str],
    weekly_probabilities: Mapping[int, Mapping[str, float]],
    game_ids: Mapping[int, Mapping[str, str]],
    opponents: Mapping[int, Mapping[str, str]],
    *,
    used_teams: Iterable[str] = (),
) -> EntryPath:
    """Build an EntryPath from the application's mapping-based data.

    This adapter uses the same week-to-team probability mappings as the
    single-entry optimizer. Game IDs and opponents supply the relationships
    needed to model shared and opposing selections correctly.
    """

    if not selections:
        raise ValueError("selections must not be empty")
    used = set(used_teams)
    selected_teams = tuple(selections[week] for week in sorted(selections))
    overlap = used.intersection(selected_teams)
    if overlap:
        raise ValueError(
            "path reuses historically selected team(s): " + ", ".join(sorted(overlap))
        )

    picks: list[Pick] = []
    for week in sorted(selections):
        team = selections[week]
        try:
            probability = float(weekly_probabilities[week][team])
            game_id = game_ids[week][team]
            opponent = opponents[week][team]
        except KeyError as exc:
            raise ValueError(
                f"missing probability, game ID, or opponent for {team!r} in week {week}"
            ) from exc
        picks.append(Pick(week, game_id, team, opponent, probability))

    return EntryPath(entry_id, tuple(picks))


def evaluate_portfolio(
    paths: Sequence[EntryPath],
    *,
    other_survivor_counts: Iterable[int] = (),
) -> PortfolioMetrics:
    """Evaluate three fixed paths exactly by alive-state propagation.

    The eight state indexes are bitmasks.  Bit 0 corresponds to ``paths[0]``,
    bit 1 to ``paths[1]``, and bit 2 to ``paths[2]``.  For example, state 5
    (binary 101) means entries one and three are alive.
    """

    normalized_paths = _validate_portfolio(paths)
    entry_count = len(normalized_paths)
    state_count = 1 << entry_count
    weeks = tuple(pick.week for pick in normalized_paths[0].picks)
    requested_f_values = tuple(dict.fromkeys(other_survivor_counts))
    for other_survivors in requested_f_values:
        _validate_other_survivors(other_survivors)

    state = [0.0] * state_count
    state[state_count - 1] = 1.0

    state_history: list[tuple[float, ...]] = [tuple(state)]
    count_history: list[tuple[float, ...]] = [
        _alive_count_distribution(state, entry_count)
    ]
    any_history: list[float] = [1.0]
    expected_alive_history: list[float] = [float(entry_count)]

    for week_index in range(len(weeks)):
        picks_for_week = tuple(path.picks[week_index] for path in normalized_paths)
        state = _advance_one_week(state, picks_for_week)
        _normalize_probability_vector(state)

        count_distribution = _alive_count_distribution(state, entry_count)
        state_history.append(tuple(state))
        count_history.append(count_distribution)
        any_history.append(1.0 - count_distribution[0])
        expected_alive_history.append(
            sum(k * probability for k, probability in enumerate(count_distribution))
        )

    final_distribution = count_history[-1]
    share_sensitivity = {
        other_survivors: _expected_share(final_distribution, other_survivors)
        for other_survivors in requested_f_values
    }

    return PortfolioMetrics(
        weeks=weeks,
        state_probabilities_by_week=tuple(state_history),
        alive_count_distribution_by_week=tuple(count_history),
        probability_any_survives_by_week=tuple(any_history),
        expected_entries_alive_by_week=tuple(expected_alive_history),
        expected_longest_survival=sum(any_history[1:]),
        final_survivor_distribution=final_distribution,
        expected_prize_share_by_other_survivors=share_sensitivity,
    )


def search_candidate_portfolios(
    candidate_paths_by_entry: Sequence[Sequence[EntryPath]],
    *,
    objective: str = "expected_prize_share",
    fixed_other_survivors: int | None = None,
    sensitivity_other_survivors: Iterable[int] = (),
    max_evaluations: int = 100_000,
    seed: int | None = None,
) -> PortfolioSearchResult:
    """Search combinations drawn from three supplied candidate-path sets.

    Supported objectives are ``expected_prize_share``,
    ``probability_any_survives``, ``expected_longest_survival``, and
    ``expected_final_survivors``.

    If the full Cartesian product fits within ``max_evaluations``, every supplied
    combination is evaluated.  Otherwise a reproducible random sample (when a
    seed is supplied) is followed by coordinate improvement.  Neither mode is a
    global certificate over legal paths that were not supplied by the caller.
    """

    entry_count = len(candidate_paths_by_entry)
    if not 1 <= entry_count <= _MAX_ENTRY_COUNT:
        raise ValueError("candidate_paths_by_entry must contain one to three sets")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    if objective not in {
        "expected_prize_share",
        "probability_any_survives",
        "expected_longest_survival",
        "expected_final_survivors",
    }:
        raise ValueError(f"unsupported objective: {objective}")
    if objective == "expected_prize_share" and fixed_other_survivors is None:
        raise ValueError(
            "fixed_other_survivors is required for expected_prize_share"
        )
    if fixed_other_survivors is not None:
        _validate_other_survivors(fixed_other_survivors)

    candidate_sets = tuple(tuple(paths) for paths in candidate_paths_by_entry)
    if any(not paths for paths in candidate_sets):
        raise ValueError("each entry must have at least one candidate path")

    for entry_index, paths in enumerate(candidate_sets):
        expected_entry_id = paths[0].entry_id
        if any(path.entry_id != expected_entry_id for path in paths):
            raise ValueError(
                f"candidate set {entry_index} contains multiple entry IDs"
            )

    sensitivity_values = list(dict.fromkeys(sensitivity_other_survivors))
    if fixed_other_survivors is not None and fixed_other_survivors not in sensitivity_values:
        sensitivity_values.append(fixed_other_survivors)
    for value in sensitivity_values:
        _validate_other_survivors(value)

    sizes = tuple(len(paths) for paths in candidate_sets)
    total_combinations = math.prod(sizes)
    cache: dict[tuple[int, ...], tuple[float, PortfolioMetrics]] = {}

    def evaluate_indexes(indexes: tuple[int, ...]) -> tuple[float, PortfolioMetrics]:
        if indexes not in cache:
            portfolio = tuple(
                candidate_sets[entry][indexes[entry]] for entry in range(entry_count)
            )
            metrics = evaluate_portfolio(
                portfolio,
                other_survivor_counts=sensitivity_values,
            )
            value = _objective_value(metrics, objective, fixed_other_survivors)
            cache[indexes] = (value, metrics)
        return cache[indexes]

    best_indexes: tuple[int, ...] | None = None
    best_value = -math.inf
    best_metrics: PortfolioMetrics | None = None

    def consider(indexes: tuple[int, ...]) -> None:
        nonlocal best_indexes, best_value, best_metrics
        value, metrics = evaluate_indexes(indexes)
        if value > best_value + _TOLERANCE:
            best_indexes = indexes
            best_value = value
            best_metrics = metrics

    if total_combinations <= max_evaluations:
        for indexes in product(*(range(size) for size in sizes)):
            consider(indexes)
        method = "exhaustive_candidate_enumeration"
        exact_within_candidates = True
    else:
        rng = random.Random(seed)
        # Include deterministic corners, then reserve part of the budget for
        # genuine coordinate improvement around the best sampled portfolio.
        consider(tuple(0 for _ in sizes))
        if max_evaluations > 1:
            consider(tuple(size - 1 for size in sizes))
        coordinate_reserve = min(
            max_evaluations - len(cache),
            sum(sizes),
        )
        random_target = max_evaluations - coordinate_reserve
        while len(cache) < random_target:
            consider(tuple(rng.randrange(size) for size in sizes))

        assert best_indexes is not None
        improved = True
        while improved and len(cache) < max_evaluations:
            improved = False
            current = best_indexes
            for entry_index, size in enumerate(sizes):
                for candidate_index in range(size):
                    trial = list(current)
                    trial[entry_index] = candidate_index
                    trial_indexes = tuple(trial)
                    before = best_value
                    consider(trial_indexes)
                    if best_value > before + _TOLERANCE:
                        improved = True
                        current = best_indexes
                    if len(cache) >= max_evaluations:
                        break
                if len(cache) >= max_evaluations:
                    break
        # If coordinate search converges before using the budget, retain the
        # remaining capacity for broader exploration.
        while len(cache) < max_evaluations:
            consider(tuple(rng.randrange(size) for size in sizes))
        method = "seeded_random_search_with_coordinate_improvement"
        exact_within_candidates = False

    assert best_indexes is not None and best_metrics is not None
    best_paths = tuple(
        candidate_sets[entry][best_indexes[entry]] for entry in range(entry_count)
    )
    assumptions = (
        "The evaluator treats game outcomes from different games as independent.",
        "Selections on the same game share one mutually exclusive game outcome.",
        "The number of other final survivors F is fixed and exogenous.",
        "Search quality is limited to the candidate paths supplied by the caller.",
    )

    return PortfolioSearchResult(
        paths=best_paths,
        metrics=best_metrics,
        objective=objective,
        objective_value=best_value,
        evaluated_portfolios=len(cache),
        method=method,
        exact_within_candidate_sets=exact_within_candidates,
        globally_optimal_over_all_legal_paths=False,
        fixed_other_survivors=fixed_other_survivors,
        expected_share_sensitivity={
            value: best_metrics.expected_prize_share(value)
            for value in sensitivity_values
        },
        assumptions=assumptions,
        total_candidate_portfolios=total_combinations,
        seed=seed,
    )


def _validate_portfolio(
    paths: Sequence[EntryPath],
) -> tuple[EntryPath, ...]:
    if not 1 <= len(paths) <= _MAX_ENTRY_COUNT:
        raise ValueError("a portfolio must contain one to three entry paths")

    normalized = tuple(paths)
    if len({path.entry_id for path in normalized}) != len(normalized):
        raise ValueError("portfolio entry IDs must be unique")

    reference_weeks = tuple(pick.week for pick in normalized[0].picks)
    for path in normalized[1:]:
        weeks = tuple(pick.week for pick in path.picks)
        if weeks != reference_weeks:
            raise ValueError("all entry paths must cover the same weeks")

    for week_index in range(len(reference_weeks)):
        _validate_week_games(tuple(path.picks[week_index] for path in normalized))

    return normalized


def _validate_week_games(picks: Sequence[Pick]) -> None:
    by_game: dict[str, list[Pick]] = {}
    for pick in picks:
        by_game.setdefault(pick.game_id, []).append(pick)

    for game_id, game_picks in by_game.items():
        teams: set[str] = set()
        for pick in game_picks:
            teams.update((pick.team, pick.opponent))
        if len(teams) != 2:
            raise ValueError(
                f"game_id {game_id!r} maps to inconsistent team pairings"
            )

        probability_by_team: dict[str, float] = {}
        for pick in game_picks:
            existing = probability_by_team.get(pick.team)
            if existing is not None and not math.isclose(
                existing, pick.win_probability, abs_tol=_TOLERANCE
            ):
                raise ValueError(
                    f"game_id {game_id!r} gives one team multiple probabilities"
                )
            probability_by_team[pick.team] = pick.win_probability

        if len(probability_by_team) == 2:
            if not math.isclose(
                sum(probability_by_team.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ):
                raise ValueError(
                    f"opposing probabilities for game_id {game_id!r} must sum to 1"
                )


def _advance_one_week(
    state_probabilities: Sequence[float], picks: Sequence[Pick]
) -> list[float]:
    next_state = [0.0] * len(state_probabilities)

    for alive_mask, state_probability in enumerate(state_probabilities):
        if state_probability <= 0.0:
            continue
        if alive_mask == 0:
            next_state[0] += state_probability
            continue

        relevant_games: dict[str, list[int]] = {}
        for entry_index, pick in enumerate(picks):
            if alive_mask & (1 << entry_index):
                relevant_games.setdefault(pick.game_id, []).append(entry_index)

        game_outcomes: list[tuple[tuple[float, int], tuple[float, int]]] = []
        for entry_indexes in relevant_games.values():
            representative = picks[entry_indexes[0]]
            side_team = representative.team
            side_probability = representative.win_probability

            side_winners = 0
            opponent_winners = 0
            for entry_index in entry_indexes:
                selected_team = picks[entry_index].team
                if selected_team == side_team:
                    side_winners |= 1 << entry_index
                else:
                    opponent_winners |= 1 << entry_index

            game_outcomes.append(
                (
                    (side_probability, side_winners),
                    (1.0 - side_probability, opponent_winners),
                )
            )

        # Enumerate at most 2^3 independent physical-game outcomes.
        for outcomes in product(*game_outcomes):
            transition_probability = state_probability
            surviving_mask = 0
            for outcome_probability, winning_entries in outcomes:
                transition_probability *= outcome_probability
                surviving_mask |= winning_entries
            next_state[surviving_mask] += transition_probability

    return next_state


def _alive_count_distribution(
    state_probabilities: Sequence[float], entry_count: int
) -> tuple[float, ...]:
    distribution = [0.0] * (entry_count + 1)
    for state, probability in enumerate(state_probabilities):
        distribution[state.bit_count()] += probability
    return tuple(distribution)


def _normalize_probability_vector(probabilities: list[float]) -> None:
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ArithmeticError(f"state probabilities sum to {total}, not 1")
    # Remove insignificant floating-point drift without changing relative mass.
    for index, probability in enumerate(probabilities):
        probabilities[index] = probability / total


def _validate_other_survivors(other_survivors: int) -> None:
    if isinstance(other_survivors, bool) or not isinstance(other_survivors, int):
        raise TypeError("other_survivors must be an integer")
    if other_survivors < 0:
        raise ValueError("other_survivors cannot be negative")


def _expected_share(
    final_survivor_distribution: Sequence[float], other_survivors: int
) -> float:
    share = 0.0
    for owned_survivors, probability in enumerate(final_survivor_distribution):
        if owned_survivors == 0:
            continue
        share += probability * owned_survivors / (owned_survivors + other_survivors)
    return share


def _objective_value(
    metrics: PortfolioMetrics,
    objective: str,
    fixed_other_survivors: int | None,
) -> float:
    if objective == "expected_prize_share":
        assert fixed_other_survivors is not None
        return metrics.expected_prize_share(fixed_other_survivors)
    if objective == "probability_any_survives":
        return metrics.probability_any_survives
    if objective == "expected_longest_survival":
        return metrics.expected_longest_survival
    if objective == "expected_final_survivors":
        return metrics.expected_final_survivors
    raise AssertionError(f"unhandled objective: {objective}")


__all__ = [
    "EntryPath",
    "Pick",
    "PortfolioMetrics",
    "PortfolioSearchResult",
    "build_entry_path",
    "evaluate_portfolio",
    "generate_conditional_candidate_paths",
    "search_candidate_portfolios",
]
