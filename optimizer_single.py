"""Exact single-entry NFL Survivor optimization.

The public entry point, :func:`solve_single_entry`, maximizes the expected
number of remaining weeks survived.  If the selected win probabilities are
``p[1], ..., p[T]``, the objective is

    p[1] + p[1]p[2] + ... + p[1]p[2]...p[T].

The model is a mixed-integer linear program.  Binary variables select one team
per week.  Continuous variables track the probability of reaching each week,
and auxiliary variables exactly linearize the product of a survival
probability and a binary selection variable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Sequence

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array
except ImportError as exc:  # pragma: no cover - exercised only without SciPy
    raise ImportError(
        "The single-entry optimizer requires SciPy with scipy.optimize.milp "
        "and NumPy. Install the project dependencies before running it."
    ) from exc


class OptimizerInputError(ValueError):
    """Raised when the supplied schedule or constraints are invalid."""


class InfeasibleScheduleError(RuntimeError):
    """Raised when no legal no-repeat path exists through the supplied weeks."""


@dataclass(frozen=True)
class CurrentPickAlternative:
    """The optimal remaining path conditional on one current-week team."""

    team: str
    expected_weeks: float
    expected_weeks_lost: float
    picks: dict[int, str]
    survival_by_week: dict[int, float]


@dataclass(frozen=True)
class SingleEntryResult:
    """A certified solution to the single-entry expected-survival problem."""

    picks: dict[int, str]
    expected_weeks: float
    survival_by_week: dict[int, float]
    current_pick_alternatives: tuple[CurrentPickAlternative, ...]
    solver_status: str
    optimality_gap: float | None


@dataclass(frozen=True)
class _ValidatedInput:
    weeks: tuple[int, ...]
    probabilities: dict[int, dict[str, float]]
    fixed_picks: dict[int, str]


@dataclass(frozen=True)
class _MilpSolution:
    picks: dict[int, str]
    expected_weeks: float
    survival_by_week: dict[int, float]
    solver_status: str
    optimality_gap: float | None


def evaluate_path(
    weekly_probabilities: Mapping[int, Mapping[str, float]],
    picks: Mapping[int, str],
) -> tuple[float, dict[int, float]]:
    """Calculate exact expected weeks and the cumulative survival curve.

    This evaluator is intentionally independent from the MILP's continuous
    variables.  Recomputing the answer from the selected teams avoids exposing
    small solver-tolerance errors in the reported probabilities.
    """

    survival = 1.0
    expected_weeks = 0.0
    survival_by_week: dict[int, float] = {}

    for week in sorted(weekly_probabilities):
        if week not in picks:
            raise OptimizerInputError(f"No pick was supplied for week {week}.")
        team = picks[week]
        try:
            probability = float(weekly_probabilities[week][team])
        except KeyError as exc:
            raise OptimizerInputError(
                f"Team {team!r} is not eligible in week {week}."
            ) from exc
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise OptimizerInputError(
                f"Probability for {team!r} in week {week} must be in (0, 1]."
            )

        survival *= probability
        survival_by_week[week] = survival
        expected_weeks += survival

    return expected_weeks, survival_by_week


def solve_single_entry(
    weekly_probabilities: Mapping[int, Mapping[str, float]],
    used_teams: set[str] | frozenset[str] | None = None,
    fixed_picks: Mapping[int, str] | None = None,
    *,
    include_current_pick_alternatives: bool = True,
    alternative_limit: int | None = None,
    solver_options: Mapping[str, object] | None = None,
) -> SingleEntryResult:
    """Find the globally optimal legal path for one Survivor entry.

    Parameters
    ----------
    weekly_probabilities:
        Mapping from week number to ``{team: win_probability}``.  It should
        contain only the remaining optimization horizon.
    used_teams:
        Teams selected in completed weeks.  They are removed from every future
        week rather than represented with artificial near-zero probabilities.
    fixed_picks:
        Optional picks to force within the supplied remaining horizon.  This is
        useful for evaluating a submitted current pick or a what-if scenario.
    include_current_pick_alternatives:
        When true, re-solve once for every feasible team in the first supplied
        week and rank those choices by exact expected weeks survived.
    alternative_limit:
        Return only this many best current-week alternatives. Candidates that
        cannot enter the requested top group are safely pruned with an
        optimistic upper bound. ``None`` evaluates every feasible team.
    solver_options:
        Options passed to :func:`scipy.optimize.milp`.

    Returns
    -------
    SingleEntryResult
        Full provisional path, cumulative survival curve, expected remaining
        weeks, current-week alternatives, and HiGHS status information.
    """

    if alternative_limit is not None and alternative_limit < 1:
        raise OptimizerInputError("alternative_limit must be positive or None.")

    validated = _validate_inputs(weekly_probabilities, used_teams, fixed_picks)
    base_solution = _solve_validated(validated, solver_options)

    alternatives: list[CurrentPickAlternative] = []
    if include_current_pick_alternatives:
        first_week = validated.weeks[0]
        if first_week in validated.fixed_picks:
            candidate_teams: Sequence[str] = (validated.fixed_picks[first_week],)
        else:
            candidate_teams = tuple(sorted(validated.probabilities[first_week]))

        future_upper = 0.0
        future_survival = 1.0
        for week in validated.weeks[1:]:
            future_survival *= max(validated.probabilities[week].values())
            future_upper += future_survival
        candidate_bounds = {
            team: validated.probabilities[first_week][team] * (1.0 + future_upper)
            for team in candidate_teams
        }
        ordered_teams = sorted(
            candidate_teams,
            key=lambda team: (-candidate_bounds[team], team),
        )

        feasible_solutions: list[_MilpSolution] = []
        base_team = base_solution.picks[first_week]
        for team in ordered_teams:
            if alternative_limit is not None and len(feasible_solutions) >= alternative_limit:
                cutoff = sorted(
                    (solution.expected_weeks for solution in feasible_solutions),
                    reverse=True,
                )[alternative_limit - 1]
                if candidate_bounds[team] < cutoff - 1e-12:
                    break

            if team == base_team:
                feasible_solutions.append(base_solution)
                continue
            forced_picks = dict(validated.fixed_picks)
            forced_picks[first_week] = team
            forced_input = _ValidatedInput(
                weeks=validated.weeks,
                probabilities=validated.probabilities,
                fixed_picks=forced_picks,
            )
            try:
                feasible_solutions.append(_solve_validated(forced_input, solver_options))
            except InfeasibleScheduleError:
                # A current team can be individually eligible yet make a legal
                # no-repeat completion impossible.  It is not a valid alternative.
                continue

        if not feasible_solutions:  # Defensive: the base problem was feasible.
            raise InfeasibleScheduleError(
                "No current-week selection has a legal completion."
            )

        feasible_solutions.sort(key=lambda solution: (-solution.expected_weeks, solution.picks[first_week]))
        if alternative_limit is not None:
            feasible_solutions = feasible_solutions[:alternative_limit]

        best_expected = base_solution.expected_weeks
        for solution in feasible_solutions:
            team = solution.picks[first_week]
            alternatives.append(
                CurrentPickAlternative(
                    team=team,
                    expected_weeks=solution.expected_weeks,
                    expected_weeks_lost=max(
                        0.0, best_expected - solution.expected_weeks
                    ),
                    picks=solution.picks,
                    survival_by_week=solution.survival_by_week,
                )
            )
        alternatives.sort(key=lambda item: (-item.expected_weeks, item.team))

    return SingleEntryResult(
        picks=base_solution.picks,
        expected_weeks=base_solution.expected_weeks,
        survival_by_week=base_solution.survival_by_week,
        current_pick_alternatives=tuple(alternatives),
        solver_status=base_solution.solver_status,
        optimality_gap=base_solution.optimality_gap,
    )


def _validate_inputs(
    weekly_probabilities: Mapping[int, Mapping[str, float]],
    used_teams: set[str] | frozenset[str] | None,
    fixed_picks: Mapping[int, str] | None,
) -> _ValidatedInput:
    if not isinstance(weekly_probabilities, Mapping) or not weekly_probabilities:
        raise OptimizerInputError(
            "weekly_probabilities must be a non-empty mapping of weeks to teams."
        )

    used = set(used_teams or ())
    if any(not isinstance(team, str) or not team for team in used):
        raise OptimizerInputError("Every used team must be a non-empty string.")

    probabilities: dict[int, dict[str, float]] = {}
    for week, team_probabilities in weekly_probabilities.items():
        if isinstance(week, bool) or not isinstance(week, int):
            raise OptimizerInputError("Every week key must be an integer.")
        if not isinstance(team_probabilities, Mapping) or not team_probabilities:
            raise OptimizerInputError(f"Week {week} has no eligible teams.")

        eligible: dict[str, float] = {}
        for team, raw_probability in team_probabilities.items():
            if not isinstance(team, str) or not team:
                raise OptimizerInputError(
                    f"Every team in week {week} must be a non-empty string."
                )
            if (
                isinstance(raw_probability, bool)
                or not isinstance(raw_probability, Real)
            ):
                raise OptimizerInputError(
                    f"Probability for {team!r} in week {week} must be numeric."
                )
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
                raise OptimizerInputError(
                    f"Probability for {team!r} in week {week} must be in (0, 1]."
                )
            if team not in used:
                eligible[team] = probability

        if not eligible:
            raise InfeasibleScheduleError(
                f"Week {week} has no eligible teams after removing used teams."
            )
        probabilities[week] = eligible

    weeks = tuple(sorted(probabilities))
    fixed = dict(fixed_picks or {})
    for week, team in fixed.items():
        if week not in probabilities:
            raise OptimizerInputError(
                f"Fixed pick week {week} is outside the remaining horizon."
            )
        if team in used:
            raise OptimizerInputError(
                f"Fixed team {team!r} was already used in a completed week."
            )
        if team not in probabilities[week]:
            raise OptimizerInputError(
                f"Fixed team {team!r} is not eligible in week {week}."
            )

    fixed_teams = list(fixed.values())
    if len(fixed_teams) != len(set(fixed_teams)):
        raise OptimizerInputError("Fixed picks cannot reuse a team.")

    return _ValidatedInput(
        weeks=weeks,
        probabilities=probabilities,
        fixed_picks=fixed,
    )


def _solve_validated(
    data: _ValidatedInput,
    solver_options: Mapping[str, object] | None,
) -> _MilpSolution:
    choices = [
        (week, team)
        for week in data.weeks
        for team in sorted(data.probabilities[week])
    ]
    number_of_choices = len(choices)
    number_of_weeks = len(data.weeks)

    # Variable blocks: binary x[w,t], continuous s[w], continuous z[w,t].
    x_index = {choice: position for position, choice in enumerate(choices)}
    s_offset = number_of_choices
    s_index = {
        week: s_offset + position for position, week in enumerate(data.weeks)
    }
    z_offset = s_offset + number_of_weeks
    z_index = {
        choice: z_offset + position for position, choice in enumerate(choices)
    }
    variable_count = 2 * number_of_choices + number_of_weeks

    objective = np.zeros(variable_count, dtype=float)
    for week in data.weeks:
        objective[s_index[week]] = -1.0  # scipy.optimize.milp minimizes

    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:number_of_choices] = 1
    # Tight probability bounds materially strengthen the McCormick
    # linearization.  Using the generic [0, 1] interval for every survival
    # variable leaves the full 18-week model with a very weak relaxation.
    survival_lower: dict[int, float] = {}
    survival_upper: dict[int, float] = {}
    running_lower = 1.0
    running_upper = 1.0
    for week in data.weeks:
        probabilities = tuple(data.probabilities[week].values())
        running_lower *= min(probabilities)
        running_upper *= max(probabilities)
        survival_lower[week] = running_lower
        survival_upper[week] = running_upper

    lower_variable_bounds = np.zeros(variable_count, dtype=float)
    upper_variable_bounds = np.ones(variable_count, dtype=float)
    for week in data.weeks:
        lower_variable_bounds[s_index[week]] = survival_lower[week]
        upper_variable_bounds[s_index[week]] = survival_upper[week]
    for week_position, week in enumerate(data.weeks):
        previous_upper = (
            1.0
            if week_position == 0
            else survival_upper[data.weeks[week_position - 1]]
        )
        for team in data.probabilities[week]:
            upper_variable_bounds[z_index[(week, team)]] = previous_upper
    bounds = Bounds(lower_variable_bounds, upper_variable_bounds)

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(
        terms: Mapping[int, float], lower: float, upper: float
    ) -> None:
        row = len(lower_bounds)
        for column, coefficient in terms.items():
            row_indices.append(row)
            column_indices.append(column)
            coefficients.append(coefficient)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Exactly one team per week.
    for week in data.weeks:
        add_constraint(
            {
                x_index[(week, team)]: 1.0
                for team in data.probabilities[week]
            },
            1.0,
            1.0,
        )

    # No team may be used in more than one remaining week.
    all_teams = sorted(
        {team for probabilities in data.probabilities.values() for team in probabilities}
    )
    for team in all_teams:
        terms = {
            x_index[(week, team)]: 1.0
            for week in data.weeks
            if (week, team) in x_index
        }
        add_constraint(terms, -np.inf, 1.0)

    # Explicit what-if or submitted-pick constraints.
    for week, team in data.fixed_picks.items():
        add_constraint({x_index[(week, team)]: 1.0}, 1.0, 1.0)

    for week_position, week in enumerate(data.weeks):
        week_teams = data.probabilities[week]

        if week_position == 0:
            # The probability of reaching the first decision is 1, so z=x.
            for team in week_teams:
                add_constraint(
                    {
                        z_index[(week, team)]: 1.0,
                        x_index[(week, team)]: -1.0,
                    },
                    0.0,
                    0.0,
                )
        else:
            previous_week = data.weeks[week_position - 1]
            previous_survival = s_index[previous_week]
            previous_lower = survival_lower[previous_week]
            previous_upper = survival_upper[previous_week]
            for team in week_teams:
                x = x_index[(week, team)]
                z = z_index[(week, team)]

                # z = previous_survival * x for binary x and the tight
                # survival interval [previous_lower, previous_upper].
                add_constraint(
                    {z: 1.0, x: -previous_upper}, -np.inf, 0.0
                )
                add_constraint(
                    {z: 1.0, x: -previous_lower}, 0.0, np.inf
                )
                add_constraint(
                    {z: 1.0, previous_survival: -1.0, x: -previous_lower},
                    -np.inf,
                    -previous_lower,
                )
                add_constraint(
                    {z: 1.0, previous_survival: -1.0, x: -previous_upper},
                    -previous_upper,
                    np.inf,
                )

        # s[w] equals the selected probability times prior survival.
        survival_terms = {s_index[week]: 1.0}
        for team, probability in week_teams.items():
            survival_terms[z_index[(week, team)]] = -probability
        add_constraint(survival_terms, 0.0, 0.0)

    constraint_matrix = coo_array(
        (coefficients, (row_indices, column_indices)),
        shape=(len(lower_bounds), variable_count),
    ).tocsc()
    constraints = LinearConstraint(
        constraint_matrix,
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )

    options = {"presolve": True}
    if solver_options:
        options.update(solver_options)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )

    status_names = {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    status = status_names.get(int(result.status), f"unknown_{result.status}")
    if result.status == 2:
        raise InfeasibleScheduleError(
            "No legal path uses one eligible, non-repeated team in every week."
        )
    if result.x is None:
        raise RuntimeError(f"HiGHS did not return a solution: {result.message}")

    picks: dict[int, str] = {}
    for week in data.weeks:
        selected = [
            team
            for team in data.probabilities[week]
            if result.x[x_index[(week, team)]] > 0.5
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"HiGHS returned {len(selected)} selected teams for week {week}."
            )
        picks[week] = selected[0]

    expected_weeks, survival_by_week = evaluate_path(data.probabilities, picks)
    raw_gap = getattr(result, "mip_gap", None)
    optimality_gap = (
        float(raw_gap) if raw_gap is not None and math.isfinite(float(raw_gap)) else None
    )

    return _MilpSolution(
        picks=picks,
        expected_weeks=expected_weeks,
        survival_by_week=survival_by_week,
        solver_status=status,
        optimality_gap=optimality_gap,
    )


__all__ = [
    "CurrentPickAlternative",
    "InfeasibleScheduleError",
    "OptimizerInputError",
    "SingleEntryResult",
    "evaluate_path",
    "solve_single_entry",
]
