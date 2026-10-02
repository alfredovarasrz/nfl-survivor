from __future__ import annotations

import itertools
import math

import pytest

from optimizer_single import (
    InfeasibleScheduleError,
    OptimizerInputError,
    evaluate_path,
    solve_single_entry,
)


def brute_force(
    probabilities: dict[int, dict[str, float]],
    *,
    used_teams: set[str] | None = None,
    fixed_picks: dict[int, str] | None = None,
) -> tuple[float, dict[int, str]]:
    """Independent reference solver for deliberately tiny test schedules."""

    weeks = sorted(probabilities)
    used = set(used_teams or ())
    fixed = dict(fixed_picks or {})
    best_value = -math.inf
    best_picks: dict[int, str] | None = None

    choices = [tuple(probabilities[week]) for week in weeks]
    for path in itertools.product(*choices):
        if len(set(path)) != len(path) or any(team in used for team in path):
            continue
        picks = dict(zip(weeks, path, strict=True))
        if any(picks[week] != team for week, team in fixed.items()):
            continue
        value, _ = evaluate_path(probabilities, picks)
        if value > best_value:
            best_value = value
            best_picks = picks

    if best_picks is None:
        raise InfeasibleScheduleError("Reference problem is infeasible.")
    return best_value, best_picks


@pytest.fixture
def toy_schedule() -> dict[int, dict[str, float]]:
    return {
        1: {"A": 0.90, "B": 0.80, "C": 0.50},
        2: {"A": 0.85, "B": 0.60, "D": 0.70},
        3: {"A": 0.70, "C": 0.95, "D": 0.65},
    }


def test_expected_weeks_matches_tail_sum() -> None:
    probabilities = {
        1: {"A": 0.8},
        2: {"B": 0.5},
        3: {"C": 0.25},
    }
    expected, curve = evaluate_path(probabilities, {1: "A", 2: "B", 3: "C"})

    assert curve == pytest.approx({1: 0.8, 2: 0.4, 3: 0.1})
    assert expected == pytest.approx(1.3)


def test_milp_matches_brute_force(toy_schedule: dict[int, dict[str, float]]) -> None:
    exact_value, _ = brute_force(toy_schedule)
    result = solve_single_entry(toy_schedule)

    assert result.solver_status == "optimal"
    assert result.optimality_gap == pytest.approx(0.0, abs=1e-9)
    assert result.expected_weeks == pytest.approx(exact_value, abs=1e-9)
    assert len(set(result.picks.values())) == len(toy_schedule)
    assert sum(result.survival_by_week.values()) == pytest.approx(
        result.expected_weeks
    )


def test_optimizer_can_save_the_best_team_for_later() -> None:
    probabilities = {
        1: {"A": 0.95, "B": 0.80},
        2: {"A": 0.94, "C": 0.60},
    }
    result = solve_single_entry(probabilities)

    # Greedy would use A first and obtain .95 + .95*.60 = 1.52.
    # The exact optimum saves A: .80 + .80*.94 = 1.552.
    assert result.picks == {1: "B", 2: "A"}
    assert result.expected_weeks == pytest.approx(1.552)


def test_used_teams_and_fixed_picks_are_enforced(
    toy_schedule: dict[int, dict[str, float]],
) -> None:
    exact_value, exact_picks = brute_force(
        toy_schedule, used_teams={"A"}, fixed_picks={2: "D"}
    )
    result = solve_single_entry(
        toy_schedule, used_teams={"A"}, fixed_picks={2: "D"}
    )

    assert result.expected_weeks == pytest.approx(exact_value)
    assert result.picks == exact_picks
    assert "A" not in result.picks.values()
    assert result.picks[2] == "D"


def test_current_pick_alternatives_match_forced_brute_force(
    toy_schedule: dict[int, dict[str, float]],
) -> None:
    result = solve_single_entry(toy_schedule)
    alternatives = {item.team: item for item in result.current_pick_alternatives}

    for team in toy_schedule[1]:
        try:
            exact_value, _ = brute_force(toy_schedule, fixed_picks={1: team})
        except InfeasibleScheduleError:
            assert team not in alternatives
        else:
            assert alternatives[team].expected_weeks == pytest.approx(exact_value)

    ranked = list(result.current_pick_alternatives)
    assert ranked == sorted(ranked, key=lambda item: (-item.expected_weeks, item.team))
    assert ranked[0].expected_weeks_lost == pytest.approx(0.0, abs=1e-9)
    for alternative in ranked:
        assert alternative.expected_weeks_lost == pytest.approx(
            ranked[0].expected_weeks - alternative.expected_weeks,
            abs=1e-9,
        )


def test_alternative_limit_returns_the_exact_top_group(
    toy_schedule: dict[int, dict[str, float]],
) -> None:
    all_alternatives = solve_single_entry(toy_schedule).current_pick_alternatives
    limited = solve_single_entry(
        toy_schedule,
        alternative_limit=2,
    ).current_pick_alternatives

    assert limited == all_alternatives[:2]


def test_infeasible_no_repeat_schedule_is_rejected() -> None:
    probabilities = {1: {"A": 0.8}, 2: {"A": 0.9}}

    with pytest.raises(InfeasibleScheduleError):
        solve_single_entry(probabilities)


@pytest.mark.parametrize("invalid_probability", [0.0, -0.1, 1.01, math.nan])
def test_invalid_probabilities_are_rejected(invalid_probability: float) -> None:
    with pytest.raises(OptimizerInputError):
        solve_single_entry({1: {"A": invalid_probability}})


def test_fixed_pick_cannot_reuse_a_historical_team() -> None:
    with pytest.raises(OptimizerInputError, match="already used"):
        solve_single_entry(
            {1: {"A": 0.8, "B": 0.2}},
            used_teams={"A"},
            fixed_picks={1: "A"},
        )
