import math

import pytest

from optimizer_portfolio import (
    EntryPath,
    Pick,
    build_entry_path,
    evaluate_portfolio,
    generate_conditional_candidate_paths,
    search_candidate_portfolios,
)


def test_conditional_beam_generator_covers_current_teams_and_legal_paths():
    probabilities = {
        1: {"A": 0.8, "B": 0.7, "C": 0.4},
        2: {"A": 0.6, "B": 0.9, "C": 0.5},
        3: {"A": 0.7, "B": 0.6, "C": 0.8},
    }
    game_ids = {
        week: {team: f"{week}-{team}" for team in teams}
        for week, teams in probabilities.items()
    }
    opponents = {
        week: {team: f"opp-{team}" for team in teams}
        for week, teams in probabilities.items()
    }

    paths = generate_conditional_candidate_paths(
        "mine",
        probabilities,
        game_ids,
        opponents,
        beam_width=20,
        paths_per_current_team=2,
    )

    assert {path.picks[0].team for path in paths} == {"A", "B", "C"}
    assert len(paths) == 6
    assert all(len({pick.team for pick in path.picks}) == 3 for path in paths)


def pick(week, game, team, opponent, probability):
    return Pick(week, game, team, opponent, probability)


def path(entry, *picks):
    return EntryPath(entry, tuple(picks))


def test_identical_team_has_one_shared_outcome():
    paths = (
        path("one", pick(1, "A-B", "A", "B", 0.75)),
        path("two", pick(1, "A-B", "A", "B", 0.75)),
        path("three", pick(1, "A-B", "A", "B", 0.75)),
    )

    metrics = evaluate_portfolio(paths, other_survivor_counts=(0, 2))

    assert metrics.state_probabilities_by_week[-1][0] == pytest.approx(0.25)
    assert metrics.state_probabilities_by_week[-1][7] == pytest.approx(0.75)
    assert metrics.final_survivor_distribution == pytest.approx((0.25, 0, 0, 0.75))
    assert metrics.probability_any_survives == pytest.approx(0.75)
    assert metrics.expected_final_survivors == pytest.approx(2.25)
    assert metrics.expected_longest_survival == pytest.approx(0.75)
    assert metrics.expected_prize_share(0) == pytest.approx(0.75)
    assert metrics.expected_prize_share(2) == pytest.approx(0.75 * 3 / 5)


def test_opposing_picks_are_mutually_exclusive_and_third_game_is_independent():
    paths = (
        path("one", pick(1, "A-B", "A", "B", 0.60)),
        path("two", pick(1, "A-B", "B", "A", 0.40)),
        path("three", pick(1, "C-D", "C", "D", 0.70)),
    )

    metrics = evaluate_portfolio(paths)
    final_states = metrics.state_probabilities_by_week[-1]

    assert final_states[5] == pytest.approx(0.42)  # entries 1 and 3
    assert final_states[1] == pytest.approx(0.18)  # entry 1 only
    assert final_states[6] == pytest.approx(0.28)  # entries 2 and 3
    assert final_states[2] == pytest.approx(0.12)  # entry 2 only
    assert final_states[0] == pytest.approx(0.0)
    assert metrics.final_survivor_distribution == pytest.approx((0, 0.30, 0.70, 0))
    assert metrics.probability_any_survives == pytest.approx(1.0)


def test_distinct_games_are_independent():
    paths = (
        path("one", pick(1, "A-B", "A", "B", 0.60)),
        path("two", pick(1, "C-D", "C", "D", 0.70)),
        path("three", pick(1, "E-F", "E", "F", 0.80)),
    )

    metrics = evaluate_portfolio(paths, other_survivor_counts=(5,))

    expected_none = 0.40 * 0.30 * 0.20
    expected_all = 0.60 * 0.70 * 0.80
    assert metrics.final_survivor_distribution[0] == pytest.approx(expected_none)
    assert metrics.final_survivor_distribution[3] == pytest.approx(expected_all)
    assert metrics.probability_any_survives == pytest.approx(1 - expected_none)
    assert metrics.expected_final_survivors == pytest.approx(0.60 + 0.70 + 0.80)
    assert sum(metrics.final_survivor_distribution) == pytest.approx(1.0)


def test_expected_longest_survival_sums_probability_any_alive_after_each_week():
    paths = (
        path(
            "one",
            pick(1, "A-B", "A", "B", 0.50),
            pick(2, "C-D", "C", "D", 0.60),
        ),
        path(
            "two",
            pick(1, "A-B", "A", "B", 0.50),
            pick(2, "E-F", "E", "F", 0.70),
        ),
        path(
            "three",
            pick(1, "A-B", "A", "B", 0.50),
            pick(2, "G-H", "G", "H", 0.80),
        ),
    )

    metrics = evaluate_portfolio(paths)
    probability_any_after_week_2 = 0.50 * (1 - 0.40 * 0.30 * 0.20)

    assert metrics.probability_any_survives_by_week == pytest.approx(
        (1.0, 0.50, probability_any_after_week_2)
    )
    assert metrics.expected_longest_survival == pytest.approx(
        0.50 + probability_any_after_week_2
    )


def test_rejects_inconsistent_probabilities_for_opposite_sides():
    paths = (
        path("one", pick(1, "A-B", "A", "B", 0.60)),
        path("two", pick(1, "A-B", "B", "A", 0.45)),
        path("three", pick(1, "C-D", "C", "D", 0.70)),
    )

    with pytest.raises(ValueError, match="must sum to 1"):
        evaluate_portfolio(paths)


def test_rejects_team_reuse_within_an_entry():
    with pytest.raises(ValueError, match="cannot be used twice"):
        path(
            "one",
            pick(1, "A-B", "A", "B", 0.60),
            pick(2, "A-C", "A", "C", 0.65),
        )


def test_candidate_search_is_exact_only_within_supplied_paths():
    candidates = (
        (
            path("one", pick(1, "A-B", "A", "B", 0.80)),
            path("one", pick(1, "C-D", "C", "D", 0.70)),
        ),
        (
            path("two", pick(1, "A-B", "A", "B", 0.80)),
            path("two", pick(1, "E-F", "E", "F", 0.70)),
        ),
        (
            path("three", pick(1, "A-B", "A", "B", 0.80)),
            path("three", pick(1, "G-H", "G", "H", 0.70)),
        ),
    )

    result = search_candidate_portfolios(
        candidates,
        objective="probability_any_survives",
        sensitivity_other_survivors=(0, 5, 15),
    )

    # One 80% pick plus two independent 70% picks gives 98.2%.
    assert tuple(path.picks[0].team for path in result.paths) == ("A", "E", "G")
    assert result.objective_value == pytest.approx(1 - 0.20 * 0.30**2)
    assert result.evaluated_portfolios == 8
    assert result.method == "exhaustive_candidate_enumeration"
    assert result.exact_within_candidate_sets is True
    assert result.globally_optimal_over_all_legal_paths is False
    assert set(result.expected_share_sensitivity) == {0, 5, 15}


def test_prize_share_requires_fixed_field_size_and_reports_sensitivity():
    candidates = (
        (path("one", pick(1, "A-B", "A", "B", 0.80)),),
        (path("two", pick(1, "C-D", "C", "D", 0.70)),),
        (path("three", pick(1, "E-F", "E", "F", 0.60)),),
    )

    with pytest.raises(ValueError, match="fixed_other_survivors"):
        search_candidate_portfolios(candidates, objective="expected_prize_share")

    result = search_candidate_portfolios(
        candidates,
        objective="expected_prize_share",
        fixed_other_survivors=5,
        sensitivity_other_survivors=(0, 1, 15),
    )

    assert result.objective_value == pytest.approx(
        result.metrics.expected_prize_share(5)
    )
    assert set(result.expected_share_sensitivity) == {0, 1, 5, 15}
    assert result.expected_share_sensitivity[0] == pytest.approx(
        result.metrics.probability_any_survives
    )
    assert result.expected_share_sensitivity[1] > result.expected_share_sensitivity[5]
    assert result.expected_share_sensitivity[5] > result.expected_share_sensitivity[15]


def test_random_search_labels_optimality_honestly():
    def candidates_for(entry):
        return tuple(
            path(
                entry,
                pick(1, f"{entry}-{index}-opp", f"{entry}{index}", f"O{entry}{index}", 0.50 + index / 100),
            )
            for index in range(5)
        )

    result = search_candidate_portfolios(
        (candidates_for("one"), candidates_for("two"), candidates_for("three")),
        objective="expected_longest_survival",
        max_evaluations=20,
        seed=7,
    )

    assert result.method == "seeded_random_search_with_coordinate_improvement"
    assert result.evaluated_portfolios == 20
    assert result.exact_within_candidate_sets is False
    assert result.globally_optimal_over_all_legal_paths is False
    assert math.isfinite(result.objective_value)


def test_state_propagation_matches_independent_full_outcome_enumeration():
    paths = (
        path(
            "one",
            pick(1, "W1_A-B", "A", "B", 0.60),
            pick(2, "W2_E-F", "E", "F", 0.70),
        ),
        path(
            "two",
            pick(1, "W1_A-B", "B", "A", 0.40),
            pick(2, "W2_G-H", "G", "H", 0.80),
        ),
        path(
            "three",
            pick(1, "W1_C-D", "C", "D", 0.75),
            pick(2, "W2_E-F", "E", "F", 0.70),
        ),
    )
    metrics = evaluate_portfolio(paths)

    games = (
        (1, "W1_A-B", "A", 0.60),
        (1, "W1_C-D", "C", 0.75),
        (2, "W2_E-F", "E", 0.70),
        (2, "W2_G-H", "G", 0.80),
    )
    expected_state_history = [[0.0] * 8 for _ in range(3)]
    for outcome_bits in range(1 << len(games)):
        first_team_wins = {}
        scenario_probability = 1.0
        for game_index, (week, game_id, _first_team, probability) in enumerate(games):
            outcome = bool(outcome_bits & (1 << game_index))
            first_team_wins[(week, game_id)] = outcome
            scenario_probability *= probability if outcome else 1 - probability

        alive_mask = 7
        expected_state_history[0][alive_mask] += scenario_probability
        for week_index, week in enumerate((1, 2), start=1):
            for entry_index, entry_path in enumerate(paths):
                if not alive_mask & (1 << entry_index):
                    continue
                selected = entry_path.picks[week_index - 1]
                first_team = next(
                    game[2]
                    for game in games
                    if game[0] == week and game[1] == selected.game_id
                )
                selected_wins = first_team_wins[(week, selected.game_id)] == (
                    selected.team == first_team
                )
                if not selected_wins:
                    alive_mask &= ~(1 << entry_index)
            expected_state_history[week_index][alive_mask] += scenario_probability

    for actual, expected in zip(
        metrics.state_probabilities_by_week, expected_state_history, strict=True
    ):
        assert actual == pytest.approx(expected)


def test_build_entry_path_adapts_mapping_format_and_historical_teams():
    probabilities = {1: {"A": 0.60, "B": 0.40}, 2: {"C": 0.70, "D": 0.30}}
    game_ids = {
        1: {"A": "W1_A-B", "B": "W1_A-B"},
        2: {"C": "W2_C-D", "D": "W2_C-D"},
    }
    opponents = {
        1: {"A": "B", "B": "A"},
        2: {"C": "D", "D": "C"},
    }

    entry_path = build_entry_path(
        "entry_1",
        {1: "A", 2: "C"},
        probabilities,
        game_ids,
        opponents,
        used_teams={"OLD"},
    )

    assert [selection.team for selection in entry_path.picks] == ["A", "C"]
    assert [selection.win_probability for selection in entry_path.picks] == [0.60, 0.70]

    with pytest.raises(ValueError, match="historically selected"):
        build_entry_path(
            "entry_1",
            {1: "A", 2: "C"},
            probabilities,
            game_ids,
            opponents,
            used_teams={"A"},
        )


def test_evaluator_and_search_support_one_remaining_owned_entry():
    only_path = path(
        "one",
        pick(1, "A-B", "A", "B", 0.80),
        pick(2, "C-D", "C", "D", 0.50),
    )
    metrics = evaluate_portfolio((only_path,), other_survivor_counts=(5,))

    assert metrics.probability_any_survives == pytest.approx(0.40)
    assert metrics.expected_final_survivors == pytest.approx(0.40)
    assert metrics.final_survivor_distribution == pytest.approx((0.60, 0.40))
    assert metrics.expected_prize_share(5) == pytest.approx(0.40 / 6)

    result = search_candidate_portfolios(
        ((only_path,),),
        objective="expected_prize_share",
        fixed_other_survivors=5,
    )
    assert len(result.paths) == 1
    assert result.objective_value == pytest.approx(0.40 / 6)
