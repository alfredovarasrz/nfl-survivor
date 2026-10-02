"""Independent exhaustive references for global-search certificates."""
from itertools import product
import math
import pytest

from optimizer_global import search_global_portfolios
from optimizer_portfolio import generate_conditional_candidate_paths, search_candidate_portfolios


def tiny_problem():
    probabilities = {1: {"A": .8, "B": .2, "C": .65, "D": .35},
                     2: {"A": .7, "B": .3, "C": .85, "D": .15}}
    games = {week: {"A": f"{week}AB", "B": f"{week}AB", "C": f"{week}CD", "D": f"{week}CD"}
             for week in probabilities}
    opponents = {week: {"A": "B", "B": "A", "C": "D", "D": "C"} for week in probabilities}
    return probabilities, games, opponents


def exhaustive(probabilities, games, opponents, ids, used, f):
    weeks = sorted(probabilities)
    paths = {}
    for entry in ids:
        legal = []
        for path in product(*(sorted(probabilities[week]) for week in weeks)):
            if len(set(path)) == len(path) and not set(path) & set(used.get(entry, ())):
                legal.append(path)
        paths[entry] = legal
    physical = []
    for week in weeks:
        for game in sorted(set(games[week].values())):
            side = min(team for team in probabilities[week] if games[week][team] == game)
            physical.append((week, side, opponents[week][side], probabilities[week][side]))
    scenarios = []
    for bits in product((False, True), repeat=len(physical)):
        winning = set()
        probability = 1.0
        for bit, (week, a, b, pa) in zip(bits, physical):
            winning.add((week, a if bit else b))
            probability *= pa if bit else 1 - pa
        scenarios.append((probability, winning))
    best = -1.0
    for portfolio in product(*(paths[entry] for entry in ids)):
        value = 0.0
        for probability, winning in scenarios:
            k = sum(all((week, team) in winning for week, team in zip(weeks, path)) for path in portfolio)
            value += probability * (k / (k + f) if k else 0.0)
        best = max(best, value)
    return best


@pytest.mark.parametrize("f", [0, 1, 10])
@pytest.mark.parametrize("used", [{}, {"one": ("D",), "two": ("B",)}])
def test_global_certificate_matches_independent_full_path_and_outcome_enumeration(f, used):
    p, g, o = tiny_problem()
    ids = ("one", "two", "three")
    expected = exhaustive(p, g, o, ids, used, f)
    result = search_global_portfolios(ids, p, g, o, used_teams_by_entry=used,
                                      fixed_other_survivors=f, max_nodes=None, max_seconds=None)
    assert result.globally_optimal_over_all_legal_paths
    assert result.termination == "optimal_within_tolerance"
    assert result.objective_value == pytest.approx(expected, abs=1e-10)
    assert result.lower_bound <= expected + 1e-11
    assert result.upper_bound >= expected - 1e-11
    assert result.absolute_gap <= result.absolute_gap_tolerance + 2e-12
    for path in result.paths:
        teams = [pick.team for pick in path.picks]
        assert len(set(teams)) == len(teams)
        assert not set(teams) & set(used.get(path.entry_id, ()))
        assert all(pick.team in p[pick.week] for pick in path.picks)
    candidates = [generate_conditional_candidate_paths(entry, p, g, o, used_teams=used.get(entry, ()),
                   candidate_limit=1, paths_per_current_team=1) for entry in ids]
    baseline = search_candidate_portfolios(candidates, fixed_other_survivors=f)
    assert result.objective_value >= baseline.objective_value - 1e-11


def test_node_and_time_limits_preserve_global_bound_without_false_certificate():
    p, g, o = tiny_problem()
    exact = exhaustive(p, g, o, ("one", "two", "three"), {}, 10)
    arguments = (("one", "two", "three"), p, g, o)
    stopped = search_global_portfolios(*arguments, fixed_other_survivors=10, max_nodes=1, max_seconds=None)
    assert stopped.termination == "node_limit"
    assert not stopped.globally_optimal_over_all_legal_paths
    assert stopped.lower_bound <= exact <= stopped.upper_bound
    assert stopped.visited_nodes == 1
    timed = search_global_portfolios(*arguments, fixed_other_survivors=10, max_nodes=None, max_seconds=1e-9)
    assert timed.termination == "time_limit"
    assert not timed.globally_optimal_over_all_legal_paths
    assert timed.lower_bound <= exact <= timed.upper_bound
    again = search_global_portfolios(*arguments, fixed_other_survivors=10, max_nodes=1, max_seconds=None)
    assert stopped.paths == again.paths
    assert stopped.lower_bound == again.lower_bound
    assert stopped.upper_bound == again.upper_bound


def test_shared_and_opposing_game_outcomes_and_single_active_entry():
    p = {1: {"A": .6, "B": .4}}
    g = {1: {"A": "AB", "B": "AB"}}
    o = {1: {"A": "B", "B": "A"}}
    result = search_global_portfolios(("one", "two", "three"), p, g, o,
                                      fixed_other_survivors=0, max_nodes=None, max_seconds=None)
    assert result.objective_value == pytest.approx(1.0)
    assert len({path.picks[0].team for path in result.paths}) == 2
    single = search_global_portfolios(("one",), p, g, o,
                                      fixed_other_survivors=10, max_nodes=None, max_seconds=None)
    assert single.objective_value == pytest.approx(.6 / 11)


def test_no_repeat_infeasibility_and_invalid_game_mapping_are_rejected():
    p, g, o = tiny_problem()
    with pytest.raises(ValueError, match="No complete legal"):
        search_global_portfolios(("one", "two", "three"), p, g, o,
            used_teams_by_entry={entry: ("A", "B", "C") for entry in ("one", "two", "three")},
            fixed_other_survivors=10)
    o[1]["A"] = "C"
    with pytest.raises(ValueError, match="consistent opposing"):
        search_global_portfolios(("one", "two", "three"), p, g, o, fixed_other_survivors=10)


@pytest.mark.parametrize("nodes", [2, 20, 100, 500])
def test_every_partial_frontier_bound_contains_independent_optimum(nodes):
    p, g, o = tiny_problem()
    exact = exhaustive(p, g, o, ("one", "two", "three"), {}, 10)
    result = search_global_portfolios(("one", "two", "three"), p, g, o,
                                      fixed_other_survivors=10, max_nodes=nodes, max_seconds=None)
    assert result.lower_bound <= exact + 1e-11
    assert result.upper_bound >= exact - 1e-11
    if result.globally_optimal_over_all_legal_paths:
        assert result.objective_value == pytest.approx(exact, abs=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_varied_physical_matchups_and_probabilities_match_independent_reference(seed):
    import random
    rng = random.Random(seed)
    p, g, o = {}, {}, {}
    for week in (1, 2):
        teams = list("ABCD")
        rng.shuffle(teams)
        p[week], g[week], o[week] = {}, {}, {}
        for index in (0, 2):
            a, b = teams[index:index+2]
            pa = rng.uniform(.2, .8)
            p[week][a], p[week][b] = pa, 1 - pa
            g[week][a] = g[week][b] = f"{week}_{index}"
            o[week][a], o[week][b] = b, a
    ids = ("one", "two", "three")
    expected = exhaustive(p, g, o, ids, {}, 5)
    result = search_global_portfolios(ids, p, g, o, fixed_other_survivors=5,
                                      max_nodes=None, max_seconds=None)
    assert result.globally_optimal_over_all_legal_paths
    assert result.objective_value == pytest.approx(expected, abs=1e-10)
