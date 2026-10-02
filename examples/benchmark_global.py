"""Bounded timing evidence on explicitly invented optimization problems."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer_global import search_global_portfolios
from optimizer_portfolio import generate_conditional_candidate_paths, search_candidate_portfolios


def problem(week_count, team_count):
    p, g, o = {}, {}, {}
    for week in range(1, week_count + 1):
        p[week], g[week], o[week] = {}, {}, {}
        for index in range(team_count // 2):
            a, b = f"T{2*index:02d}", f"T{2*index+1:02d}"
            probability = .55 + .3 * ((index + week) % (team_count // 2)) / max(1, team_count // 2 - 1)
            p[week][a], p[week][b] = probability, 1 - probability
            g[week][a] = g[week][b] = f"invented_w{week}_g{index}"
            o[week][a], o[week][b] = b, a
    return p, g, o


def main():
    ids = ("demo_1", "demo_2", "demo_3")
    for weeks, teams, nodes in ((2, 4, 100_000), (4, 8, 5000), (18, 32, 500)):
        p, g, o = problem(weeks, teams)
        result = search_global_portfolios(ids, p, g, o, fixed_other_survivors=10,
                                          max_nodes=nodes, max_seconds=2)
        row = {"synthetic_weeks": weeks, "synthetic_teams": teams,
               "elapsed_seconds": result.elapsed_seconds, "visited_nodes": result.visited_nodes,
               "termination": result.termination, "certified": result.globally_optimal_over_all_legal_paths,
               "lower_bound": result.lower_bound, "upper_bound": result.upper_bound,
               "absolute_gap": result.absolute_gap, "relative_gap": result.relative_gap,
               "node_limit": nodes, "time_limit_seconds": 2}
        if weeks == 2:
            candidates = [generate_conditional_candidate_paths(entry, p, g, o, candidate_limit=1)
                          for entry in ids]
            baseline = search_candidate_portfolios(candidates, fixed_other_survivors=10)
            row["one_candidate_per_entry_baseline"] = baseline.objective_value
        print(json.dumps(row, sort_keys=True))

if __name__ == "__main__":
    main()
