"""Run invented matchups and odds through the production optimization core."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import replay_report
from data import DataPaths
from reporting import save_weekly_reports
from probabilities import remove_vig


def build_demo(pool, *, search="candidate", global_max_nodes=100_000):
    rows = json.loads((Path(__file__).with_name("synthetic_inputs.json")).read_text())
    for row in rows:
        row["probability_a"], row["probability_b"] = remove_vig(row["moneyline_a"], row["moneyline_b"])
    report = {
        "run_id": "synthetic_" + pool, "generated_at": "2030-09-01T12:00:00+00:00",
        "season": 2030, "pool": pool, "contest": "small_pool" if pool == "small" else "large_pool",
        "entries": ["demo"] if pool == "small" else ["demo_1", "demo_2", "demo_3"],
        "first_week": 1, "end_week": 2, "pick_history": [], "probability_snapshot": rows,
        "input_description": "Entirely synthetic moneylines, matchups, dates and entries; no actual season data",
        "odds_request": {"mode": "synthetic_offline"}, "manual_max_age_hours": 24,
        "odds_refresh": {"api_records": 0, "manual_records": 0, "manual_games": [],
                         "api_warning": None, "raw_response_path": None},
        "optimizer_parameters": {"other_survivors": 10 if pool == "large" else None,
                                 "sensitivity": [0, 5, 10, 15] if pool == "large" else None,
                                 "max_evaluations": 20 if pool == "large" else None,
                                 "candidate_paths": 4 if pool == "large" else None,
                                 "beam_width": 20 if pool == "large" else None,
                                 "paths_per_current_team": 1 if pool == "large" else None,
                                 "top_alternatives": 3 if pool == "small" else None,
                                 "seed": 7 if pool == "large" and search == "candidate" else None,
                                 "search_method": search if pool == "large" else None,
                                 "global_max_nodes": global_max_nodes,
                                 "global_max_seconds": 0,
                                 "global_absolute_gap": 1e-10},
    }
    return replay_report(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".demo-output"))
    args = parser.parse_args()
    for pool in ("small", "large"):
        report = build_demo(pool)
        paths = save_weekly_reports(DataPaths(args.output), report, season=2030, week=1,
                                    contest=report["contest"], run_id=report["run_id"])
        print(pool, json.dumps(report["recommendation"], sort_keys=True))
        print("Reports:", *(path.as_posix() for path in paths))

if __name__ == "__main__":
    main()
