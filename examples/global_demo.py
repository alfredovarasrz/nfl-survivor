"""Certify an invented three-entry problem and expose a stopped-search gap."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.demo import build_demo
from data import DataPaths
from reporting import save_weekly_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".demo-output/global"))
    args = parser.parse_args()
    for nodes, label in ((1, "limited"), (100_000, "certified")):
        report = build_demo("large", search="global", global_max_nodes=nodes)
        report["run_id"] = "synthetic_global_" + label
        saved = save_weekly_reports(DataPaths(args.output), report, season=2030, week=1,
                                    contest=report["contest"], run_id=report["run_id"])
        print(label, json.dumps(report["recommendation"], sort_keys=True))
        print("Execution:", json.dumps(report["execution"], sort_keys=True))
        print("Reports:", *(str(path) for path in saved))

if __name__ == "__main__":
    main()
