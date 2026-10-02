from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import survivor
from data import APIResponse, DataPaths, SCHEDULE_FIELDS, initialize_project


class WeeklyRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = DataPaths(self.root)
        initialize_project(self.paths, 2026)
        self.games = [
            (1, "w1_buf_nyj", "BUF", "NYJ", -200, 175),
            (1, "w1_kc_lv", "KC", "LV", -180, 160),
            (1, "w1_phi_dal", "PHI", "DAL", -150, 135),
            (2, "w2_buf_nyj", "BUF", "NYJ", -170, 150),
            (2, "w2_kc_lv", "KC", "LV", -160, 140),
            (2, "w2_phi_dal", "PHI", "DAL", -140, 125),
        ]
        with self.paths.schedule(2026).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDS)
            writer.writeheader()
            for week, game_id, away, home, _, _ in self.games:
                writer.writerow(
                    {
                        "season": 2026,
                        "week": week,
                        "game_id": game_id,
                        "away_team": away,
                        "home_team": home,
                        "kickoff": f"2026-09-{week:02d}T17:00:00Z",
                    }
                )

    def tearDown(self):
        self.temporary.cleanup()

    def _events(self):
        return [
            {
                "away_team": away,
                "home_team": home,
                "commence_time": f"2026-09-{week:02d}T17:00:00Z",
                "bookmakers": [{
                    "key": "test_book",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": away, "price": away_line},
                            {"name": home, "price": home_line},
                        ],
                    }],
                }],
            }
            for week, _, away, home, away_line, home_line in self.games
        ]

    def test_small_weekly_run_refreshes_full_horizon_and_saves_both_reports(self):
        class Client:
            def fetch_odds(inner_self, **kwargs):
                return APIResponse(self._events(), remaining_requests="490")

        with patch.object(survivor, "OddsAPIClient", return_value=Client()):
            exit_code = survivor.main(
                [
                    "--root",
                    str(self.root),
                    "weekly-run",
                    "--season",
                    "2026",
                    "--pool",
                    "small",
                    "--week",
                    "1",
                    "--end-week",
                    "2",
                    "--top",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        report_dir = self.paths.outputs / "2026" / "week_01"
        json_reports = list(report_dir.glob("*_small_pool.json"))
        markdown_reports = list(report_dir.glob("*_small_pool.md"))
        self.assertEqual(len(json_reports), 1)
        self.assertEqual(len(markdown_reports), 1)
        report = json.loads(json_reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["first_week"], 1)
        self.assertEqual(report["end_week"], 2)
        self.assertEqual(report["odds_refresh"]["api_records"], 6)
        self.assertEqual(report["odds_refresh"]["manual_records"], 0)
        self.assertEqual(len(report["probability_snapshot"]), 6)
        self.assertEqual(report["recommendation"]["solver_status"], "optimal")
        self.assertIn("Recommended pick", markdown_reports[0].read_text(encoding="utf-8"))

    def test_large_weekly_run_saves_fixed_field_portfolio_report(self):
        class Client:
            def fetch_odds(inner_self, **kwargs):
                return APIResponse(self._events(), remaining_requests="489")

        with patch.object(survivor, "OddsAPIClient", return_value=Client()):
            exit_code = survivor.main(
                [
                    "--root",
                    str(self.root),
                    "weekly-run",
                    "--season",
                    "2026",
                    "--pool",
                    "large",
                    "--other-survivors",
                    "10",
                    "--end-week",
                    "2",
                    "--candidate-paths",
                    "4",
                    "--beam-width",
                    "10",
                    "--paths-per-current-team",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        report_dir = self.paths.outputs / "2026" / "week_01"
        json_reports = list(report_dir.glob("*_large_pool.json"))
        self.assertEqual(len(json_reports), 1)
        report = json.loads(json_reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["recommendation"]["fixed_other_survivors"], 10)
        self.assertEqual(len(report["recommendation"]["paths"]), 3)
        self.assertTrue(report["run_id"])


if __name__ == "__main__":
    unittest.main()
