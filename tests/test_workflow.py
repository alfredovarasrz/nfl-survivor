from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from data import (
    DataPaths,
    ODDS_FIELDS,
    SCHEDULE_FIELDS,
    PickRecord,
    StaleManualOddsError,
    SurvivorDataError,
    initialize_project,
    record_pick,
)
from workflow import (
    aligned_active_entry_week,
    load_horizon,
    recommend_single_entry,
    recommend_three_entry_portfolio,
)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.paths = DataPaths(Path(self.temporary.name))
        initialize_project(self.paths, 2026)

        games = [
            (1, "w1_buf_nyj", "BUF", "NYJ"),
            (1, "w1_kc_lv", "KC", "LV"),
            (1, "w1_phi_dal", "PHI", "DAL"),
            (2, "w2_buf_nyj", "BUF", "NYJ"),
            (2, "w2_kc_lv", "KC", "LV"),
            (2, "w2_phi_dal", "PHI", "DAL"),
        ]
        with self.paths.schedule(2026).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDS)
            writer.writeheader()
            for week, game_id, away, home in games:
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

        probability_by_game = {
            "w1_buf_nyj": (0.80, 0.20),
            "w1_kc_lv": (0.70, 0.30),
            "w1_phi_dal": (0.60, 0.40),
            "w2_buf_nyj": (0.75, 0.25),
            "w2_kc_lv": (0.65, 0.35),
            "w2_phi_dal": (0.55, 0.45),
        }
        with self.paths.normalized_odds.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ODDS_FIELDS)
            writer.writeheader()
            for week, game_id, away, home in games:
                probability_a, probability_b = probability_by_game[game_id]
                writer.writerow(
                    {
                        "season": 2026,
                        "week": week,
                        "game_id": game_id,
                        "team_a": away,
                        "team_b": home,
                        "probability_a": probability_a,
                        "probability_b": probability_b,
                        "moneyline_a": "",
                        "moneyline_b": "",
                        "bookmakers": "test",
                        "source": "test",
                        "observed_at": "2026-09-01T00:00:00Z",
                    }
                )

    def tearDown(self):
        self.temporary.cleanup()

    def test_saved_data_drives_exact_single_recommendation(self):
        horizon = load_horizon(
            self.paths, season=2026, first_week=1, last_week=2
        )
        self.assertEqual(set(horizon.probabilities), {1, 2})
        result = recommend_single_entry(
            self.paths,
            season=2026,
            contest="small_pool",
            entry_id="mine",
            first_week=1,
            last_week=2,
        )
        self.assertEqual(result.solver_status, "optimal")
        self.assertEqual(len(result.picks), 2)
        self.assertEqual(len(set(result.picks.values())), 2)

    def test_direct_horizon_load_rejects_stale_manual_record(self):
        rows = []
        with self.paths.normalized_odds.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["bookmakers"] = "manual"
        rows[0]["source"] = "manual_test"
        rows[0]["observed_at"] = "2026-09-01T00:00:00+00:00"
        with self.paths.normalized_odds.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ODDS_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(StaleManualOddsError, "stale or unverified"):
            load_horizon(
                self.paths,
                season=2026,
                first_week=1,
                last_week=2,
                as_of=datetime(2026, 9, 3, tzinfo=timezone.utc),
            )

    def test_saved_data_drives_three_entry_share_optimizer(self):
        result = recommend_three_entry_portfolio(
            self.paths,
            season=2026,
            contest="large_pool",
            entry_ids=["one", "two", "three"],
            first_week=1,
            last_week=2,
            other_survivors=5,
            sensitivity=(0, 5, 15),
        )
        self.assertEqual(len(result.paths), 3)
        self.assertEqual(result.objective, "expected_prize_share")
        self.assertEqual(result.fixed_other_survivors, 5)
        self.assertEqual(set(result.expected_share_sensitivity), {0, 5, 15})
        self.assertTrue(result.exact_within_candidate_sets)

    def test_large_optimizer_drops_eliminated_owned_entries(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "large_pool", "two", 1, "BUF", "now", "loss"),
        )
        result = recommend_three_entry_portfolio(
            self.paths,
            season=2026,
            contest="large_pool",
            entry_ids=["one", "two", "three"],
            first_week=2,
            last_week=2,
            other_survivors=5,
        )
        self.assertEqual({path.entry_id for path in result.paths}, {"one", "three"})
        self.assertEqual(len(result.metrics.final_survivor_distribution), 3)

    def test_weekly_decision_rejects_misaligned_active_entry_histories(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "large_pool", "one", 1, "BUF", "now", "win"),
        )
        with self.assertRaisesRegex(SurvivorDataError, "not aligned"):
            aligned_active_entry_week(
                self.paths,
                season=2026,
                contest="large_pool",
                entry_ids=["one", "two", "three"],
            )


if __name__ == "__main__":
    unittest.main()
