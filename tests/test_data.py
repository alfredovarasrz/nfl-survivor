from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from data import (
    APIResponse,
    DataPaths,
    MissingOddsError,
    OddsAPIClient,
    PickRecord,
    PoolStatusRecord,
    SurvivorDataError,
    StaleManualOddsError,
    apply_manual_odds,
    confirm_manual_odds,
    download_schedule,
    entry_states,
    import_schedule,
    initialize_project,
    inspect_odds_coverage,
    load_normalized_odds,
    load_picks,
    load_pool_status,
    load_schedule,
    record_pick,
    record_pool_status,
    record_result,
    refresh_odds,
    sync_results,
    update_manual_odds,
    validate_pick_against_schedule,
)
from probabilities import (
    american_odds_to_implied,
    consensus_fair_probability,
    expected_weeks_survived,
    remove_vig,
    survival_curve,
)


class FakeHTTPResponse:
    def __init__(self, payload, headers=None):
        self.payload = json.dumps(payload).encode()
        self.headers = headers or {}

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class DataLayerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = DataPaths(self.root)
        initialize_project(self.paths, 2026)
        with self.paths.schedule(2026).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("season", "week", "game_id", "away_team", "home_team", "kickoff"),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "season": 2026,
                        "week": 1,
                        "game_id": "w1_buf_nyj",
                        "away_team": "BUF",
                        "home_team": "NYJ",
                        "kickoff": "2026-09-13T17:00:00Z",
                    },
                    {
                        "season": 2026,
                        "week": 1,
                        "game_id": "w1_kc_lv",
                        "away_team": "KC",
                        "home_team": "LV",
                        "kickoff": "2026-09-13T20:00:00Z",
                    },
                ]
            )

    def tearDown(self):
        self.temporary.cleanup()

    def test_probability_formulas(self):
        self.assertAlmostEqual(american_odds_to_implied(-120), 120 / 220)
        self.assertAlmostEqual(american_odds_to_implied(150), 100 / 250)
        a, b = remove_vig(-120, 105)
        self.assertAlmostEqual(a + b, 1.0)
        consensus_a, consensus_b = consensus_fair_probability([(-120, 105), (-110, 100)])
        self.assertAlmostEqual(consensus_a + consensus_b, 1.0)

    def test_expected_survival_uses_cumulative_products(self):
        self.assertEqual(survival_curve([0.5, 1 - 1e-12]), [0.5, 0.5 * (1 - 1e-12)])
        self.assertAlmostEqual(expected_weeks_survived([0.5, 0.5]), 0.75)

    def test_schedule_rejects_duplicate_team_in_week(self):
        with self.paths.schedule(2026).open("a", newline="") as handle:
            csv.writer(handle).writerow([2026, 1, "duplicate", "BUF", "MIA", "2026-09-13T21:00:00Z"])
        with self.assertRaisesRegex(SurvivorDataError, "appears more than once"):
            load_schedule(self.paths.schedule(2026), season=2026)

    def test_schedule_is_not_replaced_without_explicit_flag(self):
        source = self.root / "source.csv"
        source.write_text(self.paths.schedule(2026).read_text())
        with self.assertRaisesRegex(SurvivorDataError, "already exists"):
            import_schedule(source, self.paths.schedule(2026), season=2026)

    def test_schedule_download_collects_all_weeks_once(self):
        class Client:
            def fetch_week(self, *, season, week):
                return {
                    "events": [{
                        "id": f"game-{week}",
                        "date": f"2026-09-{week:02d}T17:00:00Z",
                        "competitions": [{
                            "competitors": [
                                {"homeAway": "away", "team": {"abbreviation": "BUF"}},
                                {"homeAway": "home", "team": {"abbreviation": "NYJ"}},
                            ]
                        }],
                    }]
                }

        games = download_schedule(
            self.paths, season=2026, client=Client(), replace=True
        )
        self.assertEqual(len(games), 18)
        self.assertEqual({game.week for game in games}, set(range(1, 19)))
        self.assertTrue((self.paths.raw_data / "schedule_2026.json").exists())
        with self.assertRaisesRegex(SurvivorDataError, "already exists"):
            download_schedule(self.paths, season=2026, client=Client())

    def test_pick_history_rejects_reused_team(self):
        first = PickRecord(2026, "small_pool", "mine", 1, "BUF", "now", "win")
        record_pick(self.paths.picks, first)
        with self.assertRaisesRegex(SurvivorDataError, "already been used"):
            record_pick(
                self.paths.picks,
                PickRecord(2026, "small_pool", "mine", 2, "BUF", "later"),
            )

    def test_pick_must_match_the_stored_week_schedule(self):
        schedule = load_schedule(self.paths.schedule(2026), season=2026)
        with self.assertRaisesRegex(SurvivorDataError, "scheduled game"):
            validate_pick_against_schedule(
                PickRecord(2026, "small_pool", "mine", 1, "MIA", "now"),
                schedule,
            )

    def test_tie_eliminates_entry_and_rebuy_reactivates_without_resetting_teams(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now", "tie"),
        )
        state = entry_states(load_picks(self.paths.picks), season=2026)[0]
        self.assertFalse(state.active)
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 2, "KC", "later", "pending", True),
        )
        state = entry_states(load_picks(self.paths.picks), season=2026)[0]
        self.assertTrue(state.active)
        self.assertEqual(state.used_teams, frozenset(("BUF", "KC")))
        self.assertEqual(state.rebuys_used, 1)

    def test_rebuy_requires_a_prior_elimination(self):
        with self.assertRaisesRegex(SurvivorDataError, "only be recorded after"):
            record_pick(
                self.paths.picks,
                PickRecord(
                    2026, "small_pool", "mine", 1, "BUF", "now", "pending", True
                ),
            )

    def test_large_pool_rebuy_is_rejected_by_persistence(self):
        with self.assertRaisesRegex(SurvivorDataError, "does not allow rebuys"):
            record_pick(
                self.paths.picks,
                PickRecord(
                    2026, "large_pool", "mine-1", 1, "BUF", "now", "loss", True
                ),
            )

    def test_load_picks_rejects_a_hand_edited_delayed_rebuy(self):
        with self.paths.picks.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "season",
                    "contest",
                    "entry_id",
                    "week",
                    "team",
                    "submitted_at",
                    "result",
                    "rebuy",
                )
            )
            writer.writerow((2026, "small_pool", "mine", 1, "BUF", "now", "loss", "false"))
            writer.writerow((2026, "small_pool", "mine", 3, "KC", "later", "pending", "true"))

        with self.assertRaisesRegex(SurvivorDataError, "only in Week 2"):
            load_picks(self.paths.picks)

    def test_eliminated_entry_requires_rebuy_marker(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now", "loss"),
        )
        with self.assertRaisesRegex(SurvivorDataError, "Mark the next pick as a rebuy"):
            record_pick(
                self.paths.picks,
                PickRecord(2026, "small_pool", "mine", 2, "KC", "later"),
            )

    def test_rebuy_must_be_used_in_week_immediately_after_elimination(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now", "loss"),
        )
        with self.assertRaisesRegex(
            SurvivorDataError,
            "loss or tie in Week 1 permits a rebuy only in Week 2.*expired",
        ):
            record_pick(
                self.paths.picks,
                PickRecord(
                    2026, "small_pool", "mine", 3, "KC", "later", "pending", True
                ),
            )
        self.assertEqual(len(load_picks(self.paths.picks)), 1)

    def test_skipping_rebuy_week_does_not_allow_a_later_unmarked_pick(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now", "tie"),
        )
        with self.assertRaisesRegex(SurvivorDataError, "rebuy expired after Week 2"):
            record_pick(
                self.paths.picks,
                PickRecord(2026, "small_pool", "mine", 3, "KC", "later"),
            )

    def test_correcting_a_result_cannot_leave_an_invalid_rebuy_history(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now"),
        )
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 2, "KC", "later"),
        )
        with self.assertRaisesRegex(SurvivorDataError, "Mark the next pick as a rebuy"):
            record_result(
                self.paths.picks,
                season=2026,
                contest="small_pool",
                entry_id="mine",
                week=1,
                result="loss",
            )
        picks = load_picks(self.paths.picks)
        self.assertEqual([pick.result for pick in picks], ["pending", "pending"])

    def test_second_rebuy_is_rejected_even_in_the_immediate_next_week(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now", "loss"),
        )
        record_pick(
            self.paths.picks,
            PickRecord(
                2026, "small_pool", "mine", 2, "KC", "later", "loss", True
            ),
        )
        with self.assertRaisesRegex(SurvivorDataError, "already used its one rebuy"):
            record_pick(
                self.paths.picks,
                PickRecord(
                    2026, "small_pool", "mine", 3, "SF", "latest", "pending", True
                ),
            )

    def test_score_sync_refuses_to_create_an_invalid_rebuy_history(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now"),
        )
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 2, "KC", "later"),
        )
        events = [{
            "away_team": "Buffalo Bills",
            "home_team": "New York Jets",
            "completed": True,
            "scores": [
                {"name": "Buffalo Bills", "score": "10"},
                {"name": "New York Jets", "score": "20"},
            ],
        }]
        with self.assertRaisesRegex(SurvivorDataError, "Mark the next pick as a rebuy"):
            sync_results(
                self.paths.picks,
                season=2026,
                schedule=load_schedule(self.paths.schedule(2026), season=2026),
                events=events,
            )
        picks = load_picks(self.paths.picks)
        self.assertEqual([pick.result for pick in picks], ["pending", "pending"])

    def test_odds_client_is_injectable_and_never_needs_hardcoded_key(self):
        seen_url = None

        def opener(request, timeout):
            nonlocal seen_url
            seen_url = request.full_url
            return FakeHTTPResponse([], {"x-requests-remaining": "499"})

        response = OddsAPIClient("secret", opener=opener).fetch_odds()
        self.assertEqual(response.remaining_requests, "499")
        self.assertIn("apiKey=secret", seen_url)

    def test_refresh_odds_aggregates_books_and_writes_normalized_file(self):
        events = [
            {
                "away_team": "Buffalo Bills",
                "home_team": "New York Jets",
                "commence_time": "2026-09-13T17:00:00Z",
                "bookmakers": [
                    {
                        "key": "book_a",
                        "markets": [{
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -160},
                                {"name": "New York Jets", "price": 140},
                            ],
                        }],
                    },
                    {
                        "key": "book_b",
                        "markets": [{
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -150},
                                {"name": "New York Jets", "price": 135},
                            ],
                        }],
                    },
                ],
            },
            {
                "away_team": "Kansas City Chiefs",
                "home_team": "Las Vegas Raiders",
                "commence_time": "2026-09-13T20:00:00Z",
                "bookmakers": [{
                    "key": "book_a",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Kansas City Chiefs", "price": -200},
                            {"name": "Las Vegas Raiders", "price": 175},
                        ],
                    }],
                }],
            },
        ]

        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse(events, remaining_requests="498")

        result = refresh_odds(self.paths, season=2026, weeks={1}, client=Client())
        self.assertEqual(result.api_records, 2)
        self.assertEqual(result.manual_records, 0)
        stored = load_normalized_odds(self.paths.normalized_odds, season=2026)
        self.assertEqual(len(stored), 2)
        self.assertAlmostEqual(stored[0].probability_a + stored[0].probability_b, 1.0)

    def test_coverage_check_reports_actual_weeks_and_books(self):
        event = {
            "away_team": "Buffalo Bills",
            "home_team": "New York Jets",
            "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{"key": "book_a", "markets": []}],
        }

        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse(
                    [event], remaining_requests="498", used_requests="2"
                )

        coverage = inspect_odds_coverage(
            self.paths, season=2026, client=Client()
        )
        self.assertEqual(coverage.events_returned, 1)
        self.assertEqual(coverage.events_matched_schedule, 1)
        self.assertEqual(coverage.weeks, (1,))
        self.assertEqual(coverage.bookmakers, ("book_a",))
        self.assertEqual(coverage.remaining_requests, "498")
        self.assertEqual(
            [(game.week, game.away_team, game.home_team) for game in coverage.missing_games],
            [(1, "KC", "LV")],
        )

    def test_manual_odds_are_consulted_only_to_fill_api_gap(self):
        with self.paths.manual_odds.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "season", "week", "game_id", "team_a", "team_b", "moneyline_a",
                    "moneyline_b", "probability_a", "probability_b", "source", "observed_at",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "season": 2026,
                    "week": 1,
                    "game_id": "w1_kc_lv",
                    "team_a": "KC",
                    "team_b": "LV",
                    "probability_a": 0.7,
                    "source": "manual_after_warning",
                    "observed_at": "2026-09-13T11:00:00+00:00",
                }
            )
        api_event = {
            "away_team": "Buffalo Bills",
            "home_team": "New York Jets",
            "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{
                "key": "book",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": -160},
                        {"name": "New York Jets", "price": 140},
                    ],
                }],
            }],
        }

        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse([api_event])

        result = refresh_odds(
            self.paths,
            season=2026,
            weeks={1},
            client=Client(),
            as_of=datetime(2026, 9, 13, 12, tzinfo=timezone.utc),
        )
        self.assertEqual(result.api_records, 1)
        self.assertEqual(result.manual_records, 1)
        sources = {record.source for record in result.records}
        self.assertEqual(sources, {"the_odds_api", "manual_after_warning"})

    def test_stale_manual_odds_stop_before_normalized_file_changes(self):
        self.paths.normalized_odds.write_text("sentinel\n", encoding="utf-8")
        update_manual_odds(
            self.paths,
            season=2026,
            game_id="w1_buf_nyj",
            moneyline_a=-160,
            moneyline_b=140,
            source="test",
            observed_at="2026-09-10T12:00:00+00:00",
        )

        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse([])

        with self.assertRaisesRegex(StaleManualOddsError, "BUF.*NYJ.*48.0 hours"):
            refresh_odds(
                self.paths,
                season=2026,
                weeks={1},
                client=Client(),
                as_of=datetime(2026, 9, 12, 12, tzinfo=timezone.utc),
            )
        self.assertEqual(self.paths.normalized_odds.read_text(encoding="utf-8"), "sentinel\n")

    def test_api_line_supersedes_stale_manual_line(self):
        update_manual_odds(
            self.paths,
            season=2026,
            game_id="w1_buf_nyj",
            moneyline_a=-110,
            moneyline_b=-110,
            source="old",
            observed_at="2025-01-01T00:00:00+00:00",
        )
        api_event = {
            "away_team": "Buffalo Bills",
            "home_team": "New York Jets",
            "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{
                "key": "book",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": -160},
                        {"name": "New York Jets", "price": 140},
                    ],
                }],
            }],
        }
        second_api_event = {
            "away_team": "Kansas City Chiefs",
            "home_team": "Las Vegas Raiders",
            "commence_time": "2026-09-13T20:00:00Z",
            "bookmakers": [{
                "key": "book",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -200},
                        {"name": "Las Vegas Raiders", "price": 175},
                    ],
                }],
            }],
        }

        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse([api_event, second_api_event])

        result = refresh_odds(self.paths, season=2026, weeks={1}, client=Client())
        self.assertEqual(result.manual_records, 0)
        self.assertEqual({record.source for record in result.records}, {"the_odds_api"})

    def test_manual_update_and_confirmation_preserve_prices(self):
        saved = update_manual_odds(
            self.paths,
            season=2026,
            game_id="w1_kc_lv",
            moneyline_a=-200,
            moneyline_b=175,
            source="example",
            observed_at="2026-09-10T12:00:00+00:00",
        )
        confirmed = confirm_manual_odds(
            self.paths,
            season=2026,
            game_ids=["w1_kc_lv"],
            confirmed_at="2026-09-11T12:00:00+00:00",
        )[0]
        self.assertEqual((confirmed.moneyline_a, confirmed.moneyline_b), (-200.0, 175.0))
        self.assertAlmostEqual(confirmed.probability_a, saved.probability_a)
        self.assertEqual(confirmed.source, "example")
        self.assertEqual(confirmed.observed_at, "2026-09-11T12:00:00+00:00")

    def test_apply_manual_odds_updates_only_requested_normalized_game(self):
        self.paths.normalized_odds.write_text(
            "season,week,game_id,team_a,team_b,probability_a,probability_b,"
            "moneyline_a,moneyline_b,bookmakers,source,observed_at\n"
            "2026,1,w1_buf_nyj,BUF,NYJ,0.6,0.4,-150,130,api,the_odds_api,old\n",
            encoding="utf-8",
        )
        update_manual_odds(
            self.paths,
            season=2026,
            game_id="w1_kc_lv",
            moneyline_a=-200,
            moneyline_b=175,
            source="example",
            observed_at="2026-09-10T12:00:00+00:00",
        )

        applied = apply_manual_odds(
            self.paths,
            season=2026,
            game_ids=["w1_kc_lv"],
            as_of=datetime(2026, 9, 10, 13, tzinfo=timezone.utc),
        )
        stored = load_normalized_odds(self.paths.normalized_odds, season=2026)

        self.assertEqual([record.game_id for record in applied], ["w1_kc_lv"])
        self.assertEqual({record.game_id for record in stored}, {"w1_buf_nyj", "w1_kc_lv"})
        manual = next(record for record in stored if record.game_id == "w1_kc_lv")
        self.assertEqual(manual.bookmakers, "manual")
        self.assertEqual((manual.moneyline_a, manual.moneyline_b), (-200.0, 175.0))

    def test_missing_odds_raise_actionable_error(self):
        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse([])

        with self.assertRaisesRegex(
            MissingOddsError,
            "Automatic odds refresh.*Week 1: BUF at NYJ.*Week 1: KC at LV.*inputs/manual_odds.csv",
        ):
            refresh_odds(self.paths, season=2026, weeks={1}, client=Client())

    def test_pool_status_persists_rebuy_and_field_counts(self):
        record = PoolStatusRecord(
            season=2026,
            contest="small_pool",
            week=2,
            snapshot_phase="finalized_for_week",
            entries_alive=38,
            entries_with_unused_rebuy=34,
            entries_eligible_to_rebuy_this_week=4,
            rebuys_entered_this_week=2,
            total_rebuys_to_date=2,
            recorded_at="2026-09-17T12:00:00+00:00",
        )
        record_pool_status(self.paths.pool_status, record)
        self.assertEqual(load_pool_status(self.paths.pool_status), [record])
        with self.assertRaisesRegex(SurvivorDataError, "already exists"):
            record_pool_status(self.paths.pool_status, record)

    def test_pool_status_load_rejects_hand_edited_duplicate_snapshot(self):
        record = PoolStatusRecord(
            season=2026,
            contest="small_pool",
            week=2,
            snapshot_phase="finalized_for_week",
            entries_alive=38,
            entries_with_unused_rebuy=34,
            entries_eligible_to_rebuy_this_week=4,
            rebuys_entered_this_week=2,
            total_rebuys_to_date=2,
            recorded_at="2026-09-17T12:00:00+00:00",
        )
        record_pool_status(self.paths.pool_status, record)
        with self.paths.pool_status.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                (
                    record.season,
                    record.contest,
                    record.week,
                    record.snapshot_phase,
                    record.entries_alive,
                    record.entries_with_unused_rebuy,
                    record.entries_eligible_to_rebuy_this_week,
                    record.rebuys_entered_this_week,
                    record.total_rebuys_to_date,
                    record.recorded_at,
                )
            )

        with self.assertRaisesRegex(SurvivorDataError, "Duplicate pool-status"):
            load_pool_status(self.paths.pool_status)

    def test_large_pool_status_rejects_rebuy_counts(self):
        with self.assertRaisesRegex(SurvivorDataError, "does not allow rebuy"):
            record_pool_status(
                self.paths.pool_status,
                PoolStatusRecord(
                    season=2026,
                    contest="large_pool",
                    week=1,
                    snapshot_phase="finalized_for_week",
                    entries_alive=2000,
                    entries_with_unused_rebuy=0,
                    entries_eligible_to_rebuy_this_week=0,
                    rebuys_entered_this_week=1,
                    total_rebuys_to_date=1,
                    recorded_at="2026-09-10T12:00:00+00:00",
                ),
            )

    def test_score_sync_marks_tie_as_tie_and_entry_is_eliminated(self):
        record_pick(
            self.paths.picks,
            PickRecord(2026, "small_pool", "mine", 1, "BUF", "now"),
        )
        events = [{
            "away_team": "Buffalo Bills",
            "home_team": "New York Jets",
            "completed": True,
            "scores": [
                {"name": "Buffalo Bills", "score": "20"},
                {"name": "New York Jets", "score": "20"},
            ],
        }]
        updated = sync_results(
            self.paths.picks,
            season=2026,
            schedule=load_schedule(self.paths.schedule(2026), season=2026),
            events=events,
        )
        self.assertEqual(updated, 1)
        picks = load_picks(self.paths.picks)
        self.assertEqual(picks[0].result, "tie")
        self.assertFalse(entry_states(picks, season=2026)[0].active)


if __name__ == "__main__":
    unittest.main()
