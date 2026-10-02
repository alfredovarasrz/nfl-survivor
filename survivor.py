"""Guided command-line interface for the Survivor project.

Run ``python survivor.py --help`` to see the available guided commands.
The optimizer libraries remain separate, while this file connects them to the
saved schedule, odds, and pick history.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from data import (
    DataPaths,
    ESPNScheduleClient,
    OddsAPIClient,
    PickRecord,
    PoolStatusRecord,
    SurvivorDataError,
    apply_manual_odds,
    confirm_manual_odds,
    configure_odds_api_key_in_keychain,
    download_schedule,
    import_schedule,
    initialize_project,
    inspect_odds_coverage,
    load_schedule,
    load_manual_odds,
    load_normalized_odds,
    load_picks,
    project_status,
    record_pick,
    record_pool_status,
    record_result,
    refresh_odds,
    sync_results,
    update_manual_odds,
    utc_now,
    validate_pick_against_schedule,
)


ROOT = Path(__file__).resolve().parent


def _required(value: str | None, prompt: str) -> str:
    if value:
        return value
    response = input(f"{prompt}: ").strip()
    if not response:
        raise SurvivorDataError(f"{prompt} is required")
    return response


def _integer(value: int | None, prompt: str) -> int:
    if value is not None:
        return value
    try:
        return int(_required(None, prompt))
    except ValueError as exc:
        raise SurvivorDataError(f"{prompt} must be an integer") from exc


def _manual_prices(record) -> str:
    if record.moneyline_a is not None and record.moneyline_b is not None:
        return f"{record.team_a} {record.moneyline_a:+g} vs {record.team_b} {record.moneyline_b:+g}"
    return (
        f"{record.team_a} {record.probability_a:.4%} vs "
        f"{record.team_b} {record.probability_b:.4%}"
    )


def command_setup(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    if args.schedule:
        games = import_schedule(
            Path(args.schedule), paths.schedule(args.season), season=args.season, replace=args.replace
        )
        print(f"Stored and validated {len(games)} games for the {args.season} season.")
    else:
        print(f"Created the {args.season} project files.")
        print(
            f"Run download-schedule --season {args.season}, or import a schedule CSV, "
            "before refreshing odds."
        )


def command_import_schedule(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    games = import_schedule(
        Path(args.file), paths.schedule(args.season), season=args.season, replace=args.replace
    )
    print(f"Stored and validated {len(games)} games in {paths.schedule(args.season)}.")


def command_download_schedule(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    games = download_schedule(
        paths,
        season=args.season,
        client=ESPNScheduleClient(),
        replace=args.replace,
    )
    print(f"Downloaded and validated {len(games)} games for the {args.season} season.")
    print("The stored schedule will be reused; weekly commands refresh odds only.")


def command_configure_api_key(args: argparse.Namespace, paths: DataPaths) -> None:
    del args, paths
    print("macOS Keychain will now ask for your Odds API key.")
    configure_odds_api_key_in_keychain()
    print("Saved the key in macOS Keychain. It was not written to a project file.")


def command_check_odds_coverage(args: argparse.Namespace, paths: DataPaths) -> None:
    coverage = inspect_odds_coverage(
        paths,
        season=args.season,
        client=OddsAPIClient(),
        regions=args.regions,
        bookmakers=args.bookmakers,
    )
    print(f"Events returned: {coverage.events_returned}")
    print(f"Events matched to the stored schedule: {coverage.events_matched_schedule}")
    print(f"Schedule weeks covered: {', '.join(map(str, coverage.weeks)) or 'none'}")
    print(f"Bookmakers returned ({len(coverage.bookmakers)}):")
    for bookmaker in coverage.bookmakers:
        print(f"  {bookmaker}")
    if coverage.missing_games:
        print(f"Schedule games missing from the API response ({len(coverage.missing_games)}):")
        for game in coverage.missing_games:
            print(
                f"  Week {game.week}: {game.away_team} at {game.home_team} "
                f"({game.game_id})"
            )
    else:
        print("Schedule games missing from the API response: none")
    if coverage.remaining_requests is not None:
        print(f"API requests remaining: {coverage.remaining_requests}")
    if coverage.used_requests is not None:
        print(f"API requests used: {coverage.used_requests}")
    if set(coverage.weeks) != set(range(1, 19)):
        print(
            "The live API response does not cover all 18 weeks. "
            "The future-odds source is still required."
        )


def command_record_pick(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    contest = _required(args.contest, "Contest")
    entry = _required(args.entry, "Entry")
    week = _integer(args.week, "Week")
    team = _required(args.team, "Team actually submitted").upper()
    if args.rebuy and contest == "large_pool":
        raise SurvivorDataError("The large pool does not allow rebuys.")
    record = PickRecord(
        season=args.season,
        contest=contest,
        entry_id=entry,
        week=week,
        team=team,
        submitted_at=args.submitted_at or utc_now(),
        result="pending",
        rebuy=args.rebuy,
    )
    validate_pick_against_schedule(
        record, load_schedule(paths.schedule(args.season), season=args.season)
    )
    record_pick(paths.picks, record, replace=args.replace)
    suffix = " as the rebuy pick" if args.rebuy else ""
    print(f"Recorded {team} for {contest}/{entry}, Week {week}{suffix}.")


def command_record_result(args: argparse.Namespace, paths: DataPaths) -> None:
    updated = record_result(
        paths.picks,
        season=args.season,
        contest=_required(args.contest, "Contest"),
        entry_id=_required(args.entry, "Entry"),
        week=_integer(args.week, "Week"),
        result=_required(args.result, "Result (win, loss, tie, or pending)"),
    )
    status = "eliminated" if updated.result in {"loss", "tie"} else updated.result
    print(f"Recorded Week {updated.week} as {updated.result}; entry status is {status}.")


def command_record_pool_status(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    contest = _required(args.contest, "Contest")
    entries_alive = _integer(args.entries_alive, "Entries eligible to pick this week")
    if contest == "large_pool":
        supplied_rebuy_values = (
            args.entries_with_unused_rebuy,
            args.entries_eligible_to_rebuy_this_week,
            args.rebuys_entered_this_week,
            args.total_rebuys_to_date,
        )
        if any(value not in (None, 0) for value in supplied_rebuy_values):
            raise SurvivorDataError("The large pool does not allow rebuy counts")
        unused_rebuy = eligible_to_rebuy = rebuys_this_week = total_rebuys = 0
    else:
        unused_rebuy = _integer(
            args.entries_with_unused_rebuy,
            "Active entries that still hold an unused rebuy",
        )
        eligible_to_rebuy = _integer(
            args.entries_eligible_to_rebuy_this_week,
            "Entries eliminated last week and eligible to rebuy now",
        )
        rebuys_this_week = _integer(
            args.rebuys_entered_this_week,
            "Rebuys entering this week",
        )
        total_rebuys = _integer(
            args.total_rebuys_to_date,
            "Total rebuys paid through this week",
        )
    record = PoolStatusRecord(
        season=args.season,
        contest=contest,
        week=_integer(args.week, "Week"),
        snapshot_phase=args.phase,
        entries_alive=entries_alive,
        entries_with_unused_rebuy=unused_rebuy,
        entries_eligible_to_rebuy_this_week=eligible_to_rebuy,
        rebuys_entered_this_week=rebuys_this_week,
        total_rebuys_to_date=total_rebuys,
        recorded_at=utc_now(),
    )
    record_pool_status(paths.pool_status, record, replace=args.replace)
    print(
        f"Recorded {contest} Week {record.week}: {record.entries_alive} entries eligible "
        f"to pick ({record.snapshot_phase})."
    )


def command_refresh_odds(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    weeks = set(args.week or [_integer(None, "Week to refresh")])
    result = refresh_odds(
        paths,
        season=args.season,
        weeks=weeks,
        client=OddsAPIClient(),
        regions=args.regions,
        bookmakers=args.bookmakers,
        manual_max_age_hours=args.manual_max_age_hours,
    )
    print(
        f"Stored {len(result.records)} games: {result.api_records} from The Odds API and "
        f"{result.manual_records} from manual fallback."
    )
    if result.api_error:
        print(f"API warning: {result.api_error}")
    if result.remaining_requests is not None:
        print(f"API requests remaining: {result.remaining_requests}")
    if result.raw_response_path:
        print(f"Raw response: {result.raw_response_path}")
    for record in result.manual_games:
        print(
            f"Manual fallback verified {record.observed_at}: Week {record.week} "
            f"{_manual_prices(record)} ({record.source})"
        )


def command_update_manual_odds(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    record = update_manual_odds(
        paths,
        season=args.season,
        game_id=_required(args.game_id, "Game ID"),
        moneyline_a=float(_required(args.away_moneyline, "Away-team American moneyline")),
        moneyline_b=float(_required(args.home_moneyline, "Home-team American moneyline")),
        source=_required(args.source, "Odds source"),
    )
    print(
        f"Saved Week {record.week} {_manual_prices(record)}. Verified {record.observed_at}."
    )


def command_confirm_manual_odds(args: argparse.Namespace, paths: DataPaths) -> None:
    initialize_project(paths, args.season)
    schedule = load_schedule(paths.schedule(args.season), season=args.season)
    records = load_manual_odds(
        paths.manual_odds,
        schedule,
        season=args.season,
        weeks=set(range(1, 19)),
    )
    by_id = {record.game_id: record for record in records}
    requested = tuple(dict.fromkeys(args.game_id))
    missing = [game_id for game_id in requested if game_id not in by_id]
    if missing:
        raise SurvivorDataError(
            "No saved manual odds exist for game ID(s): " + ", ".join(missing)
        )
    print("Review the saved manual lines:")
    for game_id in requested:
        record = by_id[game_id]
        print(
            f"  Week {record.week}: {_manual_prices(record)}; source {record.source}; "
            f"last verified {record.observed_at or 'never'}"
        )
    if not args.yes:
        answer = input("Are these prices still current? Type yes to confirm: ").strip().lower()
        if answer != "yes":
            print("No timestamps were changed.")
            return
    updated = confirm_manual_odds(
        paths,
        season=args.season,
        game_ids=requested,
    )
    print(f"Confirmed {len(updated)} manual line(s) at {updated[0].observed_at}.")


def command_apply_manual_odds(args: argparse.Namespace, paths: DataPaths) -> None:
    records = apply_manual_odds(
        paths,
        season=args.season,
        game_ids=args.game_id,
        max_age_hours=args.manual_max_age_hours,
    )
    for record in records:
        print(
            f"Applied Week {record.week} {_manual_prices(record)} to the normalized "
            f"look-ahead snapshot ({record.source}, verified {record.observed_at})."
        )


def command_sync_results(args: argparse.Namespace, paths: DataPaths) -> None:
    client = OddsAPIClient()
    response = client.fetch_scores(days_from=args.days_from)
    schedule = load_schedule(paths.schedule(args.season), season=args.season)
    count = sync_results(paths.picks, season=args.season, schedule=schedule, events=response.events)
    print(f"Updated {count} pending pick result(s). Ties count as losses for entry status.")


def command_status(args: argparse.Namespace, paths: DataPaths) -> None:
    state = project_status(paths, season=args.season)
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    print(f"Season {state['season']}, current Week {state['current_week']}")
    print(
        f"Schedule: {state['schedule_games']} games across Weeks "
        f"{', '.join(map(str, state['schedule_weeks']))}"
    )
    print(f"Odds available for Weeks: {', '.join(map(str, state['odds_weeks'])) or 'none'}")
    print(f"Latest odds timestamp: {state['latest_odds'] or 'none'}")
    if not state["entries"]:
        print("No picks have been recorded.")
    for entry in state["entries"]:
        active = "active" if entry["active"] else "eliminated"
        teams = ", ".join(entry["used_teams"])
        print(
            f"{entry['contest']}/{entry['entry_id']}: {active}; used {teams}; "
            f"rebuys used {entry['rebuys_used']}"
        )


def _small_payload(result, first_week: int) -> dict:
    return {
        "objective": "expected_weeks_survived",
        "first_week": first_week,
        "expected_remaining_weeks": result.expected_weeks,
        "picks": result.picks,
        "survival_by_week": result.survival_by_week,
        "solver_status": result.solver_status,
        "optimality_gap": result.optimality_gap,
        "current_week_alternatives": [asdict(item) for item in result.current_pick_alternatives],
    }


def _print_small_result(result, first_week: int, top: int) -> None:
    print(f"Recommended Week {first_week} pick: {result.picks[first_week]}")
    print(f"Expected remaining weeks survived: {result.expected_weeks:.6f}")
    print(f"Solver status: {result.solver_status}; relative optimality gap: {result.optimality_gap}")
    print("Provisional full path:")
    for week, team in result.picks.items():
        print(
            f"  Week {week}: {team} "
            f"(survive through week: {result.survival_by_week[week]:.4%})"
        )
    print("Best current-week alternatives:")
    for alternative in result.current_pick_alternatives[:top]:
        print(
            f"  {alternative.team}: {alternative.expected_weeks:.6f} expected weeks "
            f"({alternative.expected_weeks_lost:.6f} below best)"
        )


def command_recommend_small(args: argparse.Namespace, paths: DataPaths) -> None:
    from workflow import next_week, recommend_single_entry

    contest = _required(args.contest, "Contest")
    entry = _required(args.entry, "Entry")
    first_week = args.week or next_week(
        paths, season=args.season, contest=contest, entry_ids=[entry]
    )
    try:
        result = recommend_single_entry(
            paths,
            season=args.season,
            contest=contest,
            entry_id=entry,
            first_week=first_week,
            last_week=args.end_week,
            alternative_limit=args.top,
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        raise SurvivorDataError(str(exc)) from exc

    payload = _small_payload(result, first_week)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    _print_small_result(result, first_week, args.top)


def _parse_sensitivity(value: str) -> tuple[int, ...]:
    try:
        numbers = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated whole numbers, such as 0,5,15.") from exc
    if not numbers or any(number < 0 for number in numbers):
        raise argparse.ArgumentTypeError("Sensitivity values must be non-negative.")
    return numbers


def command_recommend_large(args: argparse.Namespace, paths: DataPaths) -> None:
    from workflow import next_week, recommend_three_entry_portfolio

    contest = _required(args.contest, "Contest")
    entries = list(args.entry)
    first_week = args.week or next_week(
        paths, season=args.season, contest=contest, entry_ids=entries
    )
    try:
        result = recommend_three_entry_portfolio(
            paths,
            season=args.season,
            contest=contest,
            entry_ids=entries,
            first_week=first_week,
            last_week=args.end_week,
            other_survivors=args.other_survivors,
            sensitivity=args.sensitivity,
            max_evaluations=args.max_evaluations,
            seed=args.seed,
            candidate_limit=args.candidate_paths,
            beam_width=args.beam_width,
            paths_per_current_team=args.paths_per_current_team,
            search_method=args.search,
            global_max_nodes=args.global_max_nodes,
            global_max_seconds=args.global_max_seconds,
            global_absolute_gap=args.global_absolute_gap,
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        raise SurvivorDataError(str(exc)) from exc

    payload = _large_payload(result, first_week)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    _print_large_result(result, first_week, args.end_week)


def _large_payload(result, first_week: int) -> dict:
    payload = {
        "objective": result.objective,
        "first_week": first_week,
        "fixed_other_survivors": result.fixed_other_survivors,
        "expected_prize_share": result.objective_value,
        "selected_portfolio_share_sensitivity": dict(
            result.expected_share_sensitivity
        ),
        "probability_any_survives": result.metrics.probability_any_survives,
        "expected_final_survivors": result.metrics.expected_final_survivors,
        "method": result.method,
        "evaluated_portfolios": result.evaluated_portfolios,
        "total_candidate_portfolios": result.total_candidate_portfolios,
        "seed": result.seed,
        "candidate_generation": "conditional_beam_search",
        "exact_within_candidate_sets": result.exact_within_candidate_sets,
        "globally_optimal_over_all_legal_paths": result.globally_optimal_over_all_legal_paths,
        "assumptions": list(result.assumptions),
        "paths": {
            path.entry_id: {pick.week: pick.team for pick in path.picks}
            for path in result.paths
        },
    }

    if result.method == "global_branch_and_bound":
        for key in ("candidate_generation", "exact_within_candidate_sets", "total_candidate_portfolios", "seed"):
            payload.pop(key, None)
        payload.update({
            "search_scope": "all_legal_fixed_paths", "termination": result.termination,
            "global_lower_bound": result.lower_bound, "global_upper_bound": result.upper_bound,
            "global_absolute_gap": result.absolute_gap, "global_relative_gap": result.relative_gap,
            "visited_nodes": result.visited_nodes, "completed_portfolios": result.completed_portfolios,
            "absolute_gap_tolerance": result.absolute_gap_tolerance,
            "numerical_bound_slack": result.numerical_bound_slack,
        })
    return payload


def _print_large_result(result, first_week: int, end_week: int) -> None:
    print(f"Recommended Week {first_week} portfolio:")
    for path in result.paths:
        print(f"  {path.entry_id}: {path.picks[0].team}")
    print(f"Expected prize share with {result.fixed_other_survivors} other finalists: {result.objective_value:.6%}")
    print(f"Probability at least one owned entry reaches Week {end_week}: {result.metrics.probability_any_survives:.4%}")
    print(f"Expected owned entries surviving: {result.metrics.expected_final_survivors:.4f}")
    print("Selected-portfolio payoff sensitivity:")
    for field_size, share in sorted(result.expected_share_sensitivity.items()):
        print(f"  {field_size} other finalists: {share:.6%}")
    if result.method == "global_branch_and_bound":
        print(f"Global search status: {result.termination}")
        print(f"Global bounds: [{result.lower_bound:.12g}, {result.upper_bound:.12g}]; absolute gap {result.absolute_gap:.12g}")
        print(f"Nodes: {result.visited_nodes}; runtime: {result.elapsed_seconds:.3f}s; certified: {result.globally_optimal_over_all_legal_paths}")
        return
    if result.exact_within_candidate_sets:
        print(
            "Search status: every conditional beam-search candidate combination "
            "was evaluated exactly."
        )
    else:
        print(
            "Search status: sampled and locally improved within the conditional "
            "beam-search candidates; the candidate set was not exhausted."
        )
    print("Global status: no certificate over every possible three-path portfolio.")


def command_weekly_run(args: argparse.Namespace, paths: DataPaths) -> None:
    """Refresh the remaining horizon, optimize one pool, and save recommendation reports."""

    from reporting import save_weekly_reports
    from workflow import (
        aligned_active_entry_week,
        recommend_single_entry,
        recommend_three_entry_portfolio,
    )

    initialize_project(paths, args.season)
    if args.pool == "small":
        contest = args.contest or "small_pool"
        entries = list(args.entry or ["mine"])
        if len(entries) != 1:
            raise SurvivorDataError("The small weekly run requires exactly one entry ID.")
    else:
        contest = args.contest or "large_pool"
        entries = list(args.entry or ["entry_1", "entry_2", "entry_3"])
        if len(entries) != 3 or len(set(entries)) != 3:
            raise SurvivorDataError("The large weekly run requires exactly three distinct entry IDs.")
        if args.other_survivors is None:
            raise SurvivorDataError("The large weekly run requires --other-survivors.")

    expected_week = aligned_active_entry_week(
        paths,
        season=args.season,
        contest=contest,
        entry_ids=entries,
    )
    if args.week is not None and args.week != expected_week:
        raise SurvivorDataError(
            f"Requested Week {args.week}, but the saved active-entry history requires "
            f"Week {expected_week}."
        )
    first_week = expected_week
    if first_week < 1 or first_week > args.end_week or args.end_week > 18:
        raise SurvivorDataError(
            f"Decision Week {first_week} is outside the requested season horizon ending Week {args.end_week}."
        )

    refreshed = refresh_odds(
        paths,
        season=args.season,
        weeks=set(range(first_week, args.end_week + 1)),
        client=OddsAPIClient(),
        regions=args.regions,
        bookmakers=args.bookmakers,
        manual_max_age_hours=args.manual_max_age_hours,
    )

    try:
        if args.pool == "small":
            result = recommend_single_entry(
                paths,
                season=args.season,
                contest=contest,
                entry_id=entries[0],
                first_week=first_week,
                last_week=args.end_week,
                alternative_limit=args.top,
                manual_max_age_hours=args.manual_max_age_hours,
            )
            recommendation = _small_payload(result, first_week)
        else:
            result = recommend_three_entry_portfolio(
                paths,
                season=args.season,
                contest=contest,
                entry_ids=entries,
                first_week=first_week,
                last_week=args.end_week,
                other_survivors=args.other_survivors,
                sensitivity=args.sensitivity,
                max_evaluations=args.max_evaluations,
                seed=args.seed,
                candidate_limit=args.candidate_paths,
                beam_width=args.beam_width,
                paths_per_current_team=args.paths_per_current_team,
                search_method=args.search,
                global_max_nodes=args.global_max_nodes,
                global_max_seconds=args.global_max_seconds,
                global_absolute_gap=args.global_absolute_gap,
                manual_max_age_hours=args.manual_max_age_hours,
            )
            recommendation = _large_payload(result, first_week)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise SurvivorDataError(str(exc)) from exc

    probability_snapshot = [
        asdict(record)
        for record in load_normalized_odds(paths.normalized_odds, season=args.season)
        if first_week <= record.week <= args.end_week
    ]
    generated_at = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "season": args.season,
        "pool": args.pool,
        "contest": contest,
        "entries": entries,
        "pick_history": [
            asdict(pick)
            for pick in load_picks(paths.picks, season=args.season, contest=contest)
            if pick.entry_id in entries
        ],
        "first_week": first_week,
        "end_week": args.end_week,
        "manual_max_age_hours": args.manual_max_age_hours,
        "odds_request": {
            "regions": args.regions,
            "bookmakers": args.bookmakers,
        },
        "odds_refresh": {
            "api_records": refreshed.api_records,
            "manual_records": refreshed.manual_records,
            "manual_games": [asdict(record) for record in refreshed.manual_games],
            "api_warning": refreshed.api_error,
            "remaining_requests": refreshed.remaining_requests,
            "raw_response_path": str(refreshed.raw_response_path.relative_to(paths.root)) if refreshed.raw_response_path else None,
        },
        "probability_snapshot": probability_snapshot,
        "optimizer_parameters": {
            "other_survivors": args.other_survivors if args.pool == "large" else None,
            "sensitivity": list(args.sensitivity) if args.pool == "large" else None,
            "max_evaluations": args.max_evaluations if args.pool == "large" else None,
            "candidate_paths": args.candidate_paths if args.pool == "large" else None,
            "beam_width": args.beam_width if args.pool == "large" else None,
            "paths_per_current_team": args.paths_per_current_team if args.pool == "large" else None,
            "top_alternatives": args.top if args.pool == "small" else None,
            "seed": args.seed if args.pool == "large" and args.search == "candidate" else None,
            "search_method": args.search if args.pool == "large" else None,
            "global_max_nodes": args.global_max_nodes if args.pool == "large" else None,
            "global_max_seconds": args.global_max_seconds if args.pool == "large" else None,
            "global_absolute_gap": args.global_absolute_gap if args.pool == "large" else None,
        },
        "recommendation": recommendation,
    }
    if args.pool == "large" and args.search == "global":
        report["execution"] = {"elapsed_seconds": result.elapsed_seconds}
    json_path, markdown_path = save_weekly_reports(
        paths,
        report,
        season=args.season,
        week=first_week,
        contest=contest,
        run_id=run_id,
    )

    print(
        f"Refreshed Weeks {first_week}-{args.end_week}: {refreshed.api_records} API games, "
        f"{refreshed.manual_records} verified manual fallback games."
    )
    if args.pool == "small":
        _print_small_result(result, first_week, args.top)
    else:
        _print_large_result(result, first_week, args.end_week)
    print(f"Readable report: {markdown_path}")
    print(f"Complete JSON record: {json_path}")


def command_replay(args: argparse.Namespace, paths: DataPaths) -> None:
    from replay import replay_report
    from reporting import save_weekly_reports
    try:
        report = replay_report(json.loads(args.report.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise SurvivorDataError(f"Cannot replay report: {exc}") from None
    if args.save:
        saved = save_weekly_reports(paths, report, season=report['season'],
                                   week=report['first_week'], contest=report['contest'])
        print("Saved replay:", *(path.relative_to(paths.root) for path in saved))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NFL Survivor data and recommendation workflow")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup-season", help="Create a season's input files")
    setup.add_argument("--season", type=int, required=True)
    setup.add_argument("--schedule", help="Optional full-season schedule CSV to import")
    setup.add_argument("--replace", action="store_true", help="Replace a previously imported schedule")
    setup.set_defaults(handler=command_setup)

    schedule = subparsers.add_parser("import-schedule", help="Validate and store the season schedule once")
    schedule.add_argument("--season", type=int, required=True)
    schedule.add_argument("file")
    schedule.add_argument("--replace", action="store_true")
    schedule.set_defaults(handler=command_import_schedule)

    schedule_download = subparsers.add_parser(
        "download-schedule", help="Download and store the full NFL schedule once"
    )
    schedule_download.add_argument("--season", type=int, required=True)
    schedule_download.add_argument(
        "--replace", action="store_true", help="Correct a stored schedule deliberately"
    )
    schedule_download.set_defaults(handler=command_download_schedule)

    key_setup = subparsers.add_parser(
        "configure-api-key", help="Store The Odds API key securely in macOS Keychain"
    )
    key_setup.set_defaults(handler=command_configure_api_key)

    coverage = subparsers.add_parser(
        "check-odds-coverage",
        help="Report actual NFL weeks and bookmakers returned by The Odds API",
    )
    coverage.add_argument("--season", type=int, required=True)
    coverage.add_argument("--regions", default="us")
    coverage.add_argument("--bookmakers", help="Comma-separated bookmaker keys")
    coverage.set_defaults(handler=command_check_odds_coverage)

    pick = subparsers.add_parser("record-pick", help="Record the pick actually submitted")
    pick.add_argument("--season", type=int, required=True)
    pick.add_argument("--contest")
    pick.add_argument("--entry")
    pick.add_argument("--week", type=int)
    pick.add_argument("--team")
    pick.add_argument("--submitted-at")
    pick.add_argument("--rebuy", action="store_true", help="Mark this as the one allowed rebuy pick")
    pick.add_argument("--replace", action="store_true", help="Correct an existing pick for this week")
    pick.set_defaults(handler=command_record_pick)

    result = subparsers.add_parser("record-result", help="Manually record or correct one pick result")
    result.add_argument("--season", type=int, required=True)
    result.add_argument("--contest")
    result.add_argument("--entry")
    result.add_argument("--week", type=int)
    result.add_argument("--result", choices=("pending", "win", "loss", "tie"))
    result.set_defaults(handler=command_record_result)

    pool_status = subparsers.add_parser(
        "record-pool-status",
        help="Save the number of entries eligible to pick in a pool week",
    )
    pool_status.add_argument("--season", type=int, required=True)
    pool_status.add_argument("--contest", choices=("small_pool", "large_pool"))
    pool_status.add_argument("--week", type=int)
    pool_status.add_argument(
        "--phase",
        choices=("before_rebuy_deadline", "finalized_for_week"),
        default="finalized_for_week",
    )
    pool_status.add_argument(
        "--entries-alive",
        type=int,
        help="Entries eligible to submit a pick after committed rebuys",
    )
    pool_status.add_argument(
        "--entries-with-unused-rebuy",
        type=int,
        help="Active small-pool entries that could rebuy after a future loss",
    )
    pool_status.add_argument(
        "--entries-eligible-to-rebuy-this-week",
        type=int,
        help="Small-pool entries eliminated last week that can rebuy now",
    )
    pool_status.add_argument(
        "--rebuys-entered-this-week",
        type=int,
        help="Small-pool rebuys already committed for this week",
    )
    pool_status.add_argument(
        "--total-rebuys-to-date",
        type=int,
        help="Cumulative small-pool rebuy payments through this week",
    )
    pool_status.add_argument("--replace", action="store_true")
    pool_status.set_defaults(handler=command_record_pool_status)

    odds = subparsers.add_parser("refresh-odds", help="Refresh selected weeks from The Odds API")
    odds.add_argument("--season", type=int, required=True)
    odds.add_argument("--week", type=int, action="append", help="Week to refresh; may be repeated")
    odds.add_argument("--regions", default="us")
    odds.add_argument("--bookmakers", help="Comma-separated bookmaker keys; overrides regions")
    odds.add_argument(
        "--manual-max-age-hours",
        type=float,
        default=24.0,
        help="Maximum age of a required manual fallback (default: 24)",
    )
    odds.set_defaults(handler=command_refresh_odds)

    manual_update = subparsers.add_parser(
        "update-manual-odds",
        help="Add or replace one API-missing game's manually verified moneylines",
    )
    manual_update.add_argument("--season", type=int, required=True)
    manual_update.add_argument(
        "--game-id",
        help="Game ID shown by the missing or stale odds message",
    )
    manual_update.add_argument(
        "--away-moneyline",
        type=float,
        help="American moneyline for the scheduled away team, such as 100",
    )
    manual_update.add_argument(
        "--home-moneyline",
        type=float,
        help="American moneyline for the scheduled home team, such as -131",
    )
    manual_update.add_argument("--source", help="Website or sportsbook checked")
    manual_update.set_defaults(handler=command_update_manual_odds)

    manual_confirm = subparsers.add_parser(
        "confirm-manual-odds",
        help="Confirm that saved manual lines are still current",
    )
    manual_confirm.add_argument("--season", type=int, required=True)
    manual_confirm.add_argument(
        "--game-id",
        action="append",
        required=True,
        help="Saved game ID; repeat to confirm several lines together",
    )
    manual_confirm.add_argument(
        "--yes",
        action="store_true",
        help="Confirm after reviewing the displayed lines without an extra prompt",
    )
    manual_confirm.set_defaults(handler=command_confirm_manual_odds)

    manual_apply = subparsers.add_parser(
        "apply-manual-odds",
        help="Apply verified manual games without replacing a saved future week",
    )
    manual_apply.add_argument("--season", type=int, required=True)
    manual_apply.add_argument(
        "--game-id",
        action="append",
        required=True,
        help="Verified manual game ID; repeat to apply several games",
    )
    manual_apply.add_argument(
        "--manual-max-age-hours",
        type=float,
        default=24.0,
    )
    manual_apply.set_defaults(handler=command_apply_manual_odds)

    sync = subparsers.add_parser("sync-results", help="Update pending picks from completed NFL scores")
    sync.add_argument("--season", type=int, required=True)
    sync.add_argument("--days-from", type=int, default=3)
    sync.set_defaults(handler=command_sync_results)

    status = subparsers.add_parser("status", help="Show the saved season and entry state")
    status.add_argument("--season", type=int, required=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=command_status)

    small = subparsers.add_parser(
        "recommend-small", help="Optimize the exact expected-survival pick for one entry"
    )
    small.add_argument("--season", type=int, required=True)
    small.add_argument("--contest", default="small_pool")
    small.add_argument("--entry", default="mine")
    small.add_argument("--week", type=int, help="Defaults to the week after the saved history")
    small.add_argument("--end-week", type=int, default=18)
    small.add_argument("--top", type=int, default=8)
    small.add_argument("--json", action="store_true")
    small.set_defaults(handler=command_recommend_small)

    large = subparsers.add_parser(
        "recommend-large", help="Optimize the expected prize share of three entries"
    )
    large.add_argument("--season", type=int, required=True)
    large.add_argument("--contest", default="large_pool")
    large.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Entry ID; repeat this option exactly three times",
    )
    large.add_argument("--week", type=int, help="Defaults to the week after the saved history")
    large.add_argument("--end-week", type=int, default=18)
    large.add_argument("--other-survivors", type=int, required=True)
    large.add_argument("--sensitivity", type=_parse_sensitivity, default=(0, 1, 5, 15))
    large.add_argument("--search", choices=("candidate", "global"), default="candidate")
    large.add_argument("--global-max-nodes", type=int, default=100_000, help="Global node limit; 0 removes the node limit")
    large.add_argument("--global-max-seconds", type=float, default=30.0, help="Global time limit; 0 disables for deterministic node-budget runs")
    large.add_argument("--global-absolute-gap", type=float, default=1e-10, help="Absolute prize-share gap tolerance for global certification")
    large.add_argument("--seed", type=int, default=0, help="Recorded random seed for sampled search")
    large.add_argument("--max-evaluations", type=int, default=300_000)
    large.add_argument(
        "--candidate-paths",
        type=int,
        default=32,
        help="Best conditional paths retained per distinct entry history",
    )
    large.add_argument(
        "--beam-width",
        type=int,
        default=300,
        help="Partial legal paths retained while generating each candidate",
    )
    large.add_argument(
        "--paths-per-current-team",
        type=int,
        default=2,
        help="Future-path variants retained behind each current-week team",
    )
    large.add_argument("--json", action="store_true")
    large.set_defaults(handler=command_recommend_large)

    weekly = subparsers.add_parser(
        "weekly-run",
        help="Refresh every remaining week, optimize one pool, and save reports",
    )
    weekly.add_argument("--season", type=int, required=True)
    weekly.add_argument("--pool", choices=("small", "large"), required=True)
    weekly.add_argument("--contest", help="Defaults to small_pool or large_pool")
    weekly.add_argument(
        "--entry",
        action="append",
        help="Entry ID; omit for mine (small) or entry_1 through entry_3 (large)",
    )
    weekly.add_argument("--week", type=int, help="Defaults to the week after saved history")
    weekly.add_argument("--end-week", type=int, default=18)
    weekly.add_argument("--regions", default="us")
    weekly.add_argument("--bookmakers", help="Comma-separated bookmaker keys; overrides regions")
    weekly.add_argument("--manual-max-age-hours", type=float, default=24.0)
    weekly.add_argument(
        "--top",
        type=int,
        default=1,
        help="Exact small-pool alternatives to solve (default: 1; higher values take longer)",
    )
    weekly.add_argument("--other-survivors", type=int)
    weekly.add_argument("--sensitivity", type=_parse_sensitivity, default=(0, 1, 5, 15))
    weekly.add_argument("--search", choices=("candidate", "global"), default="candidate")
    weekly.add_argument("--global-max-nodes", type=int, default=100_000, help="Global node limit; 0 removes the node limit")
    weekly.add_argument("--global-max-seconds", type=float, default=30.0, help="Global time limit; 0 disables for deterministic node-budget runs")
    weekly.add_argument("--global-absolute-gap", type=float, default=1e-10, help="Absolute prize-share gap tolerance for global certification")
    weekly.add_argument("--seed", type=int, default=0, help="Recorded random seed for sampled search")
    weekly.add_argument("--max-evaluations", type=int, default=300_000)
    weekly.add_argument("--candidate-paths", type=int, default=32)
    weekly.add_argument("--beam-width", type=int, default=300)
    weekly.add_argument("--paths-per-current-team", type=int, default=2)
    weekly.set_defaults(handler=command_weekly_run)
    replay = subparsers.add_parser("replay", help="Rerun a saved JSON recommendation offline")
    replay.add_argument("report", type=Path)
    replay.add_argument("--save", action="store_true", help="Save JSON and Markdown under --root")
    replay.set_defaults(handler=command_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = DataPaths(args.root.resolve())
    try:
        args.handler(args, paths)
    except SurvivorDataError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
