"""Rerun recommendations from saved inputs without fetching live odds."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from data import OddsRecord, PickRecord, entry_states


def replay_report(report):
    """Use saved probabilities and pick history; never fetch odds or credentials."""
    from optimizer_single import solve_single_entry
    from survivor import _small_payload, _large_payload
    # Keep only ordinary decision fields, including when rerunning older reports.
    fields = ("run_id", "generated_at", "season", "pool", "contest", "entries",
              "first_week", "end_week", "pick_history", "probability_snapshot",
              "optimizer_parameters", "input_description", "odds_request",
              "manual_max_age_hours", "odds_refresh", "recommendation")
    frozen = {key: deepcopy(report[key]) for key in fields if key in report}
    first, last = int(report["first_week"]), int(report["end_week"])
    if not 1 <= first <= last <= 18:
        raise ValueError("Invalid decision horizon")
    probabilities, game_ids, opponents = {}, {}, {}
    seen = set()
    for row in report["probability_snapshot"]:
        record = OddsRecord(**row)
        if record.season != report["season"] or not first <= record.week <= last:
            raise ValueError("Snapshot contains a game outside its season/horizon")
        if not 0 < record.probability_a < 1 or not 0 < record.probability_b < 1:
            raise ValueError("Invalid probabilities")
        if abs(record.probability_a + record.probability_b - 1) > 1e-9:
            raise ValueError("Game probabilities must sum to one")
        for team, other, prob in ((record.team_a, record.team_b, record.probability_a),
                                   (record.team_b, record.team_a, record.probability_b)):
            key = (record.week, team)
            if key in seen:
                raise ValueError("Duplicate team in snapshot week")
            seen.add(key)
            probabilities.setdefault(record.week, {})[team] = prob
            game_ids.setdefault(record.week, {})[team] = record.game_id
            opponents.setdefault(record.week, {})[team] = other
    if set(probabilities) != set(range(first, last + 1)):
        raise ValueError("Incomplete saved horizon")
    history = [PickRecord(**row) for row in report["pick_history"]]
    if any(pick.season != report["season"] or pick.contest != report["contest"]
           or pick.week >= first for pick in history):
        raise ValueError("Pick history must precede this decision in the same contest")
    ids = report["entries"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Entry IDs must be distinct")
    params = report["optimizer_parameters"]
    used = {entry: {pick.team for pick in history if pick.entry_id == entry} for entry in ids}
    states = {state.entry_id: state for state in entry_states(history, season=report["season"])}
    if report["pool"] == "small":
        if len(ids) != 1:
            raise ValueError("Single-entry mode requires one entry")
        result = solve_single_entry(probabilities, used_teams=used[ids[0]],
                                    alternative_limit=params["top_alternatives"])
        frozen["recommendation"] = _small_payload(result, first)
    elif report["pool"] == "large":
        if not 1 <= len(ids) <= 3:
            raise ValueError("Portfolio mode supports one to three entries")
        if params.get("search_method", "candidate") == "global":
            from optimizer_global import search_global_portfolios
            active = [entry for entry in ids if entry not in states or states[entry].active]
            # Reproduce a wall-clock run by its actual deterministic node budget.
            node_budget = report.get("recommendation", {}).get("visited_nodes", params["global_max_nodes"])
            result = search_global_portfolios(
                active, probabilities, game_ids, opponents, used_teams_by_entry={entry: used[entry] for entry in active},
                fixed_other_survivors=params["other_survivors"],
                sensitivity_other_survivors=params["sensitivity"],
                max_nodes=node_budget or None, max_seconds=None,
                absolute_gap_tolerance=params["global_absolute_gap"])
            frozen["recommendation"] = _large_payload(result, first)
            frozen["execution"] = {"elapsed_seconds": result.elapsed_seconds,
                                   "replay_node_budget": node_budget}
        else:
            result = _replay_candidates(ids, states, used, probabilities, game_ids, opponents, params)
            frozen["recommendation"] = _large_payload(result, first)
    else:
        raise ValueError("Unknown pool mode")
    frozen["replay"] = {"replayed_at": datetime.now(timezone.utc).isoformat(),
                        "mode": "saved_inputs_offline"}
    return frozen


def _replay_candidates(ids, states, used, probabilities, game_ids, opponents, params):
    from optimizer_portfolio import EntryPath, generate_conditional_candidate_paths, search_candidate_portfolios
    candidates = []
    cache = {}
    for entry in ids:
        if entry in states and not states[entry].active:
            continue
        key = frozenset(used[entry])
        if key not in cache:
            cache[key] = generate_conditional_candidate_paths(
                entry, probabilities, game_ids, opponents, used_teams=key,
                candidate_limit=params["candidate_paths"], beam_width=params["beam_width"],
                paths_per_current_team=params["paths_per_current_team"])
        candidates.append(tuple(EntryPath(entry, path.picks) for path in cache[key]))
    result = search_candidate_portfolios(
        candidates, objective="expected_prize_share", fixed_other_survivors=params["other_survivors"],
        sensitivity_other_survivors=params["sensitivity"], max_evaluations=params["max_evaluations"],
        seed=params["seed"])
    return result
