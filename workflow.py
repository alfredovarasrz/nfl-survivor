"""Application services connecting saved data to both optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from data import (
    DataPaths,
    MissingOddsError,
    OddsRecord,
    PickRecord,
    SurvivorDataError,
    entry_states,
    load_normalized_odds,
    load_picks,
    load_schedule,
    validate_manual_odds_freshness,
    validate_odds,
)


@dataclass(frozen=True)
class HorizonData:
    probabilities: dict[int, dict[str, float]]
    game_ids: dict[int, dict[str, str]]
    opponents: dict[int, dict[str, str]]


def load_horizon(
    paths: DataPaths,
    *,
    season: int,
    first_week: int,
    last_week: int = 18,
    manual_max_age_hours: float = 24.0,
    as_of: datetime | None = None,
) -> HorizonData:
    """Load a complete, validated optimization horizon from normalized odds."""

    if first_week < 1 or last_week > 18 or first_week > last_week:
        raise SurvivorDataError("Optimization weeks must fall between 1 and 18.")
    weeks = set(range(first_week, last_week + 1))
    schedule = load_schedule(paths.schedule(season), season=season)
    records = [
        record
        for record in load_normalized_odds(paths.normalized_odds, season=season)
        if record.week in weeks
    ]
    validate_manual_odds_freshness(
        [record for record in records if record.bookmakers == "manual"],
        max_age_hours=manual_max_age_hours,
        as_of=as_of,
    )
    try:
        validate_odds(records, schedule, season=season, weeks=weeks)
    except MissingOddsError as exc:
        raise SurvivorDataError(
            f"{exc} Refresh each missing week. If the automatic source still has "
            "no line, add only those games to inputs/manual_odds.csv and rerun."
        ) from exc

    probabilities: dict[int, dict[str, float]] = {}
    game_ids: dict[int, dict[str, str]] = {}
    opponents: dict[int, dict[str, str]] = {}
    for record in records:
        _add_game(record, probabilities, game_ids, opponents)
    missing_weeks = sorted(weeks - set(probabilities))
    if missing_weeks:
        raise SurvivorDataError(f"No game probabilities exist for Weeks {missing_weeks}.")
    return HorizonData(probabilities, game_ids, opponents)


def _add_game(
    record: OddsRecord,
    probabilities: dict[int, dict[str, float]],
    game_ids: dict[int, dict[str, str]],
    opponents: dict[int, dict[str, str]],
) -> None:
    probabilities.setdefault(record.week, {}).update(
        {record.team_a: record.probability_a, record.team_b: record.probability_b}
    )
    game_ids.setdefault(record.week, {}).update(
        {record.team_a: record.game_id, record.team_b: record.game_id}
    )
    opponents.setdefault(record.week, {}).update(
        {record.team_a: record.team_b, record.team_b: record.team_a}
    )


def entry_history(
    paths: DataPaths,
    *,
    season: int,
    contest: str,
    entry_id: str,
    before_week: int,
) -> list[PickRecord]:
    return [
        pick
        for pick in load_picks(paths.picks, season=season, contest=contest)
        if pick.entry_id == entry_id and pick.week < before_week
    ]


def next_week(
    paths: DataPaths,
    *,
    season: int,
    contest: str,
    entry_ids: list[str],
) -> int:
    picks = [
        pick
        for pick in load_picks(paths.picks, season=season, contest=contest)
        if pick.entry_id in entry_ids
    ]
    return max((pick.week for pick in picks), default=0) + 1


def aligned_active_entry_week(
    paths: DataPaths,
    *,
    season: int,
    contest: str,
    entry_ids: list[str],
) -> int:
    """Return the common next week for active entries after validating history."""

    picks = load_picks(paths.picks, season=season, contest=contest)
    states = {
        state.entry_id: state
        for state in entry_states(picks, season=season)
        if state.contest == contest
    }
    active_ids = [
        entry_id
        for entry_id in entry_ids
        if entry_id not in states or states[entry_id].active
    ]
    if not active_ids:
        raise SurvivorDataError(f"No active entries remain in {contest}.")

    next_by_entry: dict[str, int] = {}
    for entry_id in active_ids:
        weeks = sorted(pick.week for pick in picks if pick.entry_id == entry_id)
        expected_history = list(range(1, (weeks[-1] if weeks else 0) + 1))
        if weeks != expected_history:
            raise SurvivorDataError(
                f"{contest}/{entry_id} has a non-contiguous pick history: {weeks}. "
                "Correct the saved history before running the weekly optimizer."
            )
        next_by_entry[entry_id] = (weeks[-1] if weeks else 0) + 1

    decision_weeks = set(next_by_entry.values())
    if len(decision_weeks) != 1:
        details = ", ".join(
            f"{entry_id}=Week {week}" for entry_id, week in sorted(next_by_entry.items())
        )
        raise SurvivorDataError(
            "Active entries are not aligned on the same decision week: " + details
        )
    return decision_weeks.pop()


def recommend_single_entry(
    paths: DataPaths,
    *,
    season: int,
    contest: str,
    entry_id: str,
    first_week: int,
    last_week: int = 18,
    alternative_limit: int | None = None,
    manual_max_age_hours: float = 24.0,
    as_of: datetime | None = None,
):
    """Return the exact expected-survival solution for one entry."""

    from optimizer_single import solve_single_entry

    horizon = load_horizon(
        paths,
        season=season,
        first_week=first_week,
        last_week=last_week,
        manual_max_age_hours=manual_max_age_hours,
        as_of=as_of,
    )
    history = entry_history(
        paths,
        season=season,
        contest=contest,
        entry_id=entry_id,
        before_week=first_week,
    )
    used_teams = {pick.team for pick in history}
    return solve_single_entry(
        horizon.probabilities,
        used_teams=used_teams,
        alternative_limit=alternative_limit,
    )


def recommend_three_entry_portfolio(
    paths: DataPaths,
    *,
    season: int,
    contest: str,
    entry_ids: list[str],
    first_week: int,
    last_week: int = 18,
    other_survivors: int,
    sensitivity: tuple[int, ...] = (0, 1, 5, 15),
    max_evaluations: int = 300_000,
    seed: int = 0,
    candidate_limit: int = 32,
    beam_width: int = 300,
    paths_per_current_team: int = 2,
    search_method: str = "candidate",
    global_max_nodes: int | None = 100_000,
    global_max_seconds: float | None = 30.0,
    global_absolute_gap: float = 1e-10,
    manual_max_age_hours: float = 24.0,
    as_of: datetime | None = None,
):
    """Build conditional beam paths and optimize fixed-field expected share."""

    if len(entry_ids) != 3 or len(set(entry_ids)) != 3:
        raise SurvivorDataError("Exactly three distinct entry IDs are required.")
    if candidate_limit < 1:
        raise SurvivorDataError("candidate_limit must be positive.")
    if beam_width < 1:
        raise SurvivorDataError("beam_width must be positive.")
    if paths_per_current_team < 1:
        raise SurvivorDataError("paths_per_current_team must be positive.")

    from optimizer_portfolio import (
        EntryPath,
        generate_conditional_candidate_paths,
        search_candidate_portfolios,
    )

    horizon = load_horizon(
        paths,
        season=season,
        first_week=first_week,
        last_week=last_week,
        manual_max_age_hours=manual_max_age_hours,
        as_of=as_of,
    )
    all_picks = load_picks(paths.picks, season=season, contest=contest)
    state_by_entry = {
        state.entry_id: state
        for state in entry_states(all_picks, season=season)
        if state.contest == contest
    }
    active_entry_ids = [
        entry_id
        for entry_id in entry_ids
        if entry_id not in state_by_entry or state_by_entry[entry_id].active
    ]
    if not active_entry_ids:
        raise SurvivorDataError("All three owned entries have been eliminated.")

    if search_method == "global":
        from optimizer_global import search_global_portfolios
        return search_global_portfolios(
            active_entry_ids, horizon.probabilities, horizon.game_ids, horizon.opponents,
            used_teams_by_entry={entry: tuple(pick.team for pick in entry_history(
                paths, season=season, contest=contest, entry_id=entry, before_week=first_week))
                for entry in active_entry_ids},
            fixed_other_survivors=other_survivors, sensitivity_other_survivors=sensitivity,
            max_nodes=global_max_nodes or None, max_seconds=global_max_seconds or None,
            absolute_gap_tolerance=global_absolute_gap,
        )
    if search_method != "candidate":
        raise SurvivorDataError("Search method must be candidate or global")
    candidate_sets = []
    candidate_cache = {}
    for entry_id in active_entry_ids:
        history = entry_history(
            paths,
            season=season,
            contest=contest,
            entry_id=entry_id,
            before_week=first_week,
        )
        used_teams = {pick.team for pick in history}
        cache_key = frozenset(used_teams)
        cached_paths = candidate_cache.get(cache_key)
        if cached_paths is None:
            cached_paths = generate_conditional_candidate_paths(
                entry_id,
                horizon.probabilities,
                horizon.game_ids,
                horizon.opponents,
                used_teams=cache_key,
                beam_width=beam_width,
                candidate_limit=candidate_limit,
                paths_per_current_team=paths_per_current_team,
            )
            candidate_cache[cache_key] = cached_paths
        candidates = tuple(
            EntryPath(entry_id, path.picks)
            for path in cached_paths
        )
        if not candidates:
            raise SurvivorDataError(f"No legal candidate path exists for {entry_id}.")
        candidate_sets.append(candidates)

    return search_candidate_portfolios(
        candidate_sets,
        objective="expected_prize_share",
        fixed_other_survivors=other_survivors,
        sensitivity_other_survivors=sensitivity,
        max_evaluations=max_evaluations,
        seed=seed,
    )
