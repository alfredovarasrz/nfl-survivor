"""Persistent inputs, validation, and external-data adapters for Survivor.

All user-owned state is stored in readable CSV or TOML files.  The module uses
only the Python standard library so importing it never requires pandas or an
optional data-frame runtime.
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import re
import statistics
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from probabilities import (
    consensus_fair_probability,
    normalize_probability_pair,
    remove_vig,
)


NFL_TEAMS = frozenset(
    {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    }
)

API_TEAM_NAMES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

SCHEDULE_FIELDS = ("season", "week", "game_id", "away_team", "home_team", "kickoff")
PICK_FIELDS = (
    "season", "contest", "entry_id", "week", "team", "submitted_at", "result", "rebuy"
)
ODDS_FIELDS = (
    "season", "week", "game_id", "team_a", "team_b", "probability_a",
    "probability_b", "moneyline_a", "moneyline_b", "bookmakers", "source", "observed_at", "bookmaker_updates",
)
MANUAL_ODDS_FIELDS = (
    "season", "week", "game_id", "team_a", "team_b", "moneyline_a", "moneyline_b",
    "probability_a", "probability_b", "source", "observed_at",
)
POOL_STATUS_FIELDS = (
    "season", "contest", "week", "snapshot_phase", "entries_alive",
    "entries_with_unused_rebuy", "entries_eligible_to_rebuy_this_week",
    "rebuys_entered_this_week", "total_rebuys_to_date", "recorded_at",
)
POPULARITY_FIELDS = ("season", "week", "team", "pick_percentage", "source", "observed_at")
OPPONENT_PICK_FIELDS = ("season", "contest", "entry_id", "week", "team", "result")
ODDS_KEYCHAIN_SERVICE = "NFL Survivor Public The Odds API"


class SurvivorDataError(ValueError):
    """Base exception for invalid or incomplete Survivor data."""


class OddsAPIError(SurvivorDataError):
    """Raised when The Odds API cannot provide a usable response."""


class ScheduleAPIError(SurvivorDataError):
    """Raised when the public NFL schedule feed is unavailable or malformed."""


class MissingOddsError(SurvivorDataError):
    """Raised when API and manual fallback together do not cover requested games."""


class StaleManualOddsError(SurvivorDataError):
    """Raised when a required manual fallback has not been verified recently."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def raw_data(self) -> Path:
        return self.root / "raw_data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def contests(self) -> Path:
        return self.inputs / "contests.toml"

    def schedule(self, season: int) -> Path:
        return self.inputs / f"schedule_{season}.csv"

    @property
    def picks(self) -> Path:
        return self.inputs / "my_picks.csv"

    @property
    def manual_odds(self) -> Path:
        return self.inputs / "manual_odds.csv"

    @property
    def normalized_odds(self) -> Path:
        return self.inputs / "odds.csv"

    @property
    def pool_status(self) -> Path:
        return self.inputs / "pool_status.csv"

    @property
    def popularity(self) -> Path:
        return self.inputs / "public_popularity.csv"

    @property
    def opponent_picks(self) -> Path:
        return self.inputs / "opponent_picks.csv"


@dataclass(frozen=True)
class Game:
    season: int
    week: int
    game_id: str
    away_team: str
    home_team: str
    kickoff: str

    @property
    def teams(self) -> frozenset[str]:
        return frozenset((self.away_team, self.home_team))


@dataclass(frozen=True)
class PickRecord:
    season: int
    contest: str
    entry_id: str
    week: int
    team: str
    submitted_at: str
    result: str = "pending"
    rebuy: bool = False


@dataclass(frozen=True)
class OddsRecord:
    season: int
    week: int
    game_id: str
    team_a: str
    team_b: str
    probability_a: float
    probability_b: float
    moneyline_a: float | None
    moneyline_b: float | None
    bookmakers: str
    source: str
    observed_at: str
    bookmaker_updates: str = ""


@dataclass(frozen=True)
class APIResponse:
    events: list[dict[str, Any]]
    remaining_requests: str | None = None
    used_requests: str | None = None


@dataclass(frozen=True)
class OddsRefreshResult:
    records: list[OddsRecord]
    api_records: int
    manual_records: int
    api_error: str | None
    remaining_requests: str | None
    raw_response_path: Path | None
    manual_games: tuple[OddsRecord, ...] = ()


@dataclass(frozen=True)
class OddsCoverage:
    events_returned: int
    events_matched_schedule: int
    weeks: tuple[int, ...]
    bookmakers: tuple[str, ...]
    missing_games: tuple[Game, ...]
    remaining_requests: str | None
    used_requests: str | None


@dataclass(frozen=True)
class EntryState:
    contest: str
    entry_id: str
    active: bool
    used_teams: frozenset[str]
    last_week: int
    rebuys_used: int


@dataclass(frozen=True)
class PoolStatusRecord:
    """One contest snapshot for entrants eligible to submit that week's pick."""

    season: int
    contest: str
    week: int
    snapshot_phase: str
    entries_alive: int
    entries_with_unused_rebuy: int
    entries_eligible_to_rebuy_this_week: int
    rebuys_entered_this_week: int
    total_rebuys_to_date: int
    recorded_at: str


DEFAULT_CONTESTS = """# Illustrative rule template. Monetary values are synthetic; not runtime configuration.

[small_pool]
season = 2026
owned_entries = 1
entry_fee = 100.0
objective = "expected_weeks_survived"
tie_is_loss = true
team_reuse_allowed = false
end_week = 18
ends_when_one_entry_remains = true
zero_survivors_rule = "split_among_week_entrants"
week_18_payout = "equal_per_surviving_entry"
all_entry_fees_to_prize_pool = true
rebuy_allowed = true
rebuy_limit = 1
rebuy_cost = 25.0
rebuy_cutoff_week = 18
used_teams_reset_on_rebuy = false
rebuy_timing = "immediate_next_week"

[large_pool]
season = 2026
owned_entries = 3
entry_fee = 10.0
objective = "expected_prize_share"
tie_is_loss = true
team_reuse_allowed = false
end_week = 18
ends_when_one_entry_remains = true
zero_survivors_rule = "split_among_week_entrants"
week_18_payout = "equal_per_surviving_entry"
all_entry_fees_to_prize_pool = true
rebuy_allowed = false
share_unit = "surviving_entry"
"""


def _ensure_csv(path: Path, fields: Sequence[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def initialize_project(paths: DataPaths, season: int) -> None:
    """Create transparent input/output files without replacing existing data."""

    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.raw_data.mkdir(parents=True, exist_ok=True)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    if not paths.contests.exists():
        paths.contests.write_text(DEFAULT_CONTESTS.replace("2026", str(season)), encoding="utf-8")
    _ensure_csv(paths.schedule(season), SCHEDULE_FIELDS)
    _ensure_csv(paths.picks, PICK_FIELDS)
    _ensure_csv(paths.manual_odds, MANUAL_ODDS_FIELDS)
    _ensure_csv(paths.normalized_odds, ODDS_FIELDS)
    _ensure_csv(paths.pool_status, POOL_STATUS_FIELDS)
    _ensure_csv(paths.popularity, POPULARITY_FIELDS)
    _ensure_csv(paths.opponent_picks, OPPONENT_PICK_FIELDS)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SurvivorDataError(f"Required file does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {field: row.get(field, "") for field in fields}
            writer.writerow(cleaned)
    temporary.replace(path)


def normalize_team(team: str) -> str:
    value = team.strip()
    normalized = API_TEAM_NAMES.get(value, value.upper())
    if normalized not in NFL_TEAMS:
        raise SurvivorDataError(f"Unknown NFL team: {team!r}")
    return normalized


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SurvivorDataError(f"{label} must be an integer") from exc
    if number <= 0:
        raise SurvivorDataError(f"{label} must be positive")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SurvivorDataError(f"{label} must be an integer") from exc
    if number < 0:
        raise SurvivorDataError(f"{label} must be nonnegative")
    return number


def validate_schedule(games: Sequence[Game]) -> None:
    if not games:
        raise SurvivorDataError("Schedule contains no games")
    identifiers: set[tuple[int, str]] = set()
    appearances: set[tuple[int, int, str]] = set()
    for game in games:
        if game.week > 18:
            raise SurvivorDataError(f"Week must be between 1 and 18: {game.week}")
        if game.away_team == game.home_team:
            raise SurvivorDataError(f"Game {game.game_id} contains the same team twice")
        game_key = (game.season, game.game_id)
        if game_key in identifiers:
            raise SurvivorDataError(f"Duplicate game_id in season {game.season}: {game.game_id}")
        identifiers.add(game_key)
        for team in game.teams:
            appearance = (game.season, game.week, team)
            if appearance in appearances:
                raise SurvivorDataError(f"{team} appears more than once in Week {game.week}")
            appearances.add(appearance)


def load_schedule(path: Path, *, season: int | None = None) -> list[Game]:
    games: list[Game] = []
    for index, row in enumerate(_read_csv(path), start=2):
        try:
            game = Game(
                season=_positive_int(row.get("season"), f"season on row {index}"),
                week=_positive_int(row.get("week"), f"week on row {index}"),
                game_id=(row.get("game_id") or "").strip(),
                away_team=normalize_team(row.get("away_team") or ""),
                home_team=normalize_team(row.get("home_team") or ""),
                kickoff=(row.get("kickoff") or "").strip(),
            )
        except SurvivorDataError as exc:
            raise SurvivorDataError(f"Invalid schedule row {index}: {exc}") from exc
        if not game.game_id:
            raise SurvivorDataError(f"Invalid schedule row {index}: game_id is required")
        if season is None or game.season == season:
            games.append(game)
    validate_schedule(games)
    return sorted(games, key=lambda game: (game.week, game.kickoff, game.game_id))


def import_schedule(source: Path, destination: Path, *, season: int, replace: bool = False) -> list[Game]:
    """Validate and store a season schedule once."""

    if destination.exists() and _read_csv(destination) and not replace:
        raise SurvivorDataError(
            f"Schedule already exists at {destination}; pass replace=True only for a deliberate correction"
        )
    games = load_schedule(source, season=season)
    _write_csv(destination, SCHEDULE_FIELDS, (asdict(game) for game in games))
    return games


def validate_pick_against_schedule(record: PickRecord, schedule: Sequence[Game]) -> None:
    """Verify that a submitted team actually plays in the recorded week."""

    matching = [
        game for game in schedule
        if game.season == record.season
        and game.week == record.week
        and record.team in game.teams
    ]
    if len(matching) != 1:
        raise SurvivorDataError(
            f"{record.team} does not have exactly one scheduled game in Week {record.week}"
        )


def load_picks(path: Path, *, season: int | None = None, contest: str | None = None) -> list[PickRecord]:
    if not path.exists():
        return []
    all_records: list[PickRecord] = []
    keys: set[tuple[int, str, str, int]] = set()
    teams_used: set[tuple[int, str, str, str]] = set()
    for index, row in enumerate(_read_csv(path), start=2):
        record_season = _positive_int(row.get("season"), f"season on row {index}")
        record_contest = (row.get("contest") or "").strip()
        entry_id = (row.get("entry_id") or "").strip()
        week = _positive_int(row.get("week"), f"week on row {index}")
        team = normalize_team(row.get("team") or "")
        result = (row.get("result") or "pending").strip().lower()
        if result not in {"pending", "win", "loss", "tie"}:
            raise SurvivorDataError(f"Invalid result on row {index}: {result!r}")
        if not record_contest or not entry_id:
            raise SurvivorDataError(f"contest and entry_id are required on row {index}")
        if record_contest not in {"small_pool", "large_pool"}:
            raise SurvivorDataError(
                f"contest on row {index} must be small_pool or large_pool"
            )
        key = (record_season, record_contest, entry_id, week)
        if key in keys:
            raise SurvivorDataError(f"Duplicate pick for {record_contest}/{entry_id}/Week {week}")
        keys.add(key)
        team_key = (record_season, record_contest, entry_id, team)
        if team_key in teams_used:
            raise SurvivorDataError(f"{team} was used more than once by {record_contest}/{entry_id}")
        teams_used.add(team_key)
        rebuy = (row.get("rebuy") or "false").strip().lower() in {"1", "true", "yes"}
        record = PickRecord(
            season=record_season,
            contest=record_contest,
            entry_id=entry_id,
            week=week,
            team=team,
            submitted_at=(row.get("submitted_at") or "").strip(),
            result=result,
            rebuy=rebuy,
        )
        all_records.append(record)
    _validate_rebuy_timelines(all_records)
    records = [
        record
        for record in all_records
        if (season is None or record.season == season)
        and (contest is None or record.contest == contest)
    ]
    return sorted(records, key=lambda pick: (pick.season, pick.contest, pick.entry_id, pick.week))


def load_pool_status(
    path: Path,
    *,
    season: int | None = None,
    contest: str | None = None,
) -> list[PoolStatusRecord]:
    """Load validated weekly pool snapshots."""

    if not path.exists():
        return []
    records: list[PoolStatusRecord] = []
    for index, row in enumerate(_read_csv(path), start=2):
        if not (row.get("season") or "").strip():
            continue
        record = PoolStatusRecord(
            season=_positive_int(row.get("season"), f"season on row {index}"),
            contest=(row.get("contest") or "").strip(),
            week=_positive_int(row.get("week"), f"week on row {index}"),
            snapshot_phase=(row.get("snapshot_phase") or "").strip(),
            entries_alive=_nonnegative_int(
                row.get("entries_alive"), f"entries_alive on row {index}"
            ),
            entries_with_unused_rebuy=_nonnegative_int(
                row.get("entries_with_unused_rebuy"),
                f"entries_with_unused_rebuy on row {index}",
            ),
            entries_eligible_to_rebuy_this_week=_nonnegative_int(
                row.get("entries_eligible_to_rebuy_this_week"),
                f"entries_eligible_to_rebuy_this_week on row {index}",
            ),
            rebuys_entered_this_week=_nonnegative_int(
                row.get("rebuys_entered_this_week"),
                f"rebuys_entered_this_week on row {index}",
            ),
            total_rebuys_to_date=_nonnegative_int(
                row.get("total_rebuys_to_date"),
                f"total_rebuys_to_date on row {index}",
            ),
            recorded_at=(row.get("recorded_at") or "").strip(),
        )
        _validate_pool_status_record(record)
        if (season is None or record.season == season) and (
            contest is None or record.contest == contest
        ):
            records.append(record)
    _validate_pool_status_sequence(records)
    return sorted(
        records,
        key=lambda item: (item.season, item.contest, item.week, item.snapshot_phase),
    )


def _validate_pool_status_record(record: PoolStatusRecord) -> None:
    if record.contest not in {"small_pool", "large_pool"}:
        raise SurvivorDataError("Pool status contest must be small_pool or large_pool")
    if record.week > 18:
        raise SurvivorDataError("Pool status week must be between 1 and 18")
    if record.snapshot_phase not in {"before_rebuy_deadline", "finalized_for_week"}:
        raise SurvivorDataError(
            "Pool status phase must be before_rebuy_deadline or finalized_for_week"
        )
    if record.entries_with_unused_rebuy > record.entries_alive:
        raise SurvivorDataError("Entries with an unused rebuy cannot exceed entries alive")
    if record.contest == "large_pool" and any(
        (
            record.entries_with_unused_rebuy,
            record.entries_eligible_to_rebuy_this_week,
            record.rebuys_entered_this_week,
            record.total_rebuys_to_date,
        )
    ):
        raise SurvivorDataError("The large pool does not allow rebuy counts")
    if record.rebuys_entered_this_week > record.total_rebuys_to_date:
        raise SurvivorDataError("Weekly rebuys cannot exceed total rebuys to date")
    if (
        record.rebuys_entered_this_week
        > record.entries_eligible_to_rebuy_this_week
    ):
        raise SurvivorDataError(
            "Rebuys entered this week cannot exceed entries eligible to rebuy"
        )
    if record.rebuys_entered_this_week > record.entries_alive:
        raise SurvivorDataError("Weekly rebuys cannot exceed entries alive")
    if record.week == 1 and record.entries_eligible_to_rebuy_this_week:
        raise SurvivorDataError("No entry can be eligible to rebuy in Week 1")
    if not record.recorded_at:
        raise SurvivorDataError("Pool status recorded_at is required")


def _validate_pool_status_sequence(records: Sequence[PoolStatusRecord]) -> None:
    grouped: dict[tuple[int, str], list[PoolStatusRecord]] = {}
    seen: set[tuple[int, str, int, str]] = set()
    for record in records:
        key = (record.season, record.contest, record.week, record.snapshot_phase)
        if key in seen:
            raise SurvivorDataError(
                "Duplicate pool-status snapshot for the same contest, week, and phase"
            )
        seen.add(key)
        grouped.setdefault((record.season, record.contest), []).append(record)
    for contest_records in grouped.values():
        ordered = sorted(contest_records, key=lambda item: (item.week, item.snapshot_phase))
        totals = [record.total_rebuys_to_date for record in ordered]
        if any(later < earlier for earlier, later in zip(totals, totals[1:])):
            raise SurvivorDataError("Total rebuys to date cannot decrease over time")


def record_pool_status(
    path: Path,
    record: PoolStatusRecord,
    *,
    replace: bool = False,
) -> None:
    """Insert or deliberately correct one pool-status snapshot."""

    _validate_pool_status_record(record)
    existing = load_pool_status(path) if path.exists() else []
    key = (record.season, record.contest, record.week, record.snapshot_phase)
    if any(
        (item.season, item.contest, item.week, item.snapshot_phase) == key
        for item in existing
    ) and not replace:
        raise SurvivorDataError(
            "A pool-status snapshot already exists for this week and phase; "
            "use --replace to correct it"
        )
    merged = [
        item
        for item in existing
        if (item.season, item.contest, item.week, item.snapshot_phase) != key
    ]
    merged.append(record)
    _validate_pool_status_sequence(merged)
    _write_csv(
        path,
        POOL_STATUS_FIELDS,
        (
            asdict(item)
            for item in sorted(
                merged,
                key=lambda item: (
                    item.season,
                    item.contest,
                    item.week,
                    item.snapshot_phase,
                ),
            )
        ),
    )


def _pick_to_row(record: PickRecord) -> dict[str, Any]:
    row = asdict(record)
    row["rebuy"] = str(record.rebuy).lower()
    return row


def _validate_rebuy_timeline(records: Sequence[PickRecord]) -> None:
    """Validate rebuy markers against an entry's chronological results."""

    if records and records[0].contest == "large_pool" and any(
        pick.rebuy for pick in records
    ):
        raise SurvivorDataError("The large pool does not allow rebuys")
    active = True
    eliminated_week: int | None = None
    rebuys_used = 0
    for pick in sorted(records, key=lambda item: item.week):
        if active:
            if pick.rebuy:
                raise SurvivorDataError(
                    "A rebuy can only be recorded after this entry is eliminated."
                )
        else:
            required_week = eliminated_week + 1 if eliminated_week is not None else None
            if not pick.rebuy:
                if pick.week == required_week:
                    raise SurvivorDataError(
                        "This entry is eliminated. Mark the next pick as a rebuy if the "
                        "contest permits it."
                    )
                raise SurvivorDataError(
                    f"This entry's rebuy expired after Week {required_week}; a loss or "
                    f"tie in Week {eliminated_week} permits a rebuy only in Week "
                    f"{required_week}."
                )
            if rebuys_used >= 1:
                raise SurvivorDataError("This entry already used its one rebuy")
            if pick.week != required_week:
                raise SurvivorDataError(
                    f"A loss or tie in Week {eliminated_week} permits a rebuy only in "
                    f"Week {required_week}; the rebuy has expired."
                )
            active = True

        if pick.rebuy:
            rebuys_used += 1
        if pick.result in {"loss", "tie"}:
            active = False
            eliminated_week = pick.week


def _validate_rebuy_timelines(records: Sequence[PickRecord]) -> None:
    grouped: dict[tuple[int, str, str], list[PickRecord]] = {}
    for pick in records:
        grouped.setdefault(
            (pick.season, pick.contest, pick.entry_id), []
        ).append(pick)
    for entry_records in grouped.values():
        _validate_rebuy_timeline(entry_records)


def record_pick(path: Path, record: PickRecord, *, replace: bool = False) -> None:
    """Record the pick actually submitted, preserving all seasons and contests."""

    normalize_team(record.team)
    if record.contest not in {"small_pool", "large_pool"}:
        raise SurvivorDataError("Contest must be small_pool or large_pool")
    if record.contest == "large_pool" and record.rebuy:
        raise SurvivorDataError("The large pool does not allow rebuys")
    if record.result not in {"pending", "win", "loss", "tie"}:
        raise SurvivorDataError(f"Invalid result: {record.result}")
    existing = load_picks(path) if path.exists() else []
    key = (record.season, record.contest, record.entry_id, record.week)
    same_key = [pick for pick in existing if (pick.season, pick.contest, pick.entry_id, pick.week) == key]
    if same_key and not replace:
        raise SurvivorDataError("A pick already exists for this entry and week; use --replace to correct it")
    duplicate_team = [
        pick for pick in existing
        if pick.season == record.season and pick.contest == record.contest
        and pick.entry_id == record.entry_id and pick.team == record.team
        and (pick.season, pick.contest, pick.entry_id, pick.week) != key
    ]
    if duplicate_team:
        raise SurvivorDataError(f"{record.team} has already been used by this entry")

    merged = [pick for pick in existing if (pick.season, pick.contest, pick.entry_id, pick.week) != key]
    merged.append(record)
    merged.sort(key=lambda pick: (pick.season, pick.contest, pick.entry_id, pick.week))
    entry_records = [
        pick
        for pick in merged
        if pick.season == record.season
        and pick.contest == record.contest
        and pick.entry_id == record.entry_id
    ]
    _validate_rebuy_timeline(entry_records)
    _write_csv(path, PICK_FIELDS, (_pick_to_row(pick) for pick in merged))


def record_result(
    path: Path,
    *,
    season: int,
    contest: str,
    entry_id: str,
    week: int,
    result: str,
) -> PickRecord:
    result = result.strip().lower()
    if result not in {"win", "loss", "tie", "pending"}:
        raise SurvivorDataError("Result must be win, loss, tie, or pending")
    picks = load_picks(path)
    target_key = (season, contest, entry_id, week)
    matching = [pick for pick in picks if (pick.season, pick.contest, pick.entry_id, pick.week) == target_key]
    if not matching:
        raise SurvivorDataError(f"No recorded pick for {contest}/{entry_id}/Week {week}")
    updated = PickRecord(**{**asdict(matching[0]), "result": result})
    record_pick(path, updated, replace=True)
    return updated


def entry_states(picks: Sequence[PickRecord], *, season: int) -> list[EntryState]:
    grouped: dict[tuple[str, str], list[PickRecord]] = {}
    for pick in picks:
        if pick.season == season:
            grouped.setdefault((pick.contest, pick.entry_id), []).append(pick)
    states: list[EntryState] = []
    for (contest, entry_id), records in sorted(grouped.items()):
        records.sort(key=lambda pick: pick.week)
        last_loss_index = max(
            (index for index, pick in enumerate(records) if pick.result in {"loss", "tie"}),
            default=-1,
        )
        active = last_loss_index == -1 or any(pick.rebuy for pick in records[last_loss_index + 1 :])
        states.append(
            EntryState(
                contest=contest,
                entry_id=entry_id,
                active=active,
                used_teams=frozenset(pick.team for pick in records),
                last_week=max(pick.week for pick in records),
                rebuys_used=sum(pick.rebuy for pick in records),
            )
        )
    return states


def load_odds_api_key_from_keychain() -> str | None:
    """Read the API key from the user's macOS Keychain, when available."""

    if sys.platform != "darwin":
        return None
    completed = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-a",
            getpass.getuser(),
            "-s",
            ODDS_KEYCHAIN_SERVICE,
            "-w",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    key = completed.stdout.strip()
    return key or None


def configure_odds_api_key_in_keychain() -> None:
    """Prompt securely and add or replace the Odds API key in macOS Keychain."""

    if sys.platform != "darwin":
        raise SurvivorDataError(
            "Guided Keychain setup is available on macOS only. "
            "Set THE_ODDS_API_KEY in the environment on this system."
        )
    try:
        subprocess.run(
            [
                "/usr/bin/security",
                "add-generic-password",
                "-U",
                "-a",
                getpass.getuser(),
                "-s",
                ODDS_KEYCHAIN_SERVICE,
                "-w",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SurvivorDataError("The API key was not saved to Keychain.") from exc


class OddsAPIClient:
    """Small, injectable client for The Odds API v4."""

    base_url = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        opener: Callable[..., Any] | None = None,
        timeout: float = 20.0,
        use_keychain: bool = False,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("THE_ODDS_API_KEY")
            or (load_odds_api_key_from_keychain()
                if use_keychain or os.environ.get("SURVIVOR_PUBLIC_USE_KEYCHAIN") == "1" else None)
        )
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout

    def _get(self, endpoint: str, parameters: Mapping[str, Any]) -> APIResponse:
        if not self.api_key:
            raise OddsAPIError(
                "THE_ODDS_API_KEY is not configured. Set it in the environment before refreshing odds."
            )
        query = urllib.parse.urlencode({**parameters, "apiKey": self.api_key})
        request = urllib.request.Request(f"{self.base_url}/{endpoint}?{query}")
        try:
            response = self.opener(request, timeout=self.timeout)
            with response:
                payload = json.loads(response.read().decode("utf-8"))
                headers = response.headers
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise OddsAPIError(self._safe_error(f"The Odds API request failed: {exc}")) from None
        if not isinstance(payload, list):
            message = payload.get("message", "unexpected response") if isinstance(payload, dict) else "unexpected response"
            raise OddsAPIError(self._safe_error(f"The Odds API returned {message}")) from None
        return APIResponse(
            events=payload,
            remaining_requests=headers.get("x-requests-remaining"),
            used_requests=headers.get("x-requests-used"),
        )

    def _safe_error(self, message: str) -> str:
        if self.api_key:
            message = message.replace(self.api_key, "[REDACTED]")
            message = message.replace(urllib.parse.quote(self.api_key, safe=""), "[REDACTED]")
            message = message.replace(urllib.parse.quote_plus(self.api_key), "[REDACTED]")
        return re.sub(r"(?i)(api[_-]?key|access_token|token)=([^&\s]+)", r"\1=[REDACTED]", message)

    def fetch_odds(self, *, regions: str = "us", bookmakers: str | None = None) -> APIResponse:
        parameters: dict[str, Any] = {
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            parameters["bookmakers"] = bookmakers
            parameters.pop("regions", None)
        return self._get("sports/americanfootball_nfl/odds", parameters)

    def fetch_scores(self, *, days_from: int = 3) -> APIResponse:
        return self._get(
            "sports/americanfootball_nfl/scores",
            {"daysFrom": days_from, "dateFormat": "iso"},
        )


class ESPNScheduleClient:
    """Injectable client for the one-time regular-season schedule download."""

    base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

    def __init__(self, *, opener: Callable[..., Any] | None = None, timeout: float = 20.0) -> None:
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout

    def fetch_week(self, *, season: int, week: int) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"year": season, "seasontype": 2, "week": week, "limit": 500}
        )
        request = urllib.request.Request(f"{self.base_url}?{query}")
        try:
            response = self.opener(request, timeout=self.timeout)
            with response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise ScheduleAPIError(f"The schedule request failed: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            raise ScheduleAPIError("The schedule feed returned an unexpected response.")
        return payload


def games_from_espn_payload(
    payload: Mapping[str, Any], *, season: int, week: int
) -> list[Game]:
    """Convert one scoreboard response into schedule rows."""

    games: list[Game] = []
    for event in payload.get("events", []):
        try:
            competition = event["competitions"][0]
            competitors = {
                competitor["homeAway"]: normalize_team(
                    str(
                        competitor.get("team", {}).get("displayName")
                        or competitor.get("team", {}).get("abbreviation")
                    )
                )
                for competitor in competition["competitors"]
            }
            games.append(
                Game(
                    season=season,
                    week=week,
                    game_id=str(event["id"]),
                    away_team=competitors["away"],
                    home_team=competitors["home"],
                    kickoff=str(event["date"]),
                )
            )
        except (KeyError, IndexError, TypeError, SurvivorDataError) as exc:
            raise ScheduleAPIError(
                f"The schedule feed contains a malformed event in Week {week}."
            ) from exc
    if not games:
        raise ScheduleAPIError(f"The schedule feed returned no games for Week {week}.")
    return games


def download_schedule(
    paths: DataPaths,
    *,
    season: int,
    client: ESPNScheduleClient,
    replace: bool = False,
) -> list[Game]:
    """Download, validate, and store all 18 regular-season weeks once."""

    destination = paths.schedule(season)
    if destination.exists() and _read_csv(destination) and not replace:
        existing = load_schedule(destination, season=season)
        if existing:
            raise SurvivorDataError(
                f"Schedule already exists at {destination}; use --replace only for a correction."
            )

    all_games: list[Game] = []
    raw_weeks: dict[str, Any] = {}
    for week in range(1, 19):
        payload = client.fetch_week(season=season, week=week)
        raw_weeks[str(week)] = payload
        all_games.extend(games_from_espn_payload(payload, season=season, week=week))

    validate_schedule(all_games)
    if {game.week for game in all_games} != set(range(1, 19)):
        raise ScheduleAPIError("The downloaded schedule does not cover all 18 weeks.")
    paths.raw_data.mkdir(parents=True, exist_ok=True)
    (paths.raw_data / f"schedule_{season}.json").write_text(
        json.dumps(raw_weeks, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(destination, SCHEDULE_FIELDS, (asdict(game) for game in all_games))
    return sorted(all_games, key=lambda game: (game.week, game.kickoff, game.game_id))


def _match_schedule_game(event: Mapping[str, Any], schedule: Sequence[Game]) -> Game | None:
    try:
        teams = frozenset((normalize_team(str(event["away_team"])), normalize_team(str(event["home_team"]))))
    except (KeyError, SurvivorDataError):
        return None
    candidates = [game for game in schedule if game.teams == teams]
    if len(candidates) == 1:
        return candidates[0]
    commence = str(event.get("commence_time", ""))[:10]
    dated = [game for game in candidates if game.kickoff[:10] == commence]
    return dated[0] if len(dated) == 1 else None


def inspect_odds_coverage(
    paths: DataPaths,
    *,
    season: int,
    client: OddsAPIClient,
    regions: str = "us",
    bookmakers: str | None = None,
) -> OddsCoverage:
    """Use one API request to report actual NFL week and bookmaker coverage."""

    schedule = load_schedule(paths.schedule(season), season=season)
    response = client.fetch_odds(regions=regions, bookmakers=bookmakers)
    matched_weeks: set[int] = set()
    matched_game_ids: set[str] = set()
    bookmaker_keys: set[str] = set()
    matched_events = 0
    for event in response.events:
        game = _match_schedule_game(event, schedule)
        if game is not None and game.season == season:
            matched_events += 1
            matched_weeks.add(game.week)
            matched_game_ids.add(game.game_id)
        for bookmaker in event.get("bookmakers", []):
            key = str(bookmaker.get("key") or bookmaker.get("title") or "").strip()
            if key:
                bookmaker_keys.add(key)
    return OddsCoverage(
        events_returned=len(response.events),
        events_matched_schedule=matched_events,
        weeks=tuple(sorted(matched_weeks)),
        bookmakers=tuple(sorted(bookmaker_keys)),
        missing_games=tuple(
            sorted(
                (
                    game
                    for game in schedule
                    if game.season == season and game.game_id not in matched_game_ids
                ),
                key=lambda game: (game.week, game.kickoff, game.game_id),
            )
        ),
        remaining_requests=response.remaining_requests,
        used_requests=response.used_requests,
    )


def odds_records_from_api(
    response: APIResponse,
    schedule: Sequence[Game],
    *,
    season: int,
    weeks: set[int],
    observed_at: str | None = None,
) -> list[OddsRecord]:
    timestamp = observed_at or utc_now()
    records: list[OddsRecord] = []
    for event in response.events:
        game = _match_schedule_game(event, schedule)
        if game is None or game.season != season or game.week not in weeks:
            continue
        team_a, team_b = game.away_team, game.home_team
        book_pairs: list[tuple[float, float]] = []
        book_names: list[str] = []
        book_updates: dict[str, str] = {}
        raw_a: list[float] = []
        raw_b: list[float] = []
        for bookmaker in event.get("bookmakers", []):
            markets = {market.get("key"): market for market in bookmaker.get("markets", [])}
            h2h = markets.get("h2h")
            if not h2h:
                continue
            prices: dict[str, float] = {}
            for outcome in h2h.get("outcomes", []):
                try:
                    prices[normalize_team(str(outcome["name"]))] = float(outcome["price"])
                except (KeyError, TypeError, ValueError, SurvivorDataError):
                    continue
            if team_a in prices and team_b in prices:
                book_pairs.append((prices[team_a], prices[team_b]))
                raw_a.append(prices[team_a])
                raw_b.append(prices[team_b])
                book_name = str(bookmaker.get("key") or bookmaker.get("title") or "unknown")
                book_names.append(book_name)
                update = h2h.get("last_update") or bookmaker.get("last_update")
                if update:
                    book_updates[book_name] = str(update)
        if not book_pairs:
            continue
        probability_a, probability_b = consensus_fair_probability(book_pairs)
        records.append(
            OddsRecord(
                season=season,
                week=game.week,
                game_id=game.game_id,
                team_a=team_a,
                team_b=team_b,
                probability_a=probability_a,
                probability_b=probability_b,
                moneyline_a=statistics.median(raw_a),
                moneyline_b=statistics.median(raw_b),
                bookmakers="|".join(sorted(book_names)),
                source="the_odds_api",
                observed_at=timestamp,
                bookmaker_updates=json.dumps(book_updates, sort_keys=True),
            )
        )
    return records


def load_manual_odds(path: Path, schedule: Sequence[Game], *, season: int, weeks: set[int]) -> list[OddsRecord]:
    if not path.exists():
        return []
    schedule_by_id = {(game.season, game.game_id): game for game in schedule}
    records: list[OddsRecord] = []
    seen_game_ids: set[str] = set()
    for index, row in enumerate(_read_csv(path), start=2):
        if not (row.get("season") or "").strip():
            continue
        row_season = _positive_int(row.get("season"), f"season on row {index}")
        week = _positive_int(row.get("week"), f"week on row {index}")
        if row_season != season or week not in weeks:
            continue
        game_id = (row.get("game_id") or "").strip()
        game = schedule_by_id.get((season, game_id))
        if game is None or game.week != week:
            raise SurvivorDataError(f"Manual odds row {index} does not match the stored schedule")
        if game_id in seen_game_ids:
            raise SurvivorDataError(f"Manual odds contain duplicate rows for {game_id}")
        seen_game_ids.add(game_id)
        team_a = normalize_team(row.get("team_a") or game.away_team)
        team_b = normalize_team(row.get("team_b") or game.home_team)
        if frozenset((team_a, team_b)) != game.teams:
            raise SurvivorDataError(f"Manual odds row {index} teams do not match {game_id}")
        moneyline_a_text = (row.get("moneyline_a") or "").strip()
        moneyline_b_text = (row.get("moneyline_b") or "").strip()
        probability_a_text = (row.get("probability_a") or "").strip()
        probability_b_text = (row.get("probability_b") or "").strip()
        try:
            if moneyline_a_text and moneyline_b_text:
                moneyline_a = float(moneyline_a_text)
                moneyline_b = float(moneyline_b_text)
                probability_a, probability_b = remove_vig(moneyline_a, moneyline_b)
            elif probability_a_text and probability_b_text:
                probability_a, probability_b = normalize_probability_pair(
                    float(probability_a_text), float(probability_b_text)
                )
                moneyline_a = moneyline_b = None
            elif probability_a_text:
                probability_a = float(probability_a_text)
                probability_a, probability_b = normalize_probability_pair(
                    probability_a, 1.0 - probability_a
                )
                moneyline_a = moneyline_b = None
            else:
                raise SurvivorDataError(
                    f"Manual odds row {index} needs two moneylines, two probabilities, "
                    "or probability_a"
                )
        except (TypeError, ValueError) as exc:
            raise SurvivorDataError(
                f"Manual odds row {index} contains invalid prices or probabilities: {exc}"
            ) from exc
        records.append(
            OddsRecord(
                season=season,
                week=week,
                game_id=game_id,
                team_a=team_a,
                team_b=team_b,
                probability_a=probability_a,
                probability_b=probability_b,
                moneyline_a=moneyline_a,
                moneyline_b=moneyline_b,
                bookmakers="manual",
                source=(row.get("source") or "manual").strip(),
                observed_at=(row.get("observed_at") or "").strip(),
            )
        )
    return records


def _parse_aware_timestamp(value: str, *, label: str) -> datetime:
    text = value.strip()
    if not text:
        raise StaleManualOddsError(f"{label} has never been verified")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StaleManualOddsError(f"{label} has an invalid verification timestamp: {value!r}") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise StaleManualOddsError(f"{label} has a timezone-free verification timestamp")
    return timestamp.astimezone(timezone.utc)


def validate_manual_odds_freshness(
    records: Sequence[OddsRecord],
    *,
    max_age_hours: float = 24.0,
    as_of: datetime | None = None,
) -> None:
    """Reject required manual lines that were not verified recently.

    Callers should pass only manual records that are actually being used. API
    records and unused fallback rows are intentionally outside this check.
    """

    if not max_age_hours > 0:
        raise SurvivorDataError("Manual odds maximum age must be positive")
    checked_at = as_of or datetime.now(timezone.utc)
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise SurvivorDataError("Freshness check time must include a timezone")
    checked_at = checked_at.astimezone(timezone.utc)
    maximum_age = timedelta(hours=max_age_hours)
    future_tolerance = timedelta(minutes=5)
    problems: list[str] = []
    problem_records: list[OddsRecord] = []
    for record in records:
        prices = (
            f"{record.team_a} {record.moneyline_a:+g}, {record.team_b} {record.moneyline_b:+g}"
            if record.moneyline_a is not None and record.moneyline_b is not None
            else f"{record.team_a} {record.probability_a:.4%}, {record.team_b} {record.probability_b:.4%}"
        )
        label = (
            f"Week {record.week} {record.team_a} vs {record.team_b} ({record.game_id}); "
            f"saved {prices}; source {record.source}"
        )
        try:
            verified_at = _parse_aware_timestamp(record.observed_at, label=label)
        except StaleManualOddsError as exc:
            problems.append(str(exc))
            problem_records.append(record)
            continue
        age = checked_at - verified_at
        if age < -future_tolerance:
            problems.append(f"{label} has a future verification timestamp {record.observed_at}")
            problem_records.append(record)
        elif age > maximum_age:
            age_hours = age.total_seconds() / 3600.0
            problems.append(
                f"{label} was last verified {age_hours:.1f} hours ago at {record.observed_at}"
            )
            problem_records.append(record)
    if problems:
        commands = "\n".join(
            f"  python3 survivor.py confirm-manual-odds --season {record.season} "
            f"--game-id {record.game_id}"
            for record in problem_records
        )
        raise StaleManualOddsError(
            "Required manual odds are stale or unverified:\n  "
            + "\n  ".join(problems)
            + "\nCheck the prices, then update them or confirm unchanged values:\n"
            + commands
        )


def update_manual_odds(
    paths: DataPaths,
    *,
    season: int,
    game_id: str,
    moneyline_a: float,
    moneyline_b: float,
    source: str,
    observed_at: str | None = None,
) -> OddsRecord:
    """Insert or replace one schedule-backed manual moneyline pair."""

    schedule = load_schedule(paths.schedule(season), season=season)
    game = next((item for item in schedule if item.game_id == game_id), None)
    if game is None:
        raise SurvivorDataError(f"Game {game_id!r} is not in the {season} schedule")
    if not source.strip():
        raise SurvivorDataError("Manual odds source is required")
    try:
        probability_a, probability_b = remove_vig(float(moneyline_a), float(moneyline_b))
    except (TypeError, ValueError) as exc:
        raise SurvivorDataError(f"Invalid manual American moneylines: {exc}") from exc
    timestamp = observed_at or utc_now()
    record = OddsRecord(
        season=season,
        week=game.week,
        game_id=game.game_id,
        team_a=game.away_team,
        team_b=game.home_team,
        probability_a=probability_a,
        probability_b=probability_b,
        moneyline_a=float(moneyline_a),
        moneyline_b=float(moneyline_b),
        bookmakers="manual",
        source=source.strip(),
        observed_at=timestamp,
    )
    existing = _read_csv(paths.manual_odds) if paths.manual_odds.exists() else []
    retained = [
        row for row in existing
        if not (
            (row.get("season") or "").strip() == str(season)
            and (row.get("game_id") or "").strip() == game_id
        )
    ]
    retained.append(_odds_to_row(record))
    _write_csv(
        paths.manual_odds,
        MANUAL_ODDS_FIELDS,
        sorted(
            retained,
            key=lambda row: (
                int((row.get("season") or 0)),
                int((row.get("week") or 0)),
                (row.get("game_id") or ""),
            ),
        ),
    )
    return record


def confirm_manual_odds(
    paths: DataPaths,
    *,
    season: int,
    game_ids: Sequence[str],
    confirmed_at: str | None = None,
) -> list[OddsRecord]:
    """Refresh verification timestamps without changing saved prices."""

    requested = tuple(dict.fromkeys(game_id.strip() for game_id in game_ids if game_id.strip()))
    if not requested:
        raise SurvivorDataError("At least one game ID is required")
    schedule = load_schedule(paths.schedule(season), season=season)
    records = load_manual_odds(
        paths.manual_odds,
        schedule,
        season=season,
        weeks=set(range(1, 19)),
    )
    by_id = {record.game_id: record for record in records}
    missing = [game_id for game_id in requested if game_id not in by_id]
    if missing:
        raise SurvivorDataError(
            "No saved manual odds exist for game ID(s): " + ", ".join(missing)
        )
    timestamp = confirmed_at or utc_now()
    updated: list[OddsRecord] = []
    for game_id in requested:
        old = by_id[game_id]
        updated.append(
            OddsRecord(
                season=old.season,
                week=old.week,
                game_id=old.game_id,
                team_a=old.team_a,
                team_b=old.team_b,
                probability_a=old.probability_a,
                probability_b=old.probability_b,
                moneyline_a=old.moneyline_a,
                moneyline_b=old.moneyline_b,
                bookmakers=old.bookmakers,
                source=old.source,
                observed_at=timestamp,
                bookmaker_updates=old.bookmaker_updates,
            )
        )
    rows = _read_csv(paths.manual_odds)
    replacement = {record.game_id: _odds_to_row(record) for record in updated}
    rewritten = []
    for row in rows:
        row_game_id = (row.get("game_id") or "").strip()
        row_season = (row.get("season") or "").strip()
        if row_season == str(season) and row_game_id in replacement:
            rewritten.append(replacement[row_game_id])
        else:
            rewritten.append(row)
    _write_csv(paths.manual_odds, MANUAL_ODDS_FIELDS, rewritten)
    return updated


def apply_manual_odds(
    paths: DataPaths,
    *,
    season: int,
    game_ids: Sequence[str],
    max_age_hours: float = 24.0,
    as_of: datetime | None = None,
) -> list[OddsRecord]:
    """Promote verified manual games into the normalized look-ahead snapshot.

    This is intentionally game-scoped. It updates only explicitly requested
    games and preserves the other saved future games when sportsbooks stop
    publishing a complete future week.
    """

    requested = tuple(
        dict.fromkeys(game_id.strip() for game_id in game_ids if game_id.strip())
    )
    if not requested:
        raise SurvivorDataError("At least one game ID is required")
    schedule = load_schedule(paths.schedule(season), season=season)
    manual = load_manual_odds(
        paths.manual_odds,
        schedule,
        season=season,
        weeks=set(range(1, 19)),
    )
    manual_by_id = {record.game_id: record for record in manual}
    missing = [game_id for game_id in requested if game_id not in manual_by_id]
    if missing:
        raise SurvivorDataError(
            "No saved manual odds exist for game ID(s): " + ", ".join(missing)
        )
    selected = [manual_by_id[game_id] for game_id in requested]
    validate_manual_odds_freshness(
        selected,
        max_age_hours=max_age_hours,
        as_of=as_of,
    )

    existing = load_normalized_odds(paths.normalized_odds)
    requested_keys = {(season, game_id) for game_id in requested}
    retained = [
        record
        for record in existing
        if (record.season, record.game_id) not in requested_keys
    ]
    combined = retained + selected
    _write_csv(
        paths.normalized_odds,
        ODDS_FIELDS,
        (
            _odds_to_row(record)
            for record in sorted(
                combined,
                key=lambda item: (item.season, item.week, item.game_id),
            )
        ),
    )
    return selected


def validate_odds(records: Sequence[OddsRecord], schedule: Sequence[Game], *, season: int, weeks: set[int]) -> None:
    expected = {game.game_id: game for game in schedule if game.season == season and game.week in weeks}
    seen: set[str] = set()
    for record in records:
        if record.game_id in seen:
            raise SurvivorDataError(f"Duplicate odds record for {record.game_id}")
        seen.add(record.game_id)
        game = expected.get(record.game_id)
        if game is None:
            raise SurvivorDataError(f"Odds contain an unexpected game: {record.game_id}")
        if frozenset((record.team_a, record.team_b)) != game.teams:
            raise SurvivorDataError(f"Odds teams do not match schedule for {record.game_id}")
        if not (0.0 < record.probability_a < 1.0 and 0.0 < record.probability_b < 1.0):
            raise SurvivorDataError(f"Odds probabilities are out of range for {record.game_id}")
        if abs(record.probability_a + record.probability_b - 1.0) > 1e-9:
            raise SurvivorDataError(f"Odds probabilities do not sum to one for {record.game_id}")
    missing = sorted(
        (expected[game_id] for game_id in set(expected) - seen),
        key=lambda game: (game.week, game.kickoff, game.game_id),
    )
    if missing:
        details = "; ".join(
            f"Week {game.week}: {game.away_team} at {game.home_team} ({game.game_id})"
            for game in missing
        )
        raise MissingOddsError(
            "Automatic odds refresh did not cover these games: " + details
            + ". Add only those games to inputs/manual_odds.csv, then rerun refresh-odds."
        )


def _odds_to_row(record: OddsRecord) -> dict[str, Any]:
    row = asdict(record)
    for field in ("moneyline_a", "moneyline_b"):
        if row[field] is None:
            row[field] = ""
    return row


def load_normalized_odds(path: Path, *, season: int | None = None) -> list[OddsRecord]:
    if not path.exists():
        return []
    records: list[OddsRecord] = []
    for row in _read_csv(path):
        if not (row.get("season") or "").strip():
            continue
        row_season = int(row["season"])
        if season is not None and row_season != season:
            continue
        records.append(
            OddsRecord(
                season=row_season,
                week=int(row["week"]),
                game_id=row["game_id"],
                team_a=normalize_team(row["team_a"]),
                team_b=normalize_team(row["team_b"]),
                probability_a=float(row["probability_a"]),
                probability_b=float(row["probability_b"]),
                moneyline_a=float(row["moneyline_a"]) if row.get("moneyline_a") else None,
                moneyline_b=float(row["moneyline_b"]) if row.get("moneyline_b") else None,
                bookmakers=row.get("bookmakers", ""),
                source=row.get("source", ""),
                observed_at=row.get("observed_at", ""),
                bookmaker_updates=row.get("bookmaker_updates", ""),
            )
        )
    return records


def refresh_odds(
    paths: DataPaths,
    *,
    season: int,
    weeks: set[int],
    client: OddsAPIClient,
    regions: str = "us",
    bookmakers: str | None = None,
    manual_max_age_hours: float = 24.0,
    as_of: datetime | None = None,
) -> OddsRefreshResult:
    """Refresh requested weeks from the API, then consult manual rows only for gaps."""

    if not weeks or any(week < 1 or week > 18 for week in weeks):
        raise SurvivorDataError("At least one week from 1 through 18 is required")
    schedule = load_schedule(paths.schedule(season), season=season)
    api_error: str | None = None
    raw_path: Path | None = None
    response: APIResponse | None = None
    api_records: list[OddsRecord] = []
    try:
        response = client.fetch_odds(regions=regions, bookmakers=bookmakers)
        raw_path = paths.raw_data / (
            f"odds_{season}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(response.events, indent=2, sort_keys=True), encoding="utf-8")
        api_records = odds_records_from_api(response, schedule, season=season, weeks=weeks)
    except OddsAPIError as exc:
        api_error = str(exc)

    records_by_game = {record.game_id: record for record in api_records}
    expected_ids = {
        game.game_id for game in schedule if game.season == season and game.week in weeks
    }
    missing_ids = expected_ids - set(records_by_game)
    manual_records = load_manual_odds(paths.manual_odds, schedule, season=season, weeks=weeks)
    used_manual_records: list[OddsRecord] = []
    for record in manual_records:
        if record.game_id in missing_ids:
            records_by_game[record.game_id] = record
            used_manual_records.append(record)
    validate_manual_odds_freshness(
        used_manual_records,
        max_age_hours=manual_max_age_hours,
        as_of=as_of,
    )
    records = sorted(records_by_game.values(), key=lambda item: (item.week, item.game_id))
    validate_odds(records, schedule, season=season, weeks=weeks)

    existing = load_normalized_odds(paths.normalized_odds)
    retained = [
        record for record in existing
        if not (record.season == season and record.week in weeks)
    ]
    _write_csv(
        paths.normalized_odds,
        ODDS_FIELDS,
        (_odds_to_row(record) for record in sorted(retained + records, key=lambda item: (item.season, item.week, item.game_id))),
    )
    return OddsRefreshResult(
        records=records,
        api_records=len(api_records),
        manual_records=len(used_manual_records),
        api_error=api_error,
        remaining_requests=response.remaining_requests if response else None,
        raw_response_path=raw_path,
        manual_games=tuple(used_manual_records),
    )


def _winner_from_score_event(event: Mapping[str, Any]) -> tuple[frozenset[str], str | None, bool] | None:
    try:
        away = normalize_team(str(event["away_team"]))
        home = normalize_team(str(event["home_team"]))
    except (KeyError, SurvivorDataError):
        return None
    if not event.get("completed"):
        return frozenset((away, home)), None, False
    scores = {normalize_team(str(item["name"])): int(item["score"]) for item in event.get("scores") or []}
    if away not in scores or home not in scores:
        return None
    if scores[away] == scores[home]:
        return frozenset((away, home)), None, True
    return frozenset((away, home)), max((away, home), key=lambda team: scores[team]), False


def sync_results(
    path: Path,
    *,
    season: int,
    schedule: Sequence[Game],
    events: Sequence[Mapping[str, Any]],
) -> int:
    """Update pending picks from completed score events. Ties are recorded as losses."""

    outcomes: dict[frozenset[str], tuple[str | None, bool]] = {}
    for event in events:
        outcome = _winner_from_score_event(event)
        if outcome is not None:
            teams, winner, tied = outcome
            outcomes[teams] = (winner, tied)
    schedule_lookup = {(game.week, game.teams): game for game in schedule if game.season == season}
    picks = load_picks(path)
    updates = 0
    merged: list[PickRecord] = []
    for pick in picks:
        if pick.season != season or pick.result != "pending":
            merged.append(pick)
            continue
        candidates = [teams for week, teams in schedule_lookup if week == pick.week and pick.team in teams]
        if len(candidates) != 1 or candidates[0] not in outcomes:
            merged.append(pick)
            continue
        winner, tied = outcomes[candidates[0]]
        result = "tie" if tied else "win" if winner == pick.team else "loss"
        merged.append(PickRecord(**{**asdict(pick), "result": result}))
        updates += 1
    _validate_rebuy_timelines(merged)
    _write_csv(path, PICK_FIELDS, (_pick_to_row(pick) for pick in merged))
    return updates


def project_status(paths: DataPaths, *, season: int) -> dict[str, Any]:
    schedule = load_schedule(paths.schedule(season), season=season)
    picks = load_picks(paths.picks, season=season)
    odds = load_normalized_odds(paths.normalized_odds, season=season)
    states = entry_states(picks, season=season)
    completed_weeks = [pick.week for pick in picks if pick.result in {"win", "loss", "tie"}]
    current_week = min(max(completed_weeks, default=0) + 1, 18)
    odds_weeks = sorted({record.week for record in odds})
    return {
        "season": season,
        "current_week": current_week,
        "schedule_games": len(schedule),
        "schedule_weeks": sorted({game.week for game in schedule}),
        "odds_weeks": odds_weeks,
        "latest_odds": max((record.observed_at for record in odds), default=None),
        "entries": [
            {
                "contest": state.contest,
                "entry_id": state.entry_id,
                "active": state.active,
                "used_teams": sorted(state.used_teams),
                "last_week": state.last_week,
                "rebuys_used": state.rebuys_used,
            }
            for state in states
        ],
    }
