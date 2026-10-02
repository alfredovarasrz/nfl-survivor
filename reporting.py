"""Saved JSON and readable Markdown reports for weekly recommendations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from data import DataPaths


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _search_lines(recommendation):
    if recommendation.get("search_scope") == "all_legal_fixed_paths":
        return ["Search scope: all legal fixed paths; no candidate truncation.",
                f"Global status: {recommendation['termination']}; certified within tolerance: {recommendation['globally_optimal_over_all_legal_paths']}.",
                f"Global bounds: [{recommendation['global_lower_bound']:.12g}, {recommendation['global_upper_bound']:.12g}]; absolute gap: {recommendation['global_absolute_gap']:.12g}.",
                f"Visited nodes: {recommendation['visited_nodes']}; complete portfolios evaluated: {recommendation['completed_portfolios']}.",
                f"Absolute tolerance: {recommendation['absolute_gap_tolerance']}; numerical bound slack: {recommendation['numerical_bound_slack']}."]
    return [("Every generated candidate combination was evaluated exactly."
             if recommendation["exact_within_candidate_sets"]
             else "Candidate combinations were sampled and locally improved; the candidate set was not exhausted."),
            "No global certificate covers all legal season paths.",
            f"Search method: {recommendation['method']}; evaluated {recommendation['evaluated_portfolios']} of {recommendation['total_candidate_portfolios']} combinations."]


def _markdown(report: Mapping[str, Any], json_path: Path) -> str:
    recommendation = report["recommendation"]
    refresh = report["odds_refresh"]
    lines = [
        f"# {report['contest'].replace('_', ' ').title()} recommendation",
        "",
        f"- Season: {report['season']}",
        f"- Decision week: {report['first_week']}",
        f"- Generated: {report['generated_at']}",
        f"- Saved odds horizon: Weeks {report['first_week']} through {report['end_week']}",
        f"- API games: {refresh['api_records']}",
        f"- Manual fallback games: {refresh['manual_records']}",
        "",
    ]
    if report["pool"] == "small":
        picks = recommendation["picks"]
        current_team = picks.get(str(report["first_week"]), picks.get(report["first_week"]))
        lines.extend(
            [
                f"## Recommended pick: {current_team}",
                "",
                f"Expected remaining weeks survived: {recommendation['expected_remaining_weeks']:.6f}",
                f"Solver status: {recommendation['solver_status']}; relative optimality gap: {recommendation['optimality_gap']}.",
                "",
                "## Provisional path",
                "",
            ]
        )
        for week, team in sorted(picks.items(), key=lambda item: int(item[0])):
            survival_by_week = recommendation["survival_by_week"]
            survival = survival_by_week.get(week, survival_by_week.get(str(week)))
            lines.append(f"- Week {week}: {team} ({survival:.4%} cumulative survival)")
        lines.extend(["", "## Current-week alternatives", ""])
        for alternative in recommendation["current_week_alternatives"]:
            lines.append(
                f"- {alternative['team']}: {alternative['expected_weeks']:.6f} expected weeks "
                f"({alternative['expected_weeks_lost']:.6f} below best)"
            )
    else:
        lines.extend(
            [
                "## Recommended current picks",
                "",
            ]
        )
        for entry_id, picks in recommendation["paths"].items():
            current_team = picks.get(str(report["first_week"]), picks.get(report["first_week"]))
            lines.append(f"- {entry_id}: {current_team}")
        lines.extend(
            [
                "",
                f"Expected prize share: {recommendation['expected_prize_share']:.6%}",
                f"Probability any owned entry survives: {recommendation['probability_any_survives']:.4%}",
                f"Expected owned finalists: {recommendation['expected_final_survivors']:.4f}",
                "",
                *_search_lines(recommendation),
            ]
        )
    seed = report["optimizer_parameters"].get("seed")
    if seed is not None:
        lines.extend(["", f"Candidate search seed: {seed}"])
    if refresh["manual_games"]:
        lines.extend(["", "## Manual fallback verification", ""])
        for game in refresh["manual_games"]:
            lines.append(
                f"- Week {game['week']} {game['team_a']} vs {game['team_b']}: "
                f"{game['moneyline_a']} / {game['moneyline_b']}, "
                f"source {game['source']}, verified {game['observed_at']}"
            )
    if refresh.get("api_warning"):
        lines.extend(["", f"API warning: {refresh['api_warning']}"])
    lines.extend(["", f"Complete machine-readable record: `{json_path.name}`", ""])
    return "\n".join(lines)


def save_weekly_reports(
    paths: DataPaths,
    report: Mapping[str, Any],
    *,
    season: int,
    week: int,
    contest: str,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Save collision-safe JSON and Markdown copies of one successful run."""

    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = paths.outputs / str(season) / f"week_{week:02d}"
    safe_contest = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in contest
    )
    stem = f"{identifier}_{safe_contest}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    if json_path.exists() or markdown_path.exists():
        raise ValueError(f"Weekly report already exists for run ID {identifier}")
    _atomic_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        _atomic_text(markdown_path, _markdown(report, json_path))
    except Exception:
        json_path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path
