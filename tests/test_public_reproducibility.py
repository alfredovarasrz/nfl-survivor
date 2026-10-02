from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
from unittest.mock import patch

import pytest
from replay import replay_report
from data import APIResponse, OddsAPIClient, OddsAPIError, odds_records_from_api, load_schedule
from examples.demo import build_demo
from reporting import _markdown
import survivor
import test_weekly_run as weekly_tests


def test_saved_demo_reruns_and_accepts_changed_inputs():
    for pool in ("small", "large"):
        report = build_demo(pool)
        replayed = replay_report(json.loads(json.dumps(report)))
        assert json.loads(json.dumps(replayed["recommendation"])) == json.loads(json.dumps(report["recommendation"]))
        assert replayed["replay"]["mode"] == "saved_inputs_offline"
        assert set(replayed) <= {
            "run_id", "generated_at", "season", "pool", "contest", "entries",
            "first_week", "end_week", "pick_history", "probability_snapshot",
            "optimizer_parameters", "input_description", "odds_request",
            "manual_max_age_hours", "odds_refresh", "recommendation", "replay"}
        edited = deepcopy(report)
        edited["probability_snapshot"][0]["probability_a"] = 0.8
        edited["probability_snapshot"][0]["probability_b"] = 0.2
        rerun = replay_report(edited)
        assert rerun["probability_snapshot"][0]["probability_a"] == 0.8
        assert rerun["recommendation"] != report["recommendation"]


def test_sampled_report_is_honest_and_repeatable():
    report = build_demo("large")
    result = report["recommendation"]
    assert result["exact_within_candidate_sets"] is False
    assert result["evaluated_portfolios"] == 20
    assert result["total_candidate_portfolios"] > 20
    assert result == build_demo("large")["recommendation"]
    markdown = _markdown(report, Path("synthetic.json"))
    assert "sampled and locally improved" in markdown
    assert "Every generated candidate combination was evaluated exactly" not in markdown
    exact = deepcopy(report)
    exact["optimizer_parameters"]["max_evaluations"] = 1000
    exact = replay_report(exact)
    assert exact["recommendation"]["exact_within_candidate_sets"] is True
    assert "Every generated candidate combination was evaluated exactly" in _markdown(exact, Path("synthetic.json"))


def test_errors_redact_plain_encoded_and_provider_reflected_credentials():
    key = "synthetic secret+/key"
    def opener(request, **kwargs):
        raise urllib.error.URLError(request.full_url + " plain=" + key)
    client = OddsAPIClient(api_key=key, opener=opener)
    with pytest.raises(OddsAPIError) as captured:
        client.fetch_odds()
    message = str(captured.value)
    assert key not in message
    assert "synthetic+secret" not in message
    assert "[REDACTED]" in message
    assert captured.value.__suppress_context__ is True
    assert key not in client._safe_error("Provider says: " + key)


def test_saved_inputs_retain_source_and_original_times():
    report = build_demo("small")
    rows = report["probability_snapshot"]
    assert rows[0]["observed_at"] == "2030-09-01T11:00:00+00:00"
    assert "2030-09-01T10:55:00+00:00" in rows[0]["bookmaker_updates"]
    edited = deepcopy(report)
    edited["probability_snapshot"][0]["bookmaker_updates"] = ""
    rerun = replay_report(edited)
    assert rerun["probability_snapshot"] == edited["probability_snapshot"]
    assert rerun["generated_at"] == edited["generated_at"]
    assert rerun["probability_snapshot"][0]["source"] == rows[0]["source"]


def test_weekly_cli_saves_seed_inputs_and_reruns_offline():
    fixture = weekly_tests.WeeklyRunTests()
    fixture.setUp()
    try:
        events = fixture._events()
        for event in events:
            event["bookmakers"][0]["last_update"] = "2026-09-01T00:00:00Z"
        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse(events)
        with patch.object(survivor, "OddsAPIClient", return_value=Client()):
            survivor.main(["--root", str(fixture.root), "weekly-run", "--season", "2026",
                           "--pool", "large", "--other-survivors", "10", "--end-week", "2",
                           "--candidate-paths", "4", "--paths-per-current-team", "1",
                           "--beam-width", "20", "--max-evaluations", "20", "--seed", "23"])
        path = next(fixture.paths.outputs.rglob("*.json"))
        report = json.loads(path.read_text())
        assert report["optimizer_parameters"]["seed"] == 23
        assert not Path(report["odds_refresh"]["raw_response_path"]).is_absolute()
        assert report["probability_snapshot"][0]["bookmaker_updates"]
        replayed = replay_report(report)
        assert json.loads(json.dumps(replayed["recommendation"])) == report["recommendation"]
        with patch.object(survivor, "OddsAPIClient", side_effect=AssertionError("Replay accessed live API")):
            assert survivor.main(["--root", str(fixture.root), "replay", str(path), "--save"]) == 0
        assert len(list(fixture.paths.outputs.rglob("*.json"))) == 2
    finally:
        fixture.tearDown()


def test_replay_cli_subprocess_is_offline_and_portable(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(build_demo("small")))
    run = subprocess.run([sys.executable, str(Path(survivor.__file__)), "replay", str(path)],
                         capture_output=True, text=True, check=True)
    output = json.loads(run.stdout)
    assert output["replay"]["mode"] == "saved_inputs_offline"
    assert output["recommendation"]["solver_status"] == "optimal"


def test_global_mode_replay_preserves_decision_and_certificate_and_limit_bound():
    for nodes in (1, 100_000):
        report = build_demo("large", search="global", global_max_nodes=nodes)
        replay = replay_report(json.loads(json.dumps(report)))
        assert json.loads(json.dumps(replay["recommendation"])) == json.loads(json.dumps(report["recommendation"]))
        assert report["recommendation"]["search_scope"] == "all_legal_fixed_paths"
        if nodes == 1:
            assert not report["recommendation"]["globally_optimal_over_all_legal_paths"]
            assert report["recommendation"]["termination"] == "node_limit"
        else:
            assert report["recommendation"]["globally_optimal_over_all_legal_paths"]
        markdown = _markdown(report, Path("synthetic.json"))
        assert "Global bounds:" in markdown
        assert "Candidate combinations were sampled" not in markdown


def test_global_cli_reports_honest_limits_and_replays():
    fixture = weekly_tests.WeeklyRunTests()
    fixture.setUp()
    try:
        class Client:
            def fetch_odds(self, **kwargs):
                return APIResponse(fixture._events())
        with patch.object(survivor, "OddsAPIClient", return_value=Client()):
            assert survivor.main(["--root", str(fixture.root), "weekly-run", "--season", "2026",
                    "--pool", "large", "--other-survivors", "10", "--end-week", "2",
                    "--search", "global", "--global-max-nodes", "1", "--global-max-seconds", "0"]) == 0
        path = next(fixture.paths.outputs.rglob("*.json"))
        report = json.loads(path.read_text())
        assert report["recommendation"]["termination"] == "node_limit"
        assert report["optimizer_parameters"]["search_method"] == "global"
        assert json.loads(json.dumps(replay_report(report)["recommendation"])) == report["recommendation"]
    finally:
        fixture.tearDown()


def test_default_client_and_offline_modes_never_access_keychain_or_environment_files(monkeypatch):
    import data
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("SURVIVOR_PUBLIC_USE_KEYCHAIN", raising=False)
    with patch.object(data, "load_odds_api_key_from_keychain", side_effect=AssertionError("Unexpected credential access")):
        client = data.OddsAPIClient()
        assert client.api_key is None
        with pytest.raises(data.OddsAPIError, match="not configured"):
            client.fetch_odds()
        for search in ("candidate", "global"):
            report = build_demo("large", search=search, global_max_nodes=1)
            replay_report(report)
    assert data.ODDS_KEYCHAIN_SERVICE == "NFL Survivor Public The Odds API"
    with patch.object(data, "load_odds_api_key_from_keychain", return_value="synthetic-owned-key") as loader:
        assert data.OddsAPIClient(use_keychain=True).api_key == "synthetic-owned-key"
        loader.assert_called_once()


def test_api_failures_do_not_log_credentials_or_include_them_in_tracebacks(capsys):
    import traceback
    key = "synthetic-owned-secret+/key"
    def opener(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 401, "Synthetic denial: " + key, {}, None)
    client = OddsAPIClient(api_key=key, opener=opener)
    try:
        client.fetch_odds()
    except OddsAPIError:
        trace = traceback.format_exc()
    else:
        raise AssertionError("Synthetic failure did not occur")
    assert key not in trace
    assert "synthetic-owned-secret%2B" not in trace
    assert "[REDACTED]" in trace
    captured_output = capsys.readouterr()
    assert captured_output.out == ""
    assert captured_output.err == ""


def test_provider_error_payload_is_redacted_and_explicit_owned_environment_key_is_isolated(monkeypatch):
    import data
    import io
    key = "synthetic-owned-provider-key"
    monkeypatch.setenv("THE_ODDS_API_KEY", key)
    monkeypatch.delenv("SURVIVOR_PUBLIC_USE_KEYCHAIN", raising=False)
    class Response(io.BytesIO):
        headers = {}
    def opener(request, **kwargs):
        return Response(json.dumps({"message": "Denied for key " + key}).encode())
    with patch.object(data, "load_odds_api_key_from_keychain", side_effect=AssertionError("Unexpected credential access")):
        client = data.OddsAPIClient(opener=opener)
        assert client.api_key == key
        with pytest.raises(data.OddsAPIError) as captured:
            client.fetch_odds()
        assert key not in str(captured.value)
        assert "[REDACTED]" in str(captured.value)
