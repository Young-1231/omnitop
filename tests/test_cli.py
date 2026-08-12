from __future__ import annotations

import json

import pytest

from omnitop.app import main, parse_args


def test_parse_args_supports_product_modes() -> None:
    args = parse_args(["--gpu-only", "--user", "alice", "--sort", "gpu", "--interval", "2"])
    assert args.gpu_only is True
    assert args.user == "alice"
    assert args.sort == "gpu"
    assert args.interval == 2.0


@pytest.mark.parametrize(
    "argv",
    [
        ["--count", "2"],
        ["--pretty"],
        ["--json", "--count", "2", "--pretty"],
        ["--gpu-only", "--no-processes"],
        ["--gpu-only", "--no-gpu"],
        ["--diagnose", "--json", "--count", "2"],
        ["--json", "--rows", "3"],
        ["--history", "3601"],
    ],
)
def test_parse_args_rejects_conflicting_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_json_cli_emits_parseable_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--json", "--no-gpu", "--no-processes", "--interval", "0.2"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["schema_version"] == 1
    assert payload["version"] == "2.0.0"
    assert payload["gpu"]["available"] is False
    assert "proc_map" not in payload["gpu"]


def test_diagnostics_json_is_fast_and_explicit(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--diagnose", "--json", "--no-gpu"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["omnitop_version"] == "2.0.0"
    assert payload["gpu"]["monitor_only"] is True
    assert "running-process queries" in payload["gpu"]["occupancy_source"]


def test_non_tty_defaults_to_one_snapshot(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--no-gpu", "--no-processes", "--interval", "0.2", "--rows", "1", "--no-color"])
    output = capsys.readouterr().out
    assert code == 0
    assert "snapshot mode" in output
    assert "OmniTop 2.0.0" in output
    assert "SNAPSHOT" in output
