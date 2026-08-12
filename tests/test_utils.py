from __future__ import annotations

import argparse
import math
from collections import namedtuple
from types import SimpleNamespace

import pytest

from omnitop.app import (
    fmt_bytes,
    fmt_duration,
    interval_arg,
    jsonable,
    positive_int_arg,
    rate,
    read_cpu_temperature,
    same_process_start,
    sanitize_text,
    shorten_mount,
    snapshot_payload,
)


@pytest.mark.parametrize("raw, expected", [("0.2", 0.2), ("1", 1.0), ("60", 60.0)])
def test_interval_arg_accepts_supported_range(raw: str, expected: float) -> None:
    assert interval_arg(raw) == expected


@pytest.mark.parametrize("raw", ["0", "0.19", "61", "nan", "inf", "-inf", "abc"])
def test_interval_arg_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        interval_arg(raw)


@pytest.mark.parametrize("raw, expected", [("1", 1), ("42", 42)])
def test_positive_int_arg(raw: str, expected: int) -> None:
    assert positive_int_arg(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "x"])
def test_positive_int_arg_rejects_non_positive(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int_arg(raw)


def test_rate_handles_growth_reset_and_non_finite_values() -> None:
    assert rate(200, 100, 2) == 50
    assert rate(50, 100, 2) == 0
    assert rate(math.inf, 100, 2) == 0
    assert rate(200, 100, 0) == 100_000


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "n/a"),
        (0, "0B"),
        (1024, "1.0KiB"),
        (10 * 1024**3, "10GiB"),
        (math.nan, "n/a"),
    ],
)
def test_fmt_bytes(value: float | None, expected: str) -> None:
    assert fmt_bytes(value) == expected


def test_duration_and_mount_edge_cases() -> None:
    assert fmt_duration(-1) == "0m"
    assert fmt_duration(math.inf) == "n/a"
    assert shorten_mount("/abcdef", 3) == "/ab"
    assert shorten_mount("/abcdef", 6) == "...def"


def test_sanitize_text_removes_control_sequences_and_caps_length() -> None:
    assert sanitize_text("ok\x1b[31m\n bad") == "ok [31m bad"
    assert sanitize_text("abc\x00def") == "abc def"
    assert sanitize_text("abcdefgh", 5) == "abcd…"


def test_same_process_start_uses_small_tolerance() -> None:
    assert same_process_start(10.005, 10.0)
    assert not same_process_start(10.1, 10.0)


def test_jsonable_handles_namedtuples_sets_dataclasses_and_non_finite(make_process) -> None:
    Pair = namedtuple("Pair", "left right")
    payload = {
        "pair": Pair(1, 2),
        "set": {"b", "a"},
        "row": make_process(7),
        "bad": math.nan,
    }
    result = jsonable(payload)
    assert result["pair"] == {"left": 1, "right": 2}
    assert result["set"] == ["a", "b"]
    assert result["row"]["pid"] == 7
    assert result["bad"] is None


def test_snapshot_payload_removes_internal_gpu_process_map() -> None:
    snapshot = {
        "time": 0.0,
        "gpu": {"proc_map": {1: {"gpu_ids": {"0"}}}, "gpus": []},
        "warnings": [],
    }
    payload = snapshot_payload(snapshot)
    assert "proc_map" not in payload["gpu"]
    assert payload["generated_at"] == "1970-01-01T00:00:00+00:00"


def test_cpu_temperature_prefers_cpu_sensor_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    sensors = {
        "nvme": [SimpleNamespace(current=92.0, label="Composite")],
        "coretemp": [SimpleNamespace(current=67.0, label="Package id 0")],
        "other": [SimpleNamespace(current=70.0, label="board")],
    }
    monkeypatch.setattr("omnitop.app.psutil.sensors_temperatures", lambda fahrenheit=False: sensors)
    assert read_cpu_temperature() == 67.0
