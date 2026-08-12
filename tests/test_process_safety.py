from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import pytest

from omnitop.app import UIState, handle_key, preserve_selection, send_signal


class FakeBackend:
    def __init__(self, token: float) -> None:
        self.token = token

    def create_time(self, monotonic: bool = False) -> float:
        assert monotonic
        return self.token


class FakeProcess:
    def __init__(self, token: float) -> None:
        self._proc = FakeBackend(token)


def test_send_signal_refuses_pid_one_and_self(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("omnitop.app.os.kill", lambda pid, sig: calls.append((pid, sig)))
    state = UIState()
    send_signal(1, signal.SIGTERM, state)
    send_signal(os.getpid(), signal.SIGTERM, state)
    assert calls == []


def test_send_signal_refuses_reused_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("omnitop.app.psutil.Process", lambda _pid: FakeProcess(20.0))
    monkeypatch.setattr("omnitop.app.os.kill", lambda pid, sig: calls.append((pid, sig)))
    state = UIState()
    send_signal(4242, signal.SIGTERM, state, expected_start_token=10.0)
    assert calls == []
    assert "identity changed" in state.message


def test_send_signal_allows_matching_process_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("omnitop.app.psutil.Process", lambda _pid: FakeProcess(10.0))
    monkeypatch.setattr("omnitop.app.os.kill", lambda pid, sig: calls.append((pid, sig)))
    state = UIState()
    send_signal(4242, signal.SIGTERM, state, expected_start_token=10.0)
    assert calls == [(4242, signal.SIGTERM)]


def test_preserve_selection_follows_pid_and_start_token(make_process) -> None:
    state = UIState(selected=0, selected_pid=20, selected_start_token=2.0)
    snapshot = {"processes": [make_process(10, start_token=1.0), make_process(20, start_token=2.0)]}
    preserve_selection(state, snapshot)
    assert state.selected == 1


def test_preserve_selection_does_not_follow_reused_pid(make_process) -> None:
    state = UIState(selected=0, selected_pid=20, selected_start_token=2.0)
    snapshot = {"processes": [make_process(20, start_token=9.0), make_process(30, start_token=3.0)]}
    preserve_selection(state, snapshot)
    assert state.selected == 0
    assert state.selected_start_token == 9.0


def test_kill_key_captures_process_identity(make_process) -> None:
    state = UIState()
    row = make_process(42, start_token=12.0)
    snapshot = {"processes": [row]}
    assert handle_key("k", state, snapshot, SimpleNamespace())
    assert state.pending_kill_pid == 42
    assert state.pending_kill_start_token == 12.0
