from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnitop.app import Sampler, UIState, process_start_token


def test_sampler_isolates_collector_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = Sampler(gpu_enabled=False, collect_processes=False)
    monkeypatch.setattr(sampler, "_sample_disks", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")))
    try:
        snapshot = sampler.sample(UIState(interval=1.0))
    finally:
        sampler.close()
    assert snapshot["disks"] == {"partitions": [], "io": []}
    assert any("disk collection failed" in warning for warning in snapshot["warnings"])


def test_process_start_token_prefers_monotonic_backend() -> None:
    backend = SimpleNamespace(create_time=lambda monotonic=False: 123.5 if monotonic else 999.0)
    proc = SimpleNamespace(_proc=backend, create_time=lambda: 999.0)
    assert process_start_token(proc, boot_time=100.0) == 123.5


def test_process_start_token_has_public_api_fallback() -> None:
    proc = SimpleNamespace(create_time=lambda: 150.0)
    assert process_start_token(proc, boot_time=100.0) == 50.0


def test_gpu_only_sampling_returns_only_nvml_processes() -> None:
    sampler = Sampler(gpu_enabled=True, history_len=4)
    try:
        first = sampler.sample(UIState(interval=1.0, gpu_only=True))
        second = sampler.sample(UIState(interval=1.0, gpu_only=True))
    finally:
        sampler.close()
    if not second["gpu"]["available"]:
        pytest.skip("NVML is unavailable on this test host")
    gpu_pids = {
        process["pid"] if isinstance(process, dict) else process.pid
        for gpu in second["gpu"]["gpus"]
        for process in gpu.processes
    }
    assert {row.pid for row in second["processes"]} <= gpu_pids
    assert second["sequence"] == first["sequence"] + 1
