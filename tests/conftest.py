from __future__ import annotations

import pytest

from omnitop.app import ProcessRow


def _make_process(
    pid: int,
    *,
    cpu: float = 0.0,
    gpu_mem: int = 0,
    gpu_ids: str = "",
    start_token: float | None = None,
) -> ProcessRow:
    token = float(pid) if start_token is None else start_token
    return ProcessRow(
        pid=pid,
        user="tester",
        name=f"proc-{pid}",
        status="running",
        cpu_pct=cpu,
        mem_pct=1.0,
        rss=1024,
        vms=2048,
        threads=1,
        read_bps=0.0,
        write_bps=0.0,
        gpu_mem=gpu_mem,
        gpu_ids=gpu_ids,
        gpu_kinds="C" if gpu_ids else "",
        command=f"python worker-{pid}.py",
        create_time=1_700_000_000.0 + token,
        start_token=token,
    )


@pytest.fixture
def make_process():
    return _make_process
