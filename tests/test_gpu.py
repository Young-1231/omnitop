from __future__ import annotations

from types import SimpleNamespace

from omnitop.app import GpuInfo, GpuProcess, NvmlSampler


class FakeProcessQueries:
    @staticmethod
    def nvmlDeviceGetComputeRunningProcesses_v3(_handle):
        return [
            SimpleNamespace(pid=10, usedGpuMemory=100),
            SimpleNamespace(pid=20, usedGpuMemory=(1 << 64) - 1),
        ]

    @staticmethod
    def nvmlDeviceGetGraphicsRunningProcesses_v3(_handle):
        return [
            SimpleNamespace(pid=10, usedGpuMemory=120),
            SimpleNamespace(pid=30, usedGpuMemory=50),
        ]


class FakeDetailQueries:
    NVML_CLOCK_SM = 0
    NVML_CLOCK_MEM = 1
    NVML_PCIE_UTIL_RX_BYTES = 2
    NVML_PCIE_UTIL_TX_BYTES = 3

    def __init__(self) -> None:
        self.pcie_calls = 0

    @staticmethod
    def nvmlDeviceGetFanSpeed(_handle):
        return 40

    @staticmethod
    def nvmlDeviceGetPerformanceState(_handle):
        return 0

    @staticmethod
    def nvmlDeviceGetClockInfo(_handle, clock):
        return 1000 + clock

    @staticmethod
    def nvmlDeviceGetEncoderUtilization(_handle):
        return 1, 1000

    @staticmethod
    def nvmlDeviceGetDecoderUtilization(_handle):
        return 2, 1000

    def nvmlDeviceGetPcieThroughput(self, _handle, metric):
        self.pcie_calls += 1
        return 10 + metric


def gpu_info(index: int, processes: list[GpuProcess]) -> GpuInfo:
    return GpuInfo(
        index=index,
        name=f"GPU {index}",
        uuid=f"uuid-{index}",
        memory_used=1024,
        memory_total=2048,
        gpu_util=50.0,
        mem_util=20.0,
        temp_c=60.0,
        power_w=100.0,
        power_limit_w=300.0,
        fan_pct=None,
        perf_state=None,
        sm_clock_mhz=None,
        mem_clock_mhz=None,
        pcie_rx_bps=None,
        pcie_tx_bps=None,
        encoder_util=None,
        decoder_util=None,
        processes=processes,
    )


def test_gpu_process_queries_deduplicate_compute_and_graphics_memory() -> None:
    sampler = NvmlSampler(enabled=False)
    sampler.nvml = FakeProcessQueries()
    rows = sampler._gpu_processes(object(), gpu_index=0, gpu_uuid="uuid")
    by_pid = {row.pid: row for row in rows}
    assert by_pid[10].kind == "C+G"
    assert by_pid[10].used_memory == 120
    assert by_pid[20].used_memory == 0
    assert by_pid[30].kind == "G"


def test_gpu_sample_sums_memory_once_per_device(monkeypatch) -> None:
    sampler = NvmlSampler(enabled=False)
    sampler.enabled = True
    sampler.available = True
    sampler.nvml = SimpleNamespace(nvmlDeviceGetCount=lambda: 2)
    sampler.count = 2
    infos = [
        gpu_info(0, [GpuProcess(10, 0, "u0", "C+G", 100)]),
        gpu_info(1, [GpuProcess(10, 1, "u1", "C", 200)]),
    ]
    monkeypatch.setattr(sampler, "_sample_one", lambda index, detailed=False: infos[index])
    result = sampler.sample()
    assert result["proc_map"][10]["gpu_mem"] == 300
    assert result["proc_map"][10]["gpu_ids"] == {"0", "1"}
    assert result["proc_map"][10]["kinds"] == {"C", "G"}


def test_unavailable_gpu_result_has_explicit_monitor_semantics() -> None:
    result = NvmlSampler(enabled=False).sample()
    assert result["available"] is False
    assert result["monitor_only"] is True
    assert "--no-gpu" in result["error"]


def test_expensive_gpu_detail_metrics_are_cached(monkeypatch) -> None:
    sampler = NvmlSampler(enabled=False)
    queries = FakeDetailQueries()
    sampler.nvml = queries
    times = iter([10.0, 11.0])
    monkeypatch.setattr("omnitop.app.time.monotonic", lambda: next(times))
    first = sampler._detail_metrics(object(), index=0)
    second = sampler._detail_metrics(object(), index=0)
    assert first == second
    assert queries.pcie_calls == 2
