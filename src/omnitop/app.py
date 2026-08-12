#!/usr/bin/env python3
"""OmniTop's collectors, terminal UI, and command-line application.

The interactive view is intentionally backed by the same snapshot model as the
JSON interface so operators and automation see consistent values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import platform
import pwd
import select
import signal
import socket
import sys
import termios
import time
import tty
from collections import defaultdict, deque
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import __version__

try:
    import psutil
except ImportError as exc:  # pragma: no cover - startup guard
    print("Missing dependency: psutil. Install with: python3 -m pip install psutil", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:  # pragma: no cover - startup guard
    print("Missing dependency: rich. Install with: python3 -m pip install rich", file=sys.stderr)
    raise SystemExit(2) from exc


LOGGER = logging.getLogger("omnitop")
LOGGER.addHandler(logging.NullHandler())
SCHEMA_VERSION = 1
BYTES_NA = -1
MIN_INTERVAL = 0.2
MAX_INTERVAL = 60.0
SORT_KEYS = ("cpu", "mem", "gpu", "io", "pid", "name")
GPU_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
)


@dataclass
class GpuProcess:
    pid: int
    gpu_index: int
    gpu_uuid: str
    kind: str
    used_memory: int


@dataclass
class GpuInfo:
    index: int
    name: str
    uuid: str
    memory_used: int
    memory_total: int
    gpu_util: float | None
    mem_util: float | None
    temp_c: float | None
    power_w: float | None
    power_limit_w: float | None
    fan_pct: float | None
    perf_state: str | None
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None
    pcie_rx_bps: float | None
    pcie_tx_bps: float | None
    encoder_util: float | None
    decoder_util: float | None
    processes: list[GpuProcess] = field(default_factory=list)
    error: str = ""


@dataclass
class ProcessRow:
    pid: int
    user: str
    name: str
    status: str
    cpu_pct: float
    mem_pct: float
    rss: int
    vms: int
    threads: int
    read_bps: float
    write_bps: float
    gpu_mem: int
    gpu_ids: str
    gpu_kinds: str
    command: str
    create_time: float | None
    start_token: float | None

    @property
    def io_bps(self) -> float:
        return self.read_bps + self.write_bps

    @property
    def identity(self) -> tuple[int, float]:
        """Return a PID-reuse-safe identity for caches and destructive actions."""

        return self.pid, float(self.start_token or 0.0)


@dataclass
class UIState:
    interval: float = 1.0
    interactive: bool = True
    sort_key: str = "cpu"
    reverse: bool = True
    paused: bool = False
    show_help: bool = False
    show_details: bool = False
    show_env: bool = False
    show_full: bool = False
    show_per_core: bool = False
    show_all_ifaces: bool = False
    show_all_mounts: bool = False
    filter_text: str = ""
    user_filter: str = ""
    gpu_only: bool = False
    input_mode: str | None = None
    filter_buffer: str = ""
    selected: int = 0
    selected_pid: int | None = None
    selected_start_token: float | None = None
    scroll: int = 0
    page_size: int = 12
    forced_rows: int | None = None
    pending_kill_pid: int | None = None
    pending_kill_start_token: float | None = None
    message: str = ""
    message_until: float = 0.0

    def set_message(self, message: str, ttl: float = 4.0) -> None:
        self.message = message
        self.message_until = time.monotonic() + ttl

    def visible_message(self) -> str:
        if self.message and time.monotonic() < self.message_until:
            return self.message
        return ""


class RawTerminal:
    def __init__(self) -> None:
        self.fd: int | None = None
        self.old_settings: list[Any] | None = None

    def __enter__(self) -> RawTerminal:
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


class TerminationRequested(Exception):
    """Raised from a termination signal so terminal state can be restored."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class GracefulSignals:
    def __init__(self) -> None:
        self.previous: dict[signal.Signals, Any] = {}

    def __enter__(self) -> GracefulSignals:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    @staticmethod
    def _handle(signum: int, _frame: Any) -> None:
        raise TerminationRequested(signum)


class NvmlSampler:
    """Collect NVIDIA metrics while degrading cleanly when NVML is unavailable.

    NVML may keep ``/dev/nvidia*`` descriptors open while this sampler is alive.
    Those descriptors are a monitoring connection, not evidence of a CUDA
    compute context; process occupancy comes exclusively from NVML's running
    process queries.
    """

    def __init__(self, enabled: bool = True, retry_seconds: float = 30.0) -> None:
        self.enabled = enabled
        self.retry_seconds = max(1.0, retry_seconds)
        self.available = False
        self.error = ""
        self.driver = ""
        self.nvml: Any | None = None
        self.count = 0
        self.detail_refresh_seconds = 5.0
        self._detail_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        self._initialized = False
        self._last_init_attempt = float("-inf")
        if enabled:
            self._initialize()
        else:
            self.error = "GPU collection disabled by --no-gpu"

    def _initialize(self) -> None:
        self._last_init_attempt = time.monotonic()
        self.error = ""
        try:
            import pynvml  # type: ignore

            self.nvml = pynvml
            pynvml.nvmlInit()
            self._initialized = True
            self.count = int(pynvml.nvmlDeviceGetCount())
            self.driver = sanitize_text(_decode(pynvml.nvmlSystemGetDriverVersion()), 128)
            self.available = True
        except Exception as exc:  # NVML commonly fails on CPU-only hosts.
            self.error = f"NVML unavailable: {exc}"
            self.available = False
            self.count = 0
            self._shutdown_safely()
            LOGGER.debug("NVML initialization failed", exc_info=True)

    def _shutdown_safely(self) -> None:
        if self._initialized and self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                LOGGER.debug("NVML shutdown failed", exc_info=True)
        self._initialized = False
        self._detail_cache.clear()

    def close(self) -> None:
        self._shutdown_safely()
        self.available = False

    def sample(self, detailed: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return self._unavailable_result()
        if not self.available or self.nvml is None:
            if time.monotonic() - self._last_init_attempt >= self.retry_seconds:
                self._initialize()
            if not self.available or self.nvml is None:
                return self._unavailable_result()

        try:
            self.count = int(self.nvml.nvmlDeviceGetCount())
        except Exception as exc:
            self.error = f"NVML device query failed: {exc}"
            LOGGER.debug("NVML device enumeration failed", exc_info=True)
            return self._unavailable_result()

        gpus: list[GpuInfo] = []
        raw_proc_map: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"gpu_mem_by_device": {}, "gpu_ids": set(), "kinds": set()}
        )
        errors: list[str] = []
        for index in range(self.count):
            try:
                info = self._sample_one(index, detailed=detailed)
            except Exception as exc:
                message = sanitize_text(str(exc), 160)
                errors.append(f"GPU {index}: {message}")
                info = GpuInfo(
                    index=index,
                    name=f"GPU {index}",
                    uuid="",
                    memory_used=0,
                    memory_total=0,
                    gpu_util=None,
                    mem_util=None,
                    temp_c=None,
                    power_w=None,
                    power_limit_w=None,
                    fan_pct=None,
                    perf_state=None,
                    sm_clock_mhz=None,
                    mem_clock_mhz=None,
                    pcie_rx_bps=None,
                    pcie_tx_bps=None,
                    encoder_util=None,
                    decoder_util=None,
                    processes=[],
                    error=message,
                )
                LOGGER.debug("GPU %s collection failed", index, exc_info=True)
            for proc in info.processes:
                entry = raw_proc_map[proc.pid]
                device = str(proc.gpu_index)
                prior = entry["gpu_mem_by_device"].get(device, 0)
                entry["gpu_mem_by_device"][device] = max(prior, max(0, proc.used_memory))
                entry["gpu_ids"].add(device)
                entry["kinds"].update(proc.kind.split("+"))
            gpus.append(info)

        proc_map = {
            pid: {
                "gpu_mem": sum(entry["gpu_mem_by_device"].values()),
                "gpu_ids": entry["gpu_ids"],
                "kinds": entry["kinds"],
            }
            for pid, entry in raw_proc_map.items()
        }
        error = "; ".join(errors[:3])
        if len(errors) > 3:
            error += f"; +{len(errors) - 3} more"
        return {
            "available": True,
            "degraded": bool(errors),
            "error": error,
            "driver": self.driver,
            "gpus": gpus,
            "proc_map": proc_map,
            "monitor_only": True,
        }

    def _unavailable_result(self) -> dict[str, Any]:
        return {
            "available": False,
            "degraded": False,
            "error": self.error,
            "driver": self.driver,
            "gpus": [],
            "proc_map": {},
            "monitor_only": True,
        }

    def _sample_one(self, index: int, detailed: bool = False) -> GpuInfo:
        nvml = self.nvml
        assert nvml is not None
        handle = nvml.nvmlDeviceGetHandleByIndex(index)

        name = sanitize_text(_decode(self._try(lambda: nvml.nvmlDeviceGetName(handle), f"GPU {index}")), 120)
        uuid = sanitize_text(_decode(self._try(lambda: nvml.nvmlDeviceGetUUID(handle), "")), 120)
        mem = self._try(lambda: nvml.nvmlDeviceGetMemoryInfo(handle), None)
        util = self._try(lambda: nvml.nvmlDeviceGetUtilizationRates(handle), None)
        temp = self._try(lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU), None)
        power = self._try(lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, None)
        power_limit = self._try(lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, None)
        details = self._detail_metrics(handle, index) if detailed else {}

        processes = self._gpu_processes(handle, index, uuid)
        return GpuInfo(
            index=index,
            name=name,
            uuid=uuid,
            memory_used=0 if mem is None else int(mem.used),
            memory_total=0 if mem is None else int(mem.total),
            gpu_util=None if util is None else float(util.gpu),
            mem_util=None if util is None else float(util.memory),
            temp_c=None if temp is None else float(temp),
            power_w=None if power is None else float(power),
            power_limit_w=None if power_limit is None else float(power_limit),
            fan_pct=details.get("fan_pct"),
            perf_state=details.get("perf_state"),
            sm_clock_mhz=details.get("sm_clock_mhz"),
            mem_clock_mhz=details.get("mem_clock_mhz"),
            pcie_rx_bps=details.get("pcie_rx_bps"),
            pcie_tx_bps=details.get("pcie_tx_bps"),
            encoder_util=details.get("encoder_util"),
            decoder_util=details.get("decoder_util"),
            processes=processes,
        )

    def _detail_metrics(self, handle: Any, index: int) -> dict[str, Any]:
        now_mono = time.monotonic()
        cached = self._detail_cache.get(index)
        if cached is not None and now_mono - cached[0] < self.detail_refresh_seconds:
            return cached[1]

        nvml = self.nvml
        assert nvml is not None
        fan = self._try(lambda: nvml.nvmlDeviceGetFanSpeed(handle), None)
        perf_state = self._try(lambda: f"P{nvml.nvmlDeviceGetPerformanceState(handle)}", None)
        sm_clock = self._try(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM), None)
        mem_clock = self._try(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM), None)
        encoder = self._try(lambda: nvml.nvmlDeviceGetEncoderUtilization(handle)[0], None)
        decoder = self._try(lambda: nvml.nvmlDeviceGetDecoderUtilization(handle)[0], None)
        rx = None
        tx = None
        if hasattr(nvml, "NVML_PCIE_UTIL_RX_BYTES") and hasattr(nvml, "nvmlDeviceGetPcieThroughput"):
            rx_kib = self._try(lambda: nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES), None)
            tx_kib = self._try(lambda: nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES), None)
            rx = None if rx_kib is None else float(rx_kib) * 1024.0
            tx = None if tx_kib is None else float(tx_kib) * 1024.0
        details = {
            "fan_pct": None if fan is None else float(fan),
            "perf_state": perf_state,
            "sm_clock_mhz": None if sm_clock is None else int(sm_clock),
            "mem_clock_mhz": None if mem_clock is None else int(mem_clock),
            "pcie_rx_bps": rx,
            "pcie_tx_bps": tx,
            "encoder_util": None if encoder is None else float(encoder),
            "decoder_util": None if decoder is None else float(decoder),
        }
        self._detail_cache[index] = (now_mono, details)
        return details

    def _gpu_processes(self, handle: Any, gpu_index: int, gpu_uuid: str) -> list[GpuProcess]:
        nvml = self.nvml
        assert nvml is not None
        aggregated: dict[int, dict[str, Any]] = {}
        query_plan = (
            (
                "C",
                (
                    "nvmlDeviceGetComputeRunningProcesses_v3",
                    "nvmlDeviceGetComputeRunningProcesses_v2",
                    "nvmlDeviceGetComputeRunningProcesses",
                ),
            ),
            (
                "G",
                (
                    "nvmlDeviceGetGraphicsRunningProcesses_v3",
                    "nvmlDeviceGetGraphicsRunningProcesses_v2",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ),
            ),
        )
        for kind, func_names in query_plan:
            raw_processes: list[Any] = []
            for func_name in func_names:
                func = getattr(nvml, func_name, None)
                if func is None:
                    continue
                try:
                    raw_processes = list(func(handle))
                    break
                except Exception:
                    continue

            for proc in raw_processes:
                pid = int(getattr(proc, "pid", 0))
                if pid <= 0:
                    continue
                used = getattr(proc, "usedGpuMemory", 0)
                if used is None or used == BYTES_NA or used > 1 << 60:
                    used = 0
                entry = aggregated.setdefault(pid, {"memory": 0, "kinds": set()})
                entry["memory"] = max(entry["memory"], int(used))
                entry["kinds"].add(kind)

        processes = []
        for pid, entry in sorted(aggregated.items()):
            kinds = "+".join(kind for kind in ("C", "G") if kind in entry["kinds"])
            processes.append(
                GpuProcess(
                    pid=pid,
                    gpu_index=gpu_index,
                    gpu_uuid=gpu_uuid,
                    kind=kinds,
                    used_memory=entry["memory"],
                )
            )
        return processes

    @staticmethod
    def _try(func: Any, default: Any) -> Any:
        try:
            return func()
        except Exception:
            return default


class Sampler:
    """Collect a coherent host snapshot and maintain rate/history state."""

    def __init__(
        self,
        gpu_enabled: bool = True,
        history_len: int = 90,
        gpu_retry_seconds: float = 30.0,
        collect_processes: bool = True,
    ) -> None:
        self.gpu = NvmlSampler(enabled=gpu_enabled, retry_seconds=gpu_retry_seconds)
        self.collect_processes = collect_processes
        self.prev_mono: float | None = None
        self.prev_disk: dict[str, Any] = {}
        self.prev_net: dict[str, Any] = {}
        self.prev_proc_cpu: dict[tuple[int, float], float] = {}
        self.prev_proc_io: dict[tuple[int, float], tuple[int, int]] = {}
        self.proc_cmd_cache: dict[tuple[int, float], tuple[str, float]] = {}
        self.uid_cache: dict[int, str] = {}
        self.cpu_history: deque[float] = deque(maxlen=history_len)
        self.mem_history: deque[float] = deque(maxlen=history_len)
        self.net_history: deque[float] = deque(maxlen=history_len)
        self.gpu_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=history_len))
        self.hostname = sanitize_text(socket.gethostname(), 255)
        self.boot_time = safe_call(psutil.boot_time, time.time())
        self.logical_cpus = psutil.cpu_count(logical=True) or 1
        self.memory_total = 1
        self.sequence = 0
        self._partition_cache: dict[bool, tuple[float, list[Any]]] = {}
        psutil.cpu_percent(interval=None, percpu=True)

    def close(self) -> None:
        self.gpu.close()

    def reset_rates(self) -> None:
        self.prev_mono = None
        self.prev_disk.clear()
        self.prev_net.clear()
        self.prev_proc_cpu.clear()
        self.prev_proc_io.clear()
        self.proc_cmd_cache.clear()
        self.cpu_history.clear()
        self.mem_history.clear()
        self.net_history.clear()
        self.gpu_history.clear()
        psutil.cpu_percent(interval=None, percpu=True)

    def sample(self, state: UIState) -> dict[str, Any]:
        started_mono = time.monotonic()
        now = time.time()
        elapsed = (
            max(0.001, started_mono - self.prev_mono) if self.prev_mono is not None else max(0.001, state.interval)
        )
        warnings: list[str] = []

        cpu_per = psutil.cpu_percent(interval=None, percpu=True)
        cpu_total = sum(cpu_per) / max(1, len(cpu_per))
        mem = psutil.virtual_memory()
        self.memory_total = max(1, int(mem.total))
        swap = psutil.swap_memory()
        cpu_temp = read_cpu_temperature()
        cpu_freq = safe_call(psutil.cpu_freq, None)
        cpu_stats = safe_call(psutil.cpu_stats, None)

        disks = self._collect_or_default(
            "disk",
            lambda: self._sample_disks(started_mono, elapsed, state),
            {"partitions": [], "io": []},
            warnings,
        )
        net = self._collect_or_default(
            "network",
            lambda: self._sample_net(elapsed, state),
            {"interfaces": []},
            warnings,
        )
        gpu = self._collect_or_default(
            "GPU",
            lambda: self.gpu.sample(detailed=state.show_full),
            self.gpu._unavailable_result(),
            warnings,
        )
        if self.collect_processes:
            processes, total_processes = self._collect_or_default(
                "process",
                lambda: self._sample_processes(started_mono, elapsed, gpu.get("proc_map", {}), state),
                ([], 0),
                warnings,
            )
        else:
            processes, total_processes = [], 0

        self.cpu_history.append(cpu_total)
        self.mem_history.append(float(mem.percent))
        total_net = sum(item["rx_bps"] + item["tx_bps"] for item in net["interfaces"])
        self.net_history.append(total_net)
        for gpu_info in gpu["gpus"]:
            if gpu_info.gpu_util is not None:
                self.gpu_history[gpu_info.index].append(float(gpu_info.gpu_util))

        self.prev_mono = started_mono
        self.sequence += 1
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "version": __version__,
            "sequence": self.sequence,
            "time": now,
            "elapsed": elapsed,
            "hostname": self.hostname,
            "uptime": max(0.0, now - self.boot_time),
            "logical_cpus": self.logical_cpus,
            "cpu": {
                "total": cpu_total,
                "percpu": cpu_per,
                "load": os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0),
                "temp": cpu_temp,
                "freq": cpu_freq,
                "stats": cpu_stats,
                "history": list(self.cpu_history),
            },
            "memory": {"virtual": mem, "swap": swap, "history": list(self.mem_history)},
            "disks": disks,
            "network": {**net, "history": list(self.net_history)},
            "gpu": {**gpu, "history": {idx: list(values) for idx, values in self.gpu_history.items()}},
            "processes": processes,
            "processes_total": total_processes,
            "warnings": warnings,
        }
        duration_ms = (time.monotonic() - started_mono) * 1000.0
        snapshot["sample_duration_ms"] = duration_ms
        if duration_ms > state.interval * 1000.0:
            warnings.append(
                f"collection took {duration_ms:.0f}ms, longer than the {state.interval * 1000.0:.0f}ms interval"
            )
        return snapshot

    @staticmethod
    def _collect_or_default(
        name: str,
        collector: Any,
        default: Any,
        warnings: list[str],
    ) -> Any:
        try:
            return collector()
        except Exception as exc:
            message = f"{name} collection failed: {sanitize_text(str(exc), 180)}"
            warnings.append(message)
            LOGGER.warning(message, exc_info=True)
            return default

    def _sample_disks(self, now_mono: float, elapsed: float, state: UIState) -> dict[str, Any]:
        partitions = []
        cached = self._partition_cache.get(state.show_all_mounts)
        if cached is None or now_mono - cached[0] >= 10.0:
            raw_partitions = list(psutil.disk_partitions(all=state.show_all_mounts))
            self._partition_cache[state.show_all_mounts] = (now_mono, raw_partitions)
        else:
            raw_partitions = cached[1]
        seen_mounts: set[str] = set()
        for part in raw_partitions:
            if not state.show_all_mounts and (
                not part.fstype or part.fstype in {"autofs", "tmpfs", "devtmpfs", "squashfs"}
            ):
                continue
            if part.mountpoint in seen_mounts:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except (PermissionError, FileNotFoundError, OSError):
                continue
            seen_mounts.add(part.mountpoint)
            partitions.append(
                {
                    "device": sanitize_text(part.device, 4096),
                    "mountpoint": sanitize_text(part.mountpoint, 4096),
                    "fstype": sanitize_text(part.fstype, 128),
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            )

        io_rows = []
        current = psutil.disk_io_counters(perdisk=True) or {}
        for name, counters in current.items():
            prev = self.prev_disk.get(name)
            if prev is None:
                read_bps = write_bps = read_iops = write_iops = busy_pct = 0.0
            else:
                read_bps = rate(counters.read_bytes, prev.read_bytes, elapsed)
                write_bps = rate(counters.write_bytes, prev.write_bytes, elapsed)
                read_iops = rate(counters.read_count, prev.read_count, elapsed)
                write_iops = rate(counters.write_count, prev.write_count, elapsed)
                busy_delta = getattr(counters, "busy_time", 0) - getattr(prev, "busy_time", 0)
                busy_pct = clamp((busy_delta / (elapsed * 1000.0)) * 100.0, 0.0, 100.0)
            io_rows.append(
                {
                    "name": sanitize_text(name, 256),
                    "read_bps": read_bps,
                    "write_bps": write_bps,
                    "read_iops": read_iops,
                    "write_iops": write_iops,
                    "busy_pct": busy_pct,
                }
            )
        self.prev_disk = dict(current)
        io_rows.sort(key=lambda item: item["read_bps"] + item["write_bps"], reverse=True)
        partitions.sort(key=lambda item: item["percent"], reverse=True)
        return {"partitions": partitions, "io": io_rows}

    def _sample_net(self, elapsed: float, state: UIState) -> dict[str, Any]:
        stats = safe_call(psutil.net_if_stats, {})
        current = psutil.net_io_counters(pernic=True) or {}
        rows = []
        for name, counters in current.items():
            prev = self.prev_net.get(name)
            if prev is None:
                rx_bps = tx_bps = rx_pps = tx_pps = 0.0
            else:
                rx_bps = rate(counters.bytes_recv, prev.bytes_recv, elapsed)
                tx_bps = rate(counters.bytes_sent, prev.bytes_sent, elapsed)
                rx_pps = rate(counters.packets_recv, prev.packets_recv, elapsed)
                tx_pps = rate(counters.packets_sent, prev.packets_sent, elapsed)
            nic_stats = stats.get(name)
            is_up = bool(getattr(nic_stats, "isup", True))
            speed_mbps = getattr(nic_stats, "speed", 0) if nic_stats is not None else 0
            if not state.show_all_ifaces and _is_quiet_interface(name, is_up, rx_bps, tx_bps):
                continue
            rows.append(
                {
                    "name": sanitize_text(name, 256),
                    "is_up": is_up,
                    "speed_mbps": speed_mbps,
                    "rx_bps": rx_bps,
                    "tx_bps": tx_bps,
                    "rx_pps": rx_pps,
                    "tx_pps": tx_pps,
                    "errin": counters.errin,
                    "errout": counters.errout,
                    "dropin": counters.dropin,
                    "dropout": counters.dropout,
                }
            )
        self.prev_net = dict(current)
        rows.sort(key=lambda item: item["rx_bps"] + item["tx_bps"], reverse=True)
        return {"interfaces": rows}

    def _sample_processes(
        self,
        now_mono: float,
        elapsed: float,
        gpu_proc_map: dict[int, dict[str, Any]],
        state: UIState,
    ) -> tuple[list[ProcessRow], int]:
        rows: list[ProcessRow] = []
        seen_identities: set[tuple[int, float]] = set()
        total_processes = 0
        filter_text = state.filter_text.casefold().strip()
        user_filter = state.user_filter.casefold().strip()
        collect_io = state.sort_key == "io"
        io_sampled: set[tuple[int, float]] = set()
        if state.gpu_only:
            total_processes = len(psutil.pids())
            candidates = self._process_candidates_for_pids(gpu_proc_map)
        else:
            candidates = ((proc, proc.info) for proc in psutil.process_iter(["pid", "uids", "name", "status"]))

        for proc, info in candidates:
            pid = info.get("pid")
            if not isinstance(pid, int):
                continue
            if not state.gpu_only:
                total_processes += 1
            gpu_entry = gpu_proc_map.get(pid, {})
            gpu_ids = ",".join(
                sorted(
                    gpu_entry.get("gpu_ids", set()),
                    key=lambda value: int(value) if value.isdigit() else value,
                )
            )
            if state.gpu_only and not gpu_ids:
                continue
            user = self._username_from_uids(info.get("uids"))
            normalized_user = sanitize_text(str(user).split("\\")[-1], 128)
            if user_filter and normalized_user.casefold() != user_filter:
                continue
            try:
                with proc.oneshot():
                    cpu_times = proc.cpu_times()
                    start_token = process_start_token(proc, self.boot_time)
                    identity = (pid, start_token)
                    seen_identities.add(identity)
                    proc_cpu_total = float(cpu_times.user + cpu_times.system)
                    prev_cpu_total = self.prev_proc_cpu.get(identity)
                    cpu_pct = (
                        0.0 if prev_cpu_total is None else max(0.0, (proc_cpu_total - prev_cpu_total) / elapsed * 100.0)
                    )
                    self.prev_proc_cpu[identity] = proc_cpu_total

                    mem_info = proc.memory_info()
                    mem_pct = float(mem_info.rss) / self.memory_total * 100.0
                    if collect_io:
                        try:
                            io_counters = proc.io_counters()
                            prev_io = self.prev_proc_io.get(identity)
                            if prev_io is None:
                                read_bps = write_bps = 0.0
                            else:
                                read_bps = rate(io_counters.read_bytes, prev_io[0], elapsed)
                                write_bps = rate(io_counters.write_bytes, prev_io[1], elapsed)
                            self.prev_proc_io[identity] = (io_counters.read_bytes, io_counters.write_bytes)
                            io_sampled.add(identity)
                        except (psutil.AccessDenied, AttributeError, NotImplementedError):
                            read_bps = write_bps = 0.0
                    else:
                        read_bps = write_bps = 0.0

                    name = sanitize_text(info.get("name") or "?", 256)
                    status = sanitize_text(info.get("status") or "?", 64)
                    threads = 0
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue

            gpu_mem = int(gpu_entry.get("gpu_mem", 0))
            gpu_kinds = "+".join(kind for kind in ("C", "G") if kind in gpu_entry.get("kinds", set()))

            cached_command = self.proc_cmd_cache.get(identity, ("", 0.0))[0]
            command = cached_command or name
            row = ProcessRow(
                pid=pid,
                user=normalized_user,
                name=name,
                status=status,
                cpu_pct=cpu_pct,
                mem_pct=mem_pct,
                rss=int(mem_info.rss),
                vms=int(mem_info.vms),
                threads=threads,
                read_bps=read_bps,
                write_bps=write_bps,
                gpu_mem=gpu_mem,
                gpu_ids=gpu_ids,
                gpu_kinds=gpu_kinds,
                command=command,
                create_time=self.boot_time + start_token,
                start_token=start_token,
            )
            if filter_text:
                basic_haystack = f"{row.pid} {row.user} {row.name} {row.status} {row.gpu_ids}".casefold()
                if filter_text not in basic_haystack:
                    row.command = self._resolve_command(identity, name, now_mono, ttl=60.0)
                    if not _process_matches(row, filter_text):
                        continue
            rows.append(row)

        stale = set(self.prev_proc_cpu) - seen_identities
        for identity in stale:
            self.prev_proc_cpu.pop(identity, None)
            self.prev_proc_io.pop(identity, None)
            self.proc_cmd_cache.pop(identity, None)

        rows.sort(key=process_sort_key(state.sort_key), reverse=state.reverse)
        warm_count = min(len(rows), max(40, state.scroll + state.page_size + 10, state.selected + 10))
        for row in rows[:warm_count]:
            row.command = self._resolve_command(row.identity, row.name, now_mono, ttl=30.0)
            if state.show_full:
                row.threads = self._resolve_threads(row.pid)
                if not collect_io:
                    row.read_bps, row.write_bps = self._resolve_process_io(row, elapsed)
                    io_sampled.add(row.identity)
        for identity in set(self.prev_proc_io) - io_sampled:
            self.prev_proc_io.pop(identity, None)
        return rows, total_processes

    @staticmethod
    def _process_candidates_for_pids(
        gpu_proc_map: dict[int, dict[str, Any]],
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        for pid in sorted(gpu_proc_map):
            try:
                proc = psutil.Process(pid)
                with proc.oneshot():
                    info = {
                        "pid": pid,
                        "uids": proc.uids(),
                        "name": proc.name(),
                        "status": proc.status(),
                    }
                yield proc, info
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue

    def _username_from_uids(self, uids: Any) -> str:
        uid = getattr(uids, "real", None)
        if uid is None:
            return "?"
        uid = int(uid)
        cached = self.uid_cache.get(uid)
        if cached is not None:
            return cached
        try:
            username = pwd.getpwuid(uid).pw_name
        except KeyError:
            username = str(uid)
        self.uid_cache[uid] = username
        return username

    def _resolve_command(self, identity: tuple[int, float], fallback: str, now_mono: float, ttl: float) -> str:
        cached = self.proc_cmd_cache.get(identity)
        if cached is not None and now_mono - cached[1] < ttl:
            return cached[0] or fallback
        pid, expected_start_token = identity
        try:
            proc = psutil.Process(pid)
            if expected_start_token and not same_process_start(
                process_start_token(proc, self.boot_time), expected_start_token
            ):
                return fallback
            cmd = sanitize_text(" ".join(proc.cmdline()) or fallback, 8192)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            cmd = fallback
        self.proc_cmd_cache[identity] = (cmd, now_mono)
        return cmd

    def _resolve_process_io(self, row: ProcessRow, elapsed: float) -> tuple[float, float]:
        try:
            proc = psutil.Process(row.pid)
            with proc.oneshot():
                if row.start_token and not same_process_start(
                    process_start_token(proc, self.boot_time), row.start_token
                ):
                    return 0.0, 0.0
                counters = proc.io_counters()
            previous = self.prev_proc_io.get(row.identity)
            self.prev_proc_io[row.identity] = (counters.read_bytes, counters.write_bytes)
            if previous is None:
                return 0.0, 0.0
            return (
                rate(counters.read_bytes, previous[0], elapsed),
                rate(counters.write_bytes, previous[1], elapsed),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError, NotImplementedError, ProcessLookupError):
            return 0.0, 0.0

    @staticmethod
    def _resolve_threads(pid: int) -> int:
        try:
            return psutil.Process(pid).num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            return 0


def process_sort_key(sort_key: str) -> Any:
    if sort_key == "mem":
        return lambda row: (row.mem_pct, row.rss, row.cpu_pct)
    if sort_key == "gpu":
        return lambda row: (row.gpu_mem, row.cpu_pct, row.rss)
    if sort_key == "io":
        return lambda row: (row.io_bps, row.cpu_pct, row.rss)
    if sort_key == "pid":
        return lambda row: row.pid
    if sort_key == "name":
        return lambda row: row.name.casefold()
    return lambda row: (row.cpu_pct, row.mem_pct, row.rss)


def render_dashboard(snapshot: dict[str, Any], state: UIState, console: Console) -> Group:
    console_height = console.size.height
    narrow = console.size.width < 80
    compact = console.size.width < 120
    fixed_height = (38 if compact else 22) if state.show_full else (24 if compact else 14)
    if state.show_details:
        fixed_height += 7
    if state.show_help:
        fixed_height += 9
    if narrow:
        fixed_height += 1
    state.page_size = state.forced_rows or max(3 if narrow else 6, console_height - fixed_height)
    sync_scroll(state, len(snapshot["processes"]))

    top_grid = Table.grid(expand=True)
    if state.show_full and compact:
        top_grid.add_column(ratio=1)
        top_grid.add_row(render_system_panel(snapshot, state, compact))
        top_grid.add_row(render_gpu_panel(snapshot, compact))
        top_grid.add_row(render_disk_panel(snapshot, state, compact))
        top_grid.add_row(render_network_panel(snapshot, state, compact))
    elif state.show_full:
        top_grid.add_column(ratio=1)
        top_grid.add_column(ratio=1)
        left = Group(render_system_panel(snapshot, state, compact), render_disk_panel(snapshot, state, compact))
        right = Group(render_gpu_panel(snapshot, compact), render_network_panel(snapshot, state, compact))
        top_grid.add_row(left, right)
    elif compact:
        top_grid.add_column(ratio=1)
        top_grid.add_row(render_overview_panel(snapshot, state, compact))
        top_grid.add_row(render_gpu_summary_panel(snapshot, compact))
    else:
        top_grid.add_column(ratio=1)
        top_grid.add_column(ratio=1)
        top_grid.add_row(render_overview_panel(snapshot, state, compact), render_gpu_summary_panel(snapshot, compact))

    blocks: list[Any] = [
        render_header(snapshot, state),
        top_grid,
        render_process_panel(snapshot, state, compact, state.show_full, narrow),
    ]
    if state.show_details:
        blocks.append(render_detail_panel(snapshot, state))
    if state.show_help:
        blocks.append(render_help_panel())
    blocks.append(render_footer(state))
    return Group(*blocks)


def literal_text(value: Any) -> Text:
    return Text(sanitize_text(value))


def render_header(snapshot: dict[str, Any], state: UIState) -> Panel:
    now = dt.datetime.fromtimestamp(snapshot["time"]).strftime("%Y-%m-%d %H:%M:%S")
    if not state.interactive:
        live_state = "SNAPSHOT"
        mode_style = "cyan"
    elif state.paused:
        live_state = "PAUSED"
        mode_style = "yellow"
    else:
        live_state = "LIVE"
        mode_style = "green"
    view = "detail" if state.show_full else "summary"
    msg = state.visible_message()
    filter_part = f" filter={state.filter_text!r}" if state.filter_text else ""
    parts = Text()
    parts.append(f"OmniTop {__version__} ", style="bold cyan")
    parts.append(f"{snapshot['hostname']} ", style="bold")
    parts.append(f"{now}  ")
    parts.append(live_state, style=f"bold {mode_style}")
    parts.append(f"  {view}  {state.interval:.1f}s  sort={state.sort_key}{filter_part}")
    if snapshot["gpu"]["available"] and snapshot["gpu"].get("driver"):
        parts.append(f"  NVIDIA={snapshot['gpu']['driver']}", style="dim")
    if snapshot.get("warnings"):
        parts.append(f"  DEGRADED({len(snapshot['warnings'])})", style="bold yellow")
    elif snapshot["gpu"].get("degraded"):
        parts.append("  GPU DEGRADED", style="bold yellow")
    duration = snapshot.get("sample_duration_ms")
    if isinstance(duration, (int, float)):
        parts.append(f"  sample={duration:.0f}ms", style="dim")
    if msg:
        parts.append(f"  {msg}", style="bold yellow")
    return Panel(parts, box=box.SIMPLE, padding=(0, 1))


def render_overview_panel(snapshot: dict[str, Any], state: UIState, compact: bool) -> Panel:
    cpu = snapshot["cpu"]
    mem = snapshot["memory"]["virtual"]
    swap = snapshot["memory"]["swap"]
    disks = snapshot["disks"]
    network = snapshot["network"]

    worst_disk = max(disks["partitions"], key=lambda item: item["percent"], default=None)
    disk_read = sum(item["read_bps"] for item in disks["io"])
    disk_write = sum(item["write_bps"] for item in disks["io"])
    disk_busy = max((item["busy_pct"] for item in disks["io"]), default=0.0)
    net_rx = sum(item["rx_bps"] for item in network["interfaces"])
    net_tx = sum(item["tx_bps"] for item in network["interfaces"])
    load1, load5, _load15 = cpu["load"]
    logical_cpus = snapshot.get("logical_cpus") or 1
    load_pct = load1 / logical_cpus * 100.0
    temp = cpu["temp"]

    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("Signal", no_wrap=True)
    table.add_column("Now", justify="right", no_wrap=True)
    table.add_column("Trend", overflow="ellipsis")
    table.add_column("State", justify="right", no_wrap=True)

    cpu_state = level_for_pct(cpu["total"], warn=70.0, crit=90.0)
    table.add_row(
        "CPU",
        meter(cpu["total"], width=10 if compact else 12),
        Text(spark(cpu["history"], unit_pct=True), style=style_for_pct(cpu["total"])),
        state_text(cpu_state),
    )

    mem_state = level_for_pct(mem.percent, warn=75.0, crit=90.0)
    table.add_row(
        "Memory",
        f"{fmt_pct(mem.percent)}  {fmt_bytes(mem.used)}/{fmt_bytes(mem.total)}",
        Text(spark(snapshot["memory"]["history"], unit_pct=True), style=style_for_pct(mem.percent)),
        state_text(mem_state),
    )

    if worst_disk is not None:
        disk_state = worst_of(
            level_for_pct(worst_disk["percent"], warn=80.0, crit=92.0), level_for_pct(disk_busy, warn=70.0, crit=90.0)
        )
        disk_now = (
            f"{pct_text_plain(worst_disk['percent'])}  {shorten_mount(worst_disk['mountpoint'], 12 if compact else 18)}"
        )
    else:
        disk_state = "unknown"
        disk_now = "n/a"
    disk_trend = f"R {fmt_bps(disk_read)}  W {fmt_bps(disk_write)}"
    table.add_row("Disk", disk_now, disk_trend, state_text(disk_state))

    net_now = f"RX {fmt_bps(net_rx)}  TX {fmt_bps(net_tx)}"
    table.add_row("Network", net_now, Text(spark(network["history"], unit_pct=False), style="cyan"), state_text("ok"))

    load_state = level_for_pct(load_pct, warn=70.0, crit=100.0)
    temp_state = "ok" if temp is None else level_for_temp(temp)
    table.add_row(
        "Load/Temp",
        f"{load1:.1f}/{load5:.1f} on {logical_cpus} CPU",
        "n/a" if temp is None else f"{temp:.0f} C",
        state_text(worst_of(load_state, temp_state)),
    )

    if swap.total and swap.percent > 1.0:
        table.add_row(
            "Swap",
            f"{fmt_pct(swap.percent)}  {fmt_bytes(swap.used)}/{fmt_bytes(swap.total)}",
            "",
            state_text(level_for_pct(swap.percent, warn=50.0, crit=80.0)),
        )

    subtitle = "golden signals: saturation, throughput, pressure"
    if state.show_all_ifaces or state.show_all_mounts:
        subtitle += " | all devices"
    return Panel(table, title=literal_text(f"Overview | {subtitle}"), box=box.ROUNDED, padding=(0, 1))


def render_gpu_summary_panel(snapshot: dict[str, Any], compact: bool) -> Panel:
    gpu = snapshot["gpu"]
    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("GPU", justify="right", no_wrap=True)
    table.add_column("Util", justify="right", no_wrap=True)
    table.add_column("Memory", justify="right", no_wrap=True)
    table.add_column("Temp", justify="right", no_wrap=True)
    table.add_column("Power", justify="right", no_wrap=True)
    table.add_column("Proc", justify="right", no_wrap=True)
    table.add_column("State", justify="right", no_wrap=True)

    if not gpu["available"]:
        message = gpu["error"] or "No NVIDIA GPU detected"
        table.add_row("-", message, "", "", "", "", state_text("unknown"))
        return Panel(table, title=literal_text("GPU Fleet"), box=box.ROUNDED, padding=(0, 1))

    gpus: list[GpuInfo] = gpu["gpus"]
    for info in gpus:
        if info.error:
            table.add_row(
                str(info.index),
                Text("error", style="bold red"),
                "n/a",
                "n/a",
                "n/a",
                "0",
                state_text("error"),
            )
            continue
        mem_pct = (info.memory_used / info.memory_total * 100.0) if info.memory_total else 0.0
        temp_state = "ok" if info.temp_c is None else level_for_temp(info.temp_c)
        mem_state = level_for_pct(mem_pct, warn=85.0, crit=95.0)
        power_state = "ok"
        if info.power_w is not None and info.power_limit_w:
            power_state = level_for_pct(info.power_w / info.power_limit_w * 100.0, warn=90.0, crit=105.0)
        gpu_state = worst_of(mem_state, temp_state, power_state)
        table.add_row(
            str(info.index),
            meter(info.gpu_util, width=8 if compact else 10),
            f"{fmt_bytes(info.memory_used)}/{fmt_bytes(info.memory_total)} {mem_pct:.0f}%",
            "n/a" if info.temp_c is None else f"{info.temp_c:.0f} C",
            "n/a" if info.power_w is None else f"{info.power_w:.0f}W",
            str(len(info.processes)),
            state_text(gpu_state),
        )

    utils = [info.gpu_util for info in gpus if info.gpu_util is not None]
    avg_util = sum(utils) / len(utils) if utils else 0.0
    max_mem = (
        max(((info.memory_used / info.memory_total * 100.0) if info.memory_total else 0.0) for info in gpus)
        if gpus
        else 0.0
    )
    error_count = sum(bool(info.error) for info in gpus)
    title = f"GPU Fleet | {len(gpus)} GPUs | avg {avg_util:.0f}% | max mem {max_mem:.0f}%"
    if error_count:
        title += f" | {error_count} errors"
    return Panel(table, title=literal_text(title), box=box.ROUNDED, padding=(0, 1))


def render_system_panel(snapshot: dict[str, Any], state: UIState, compact: bool) -> Panel:
    cpu = snapshot["cpu"]
    mem = snapshot["memory"]["virtual"]
    swap = snapshot["memory"]["swap"]

    bar_width = 14 if compact else 20
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(justify="right", no_wrap=True)
    table.add_column(ratio=1)
    table.add_column(justify="right", no_wrap=True)

    table.add_row("CPU", bar(cpu["total"], width=bar_width), fmt_pct(cpu["total"]))
    load = cpu["load"]
    freq = cpu["freq"]
    freq_text = (
        "n/a" if freq is None else f"{freq.current:.0f}/{freq.max:.0f} MHz" if freq.max else f"{freq.current:.0f} MHz"
    )
    temp_text = "n/a" if cpu["temp"] is None else f"{cpu['temp']:.0f} C"
    table.add_row(
        "Load",
        Text(f"{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}"),
        f"{snapshot.get('logical_cpus') or 1} CPUs",
    )
    table.add_row("Freq/Temp", Text(freq_text), temp_text)
    table.add_row("CPU hist", Text(spark(cpu["history"], unit_pct=True), style=style_for_pct(cpu["total"])), "")

    table.add_row("Mem", bar(mem.percent, width=bar_width), f"{fmt_bytes(mem.used)} / {fmt_bytes(mem.total)}")
    table.add_row("Swap", bar(swap.percent, width=bar_width), f"{fmt_bytes(swap.used)} / {fmt_bytes(swap.total)}")
    table.add_row(
        "Mem hist", Text(spark(snapshot["memory"]["history"], unit_pct=True), style=style_for_pct(mem.percent)), ""
    )

    if state.show_per_core:
        per_core = cpu["percpu"]
        core_lines = []
        max_cores = 32
        for idx, pct in enumerate(per_core[:max_cores]):
            core_lines.append(f"C{idx:02d} {pct:5.1f}%")
        suffix = "" if len(per_core) <= max_cores else f" ... +{len(per_core) - max_cores}"
        table.add_row("Cores", Text("  ".join(core_lines) + suffix), "")
    else:
        per_core = cpu["percpu"]
        if per_core:
            busiest = sorted(enumerate(per_core), key=lambda item: item[1], reverse=True)[:6]
            table.add_row("Hot cores", Text("  ".join(f"C{idx}:{pct:.0f}%" for idx, pct in busiest)), "")

    uptime = fmt_duration(snapshot["uptime"])
    stats = cpu["stats"]
    stats_text = ""
    if stats is not None:
        stats_text = f"ctx={fmt_int(stats.ctx_switches)} intr={fmt_int(stats.interrupts)}"
    table.add_row("Uptime", Text(uptime), stats_text)
    return Panel(table, title=literal_text("CPU / Memory"), box=box.ROUNDED, padding=(0, 1))


def render_disk_panel(snapshot: dict[str, Any], state: UIState, compact: bool) -> Panel:
    disks = snapshot["disks"]
    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("Mount", overflow="ellipsis")
    table.add_column("Use", justify="right", no_wrap=True)
    table.add_column("%", justify="right", no_wrap=True)
    table.add_column("Device", overflow="ellipsis")
    for part in disks["partitions"][: (4 if compact else 5)]:
        table.add_row(
            shorten_mount(part["mountpoint"], 22),
            f"{fmt_bytes(part['used'])}/{fmt_bytes(part['total'])}",
            pct_text(part["percent"]),
            part["device"],
        )
    if not disks["partitions"]:
        table.add_row("n/a", "", "", "")

    io_table = Table(box=None, expand=True, pad_edge=False)
    io_table.add_column("Disk")
    io_table.add_column("Read", justify="right")
    io_table.add_column("Write", justify="right")
    io_table.add_column("IOPS", justify="right")
    io_table.add_column("Busy", justify="right")
    for item in disks["io"][: (4 if compact else 5)]:
        iops = item["read_iops"] + item["write_iops"]
        io_table.add_row(
            item["name"],
            fmt_bps(item["read_bps"]),
            fmt_bps(item["write_bps"]),
            f"{iops:.0f}",
            pct_text(item["busy_pct"]),
        )

    group = Group(table, Text(""), io_table)
    title = "Disk / Filesystem" + (" (all)" if state.show_all_mounts else "")
    return Panel(group, title=literal_text(title), box=box.ROUNDED, padding=(0, 1))


def render_gpu_panel(snapshot: dict[str, Any], compact: bool) -> Panel:
    gpu = snapshot["gpu"]
    table = Table(box=None, expand=True, pad_edge=False)
    if compact:
        table.add_column("GPU", justify="right", no_wrap=True)
        table.add_column("Name", overflow="ellipsis")
        table.add_column("Util", justify="right", no_wrap=True)
        table.add_column("Mem", justify="right", no_wrap=True)
        table.add_column("Temp", justify="right", no_wrap=True)
        table.add_column("Power", justify="right", no_wrap=True)
        table.add_column("Proc", justify="right", no_wrap=True)
    else:
        table.add_column("GPU", justify="right", no_wrap=True)
        table.add_column("Name", overflow="ellipsis")
        table.add_column("Util", justify="right", no_wrap=True)
        table.add_column("Mem", justify="right", no_wrap=True)
        table.add_column("Temp", justify="right", no_wrap=True)
        table.add_column("Power", justify="right", no_wrap=True)
        table.add_column("Fan", justify="right", no_wrap=True)
        table.add_column("Clock", justify="right", no_wrap=True)
        table.add_column("PCIe", justify="right", no_wrap=True)

    if not gpu["available"]:
        message = gpu["error"] or "No NVIDIA GPU detected"
        table.add_row("-", message, "", "", "", "", "") if compact else table.add_row(
            "-", message, "", "", "", "", "", "", ""
        )
        return Panel(table, title=literal_text("NVIDIA GPU"), box=box.ROUNDED, padding=(0, 1))

    for info in gpu["gpus"]:
        if info.error:
            error_name = f"{info.name}: {info.error}"
            if compact:
                table.add_row(str(info.index), error_name, "error", "n/a", "n/a", "n/a", "0")
            else:
                table.add_row(str(info.index), error_name, "error", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a")
            continue
        mem_pct = (info.memory_used / info.memory_total * 100.0) if info.memory_total else 0.0
        power_text = (
            "n/a"
            if info.power_w is None
            else (f"{info.power_w:.0f}W/{info.power_limit_w:.0f}W" if info.power_limit_w else f"{info.power_w:.0f}W")
        )
        clock_text = "n/a" if info.sm_clock_mhz is None else f"{info.sm_clock_mhz}/{info.mem_clock_mhz or 0}"
        pcie_text = "n/a"
        if info.pcie_rx_bps is not None or info.pcie_tx_bps is not None:
            pcie_text = f"{fmt_bps(info.pcie_rx_bps or 0)}/{fmt_bps(info.pcie_tx_bps or 0)}"
        util_text = "n/a" if info.gpu_util is None else pct_text(info.gpu_util)
        if compact:
            table.add_row(
                str(info.index),
                info.name,
                util_text,
                f"{fmt_bytes(info.memory_used)}/{fmt_bytes(info.memory_total)} {mem_pct:.0f}%",
                "n/a" if info.temp_c is None else f"{info.temp_c:.0f} C",
                power_text,
                str(len(info.processes)),
            )
        else:
            table.add_row(
                str(info.index),
                info.name,
                util_text,
                f"{fmt_bytes(info.memory_used)}/{fmt_bytes(info.memory_total)} {mem_pct:.0f}%",
                "n/a" if info.temp_c is None else f"{info.temp_c:.0f} C",
                power_text,
                "n/a" if info.fan_pct is None else f"{info.fan_pct:.0f}%",
                clock_text,
                pcie_text,
            )
            history = gpu["history"].get(info.index, [])
            if history:
                table.add_row(
                    "",
                    Text("GPU hist", style="dim"),
                    Text(spark(history, unit_pct=True), style=style_for_pct(info.gpu_util or 0)),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                )

    return Panel(table, title=literal_text("NVIDIA GPU"), box=box.ROUNDED, padding=(0, 1))


def render_network_panel(snapshot: dict[str, Any], state: UIState, compact: bool) -> Panel:
    network = snapshot["network"]
    table = Table(box=None, expand=True, pad_edge=False)
    if compact:
        table.add_column("Iface", overflow="ellipsis")
        table.add_column("State", justify="center", no_wrap=True)
        table.add_column("RX", justify="right", no_wrap=True)
        table.add_column("TX", justify="right", no_wrap=True)
        table.add_column("Err/Drop", justify="right", no_wrap=True)
    else:
        table.add_column("Iface", overflow="ellipsis")
        table.add_column("State", justify="center", no_wrap=True)
        table.add_column("RX", justify="right", no_wrap=True)
        table.add_column("TX", justify="right", no_wrap=True)
        table.add_column("Packets", justify="right", no_wrap=True)
        table.add_column("Err/Drop", justify="right", no_wrap=True)

    for item in network["interfaces"][: (6 if compact else 8)]:
        errors = item["errin"] + item["errout"]
        drops = item["dropin"] + item["dropout"]
        state_text = "up" if item["is_up"] else "down"
        if item["speed_mbps"]:
            state_text += f"/{item['speed_mbps']}M"
        if compact:
            table.add_row(
                item["name"], state_text, fmt_bps(item["rx_bps"]), fmt_bps(item["tx_bps"]), f"{errors}/{drops}"
            )
        else:
            table.add_row(
                item["name"],
                state_text,
                fmt_bps(item["rx_bps"]),
                fmt_bps(item["tx_bps"]),
                f"{item['rx_pps']:.0f}/{item['tx_pps']:.0f} p/s",
                f"{errors}/{drops}",
            )
    if not network["interfaces"]:
        table.add_row("n/a", "", "", "", "") if compact else table.add_row("n/a", "", "", "", "", "")

    total = network["history"][-1] if network["history"] else 0
    group = Group(
        table,
        Text(
            f"Total bandwidth history {spark(network['history'], unit_pct=False)}  now={fmt_bps(total)}", style="cyan"
        ),
    )
    title = "Network" + (" (all)" if state.show_all_ifaces else "")
    return Panel(group, title=literal_text(title), box=box.ROUNDED, padding=(0, 1))


def render_process_panel(
    snapshot: dict[str, Any],
    state: UIState,
    compact: bool,
    detailed: bool,
    narrow: bool = False,
) -> Panel:
    rows: list[ProcessRow] = snapshot["processes"]
    sync_scroll(state, len(rows))
    if rows:
        state.selected_pid = rows[state.selected].pid
        state.selected_start_token = rows[state.selected].start_token
    else:
        state.selected_pid = None
        state.selected_start_token = None
    visible = rows[state.scroll : state.scroll + state.page_size]

    table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("PID", justify="right", no_wrap=True)
    if not narrow:
        table.add_column("USER", no_wrap=True, overflow="ellipsis", max_width=14 if not compact else 10)
    table.add_column("CPU", justify="right", no_wrap=True)
    if not narrow:
        table.add_column("MEM", justify="right", no_wrap=True)
    if detailed and not narrow:
        table.add_column("RSS", justify="right", no_wrap=True)
    if detailed and not compact:
        table.add_column("IO R/W", justify="right", no_wrap=True)
    table.add_column("GPU", justify="right", no_wrap=True)
    if not narrow:
        table.add_column("DEV", justify="right", no_wrap=True)
        if detailed:
            table.add_column("TYPE", justify="center", no_wrap=True)
    if detailed and not compact:
        table.add_column("THR", justify="right", no_wrap=True)
        table.add_column("S", justify="center", no_wrap=True)
    table.add_column("COMMAND", overflow="ellipsis", ratio=1, no_wrap=True)

    for offset, row in enumerate(visible):
        absolute_index = state.scroll + offset
        selected = absolute_index == state.selected
        marker = ">" if selected else " "
        style = "reverse cyan" if selected else ""
        cells: list[Any] = [marker, str(row.pid)]
        if not narrow:
            cells.append(row.user)
        cells.append(Text(f"{row.cpu_pct:5.1f}", style=style_for_pct(min(row.cpu_pct, 100.0))))
        if not narrow:
            cells.append(Text(f"{row.mem_pct:5.1f}", style=style_for_pct(row.mem_pct)))
        if detailed and not narrow:
            cells.append(fmt_bytes(row.rss))
        if detailed and not compact:
            cells.append(f"{fmt_bps(row.read_bps)}/{fmt_bps(row.write_bps)}")
        cells.append(fmt_bytes(row.gpu_mem) if row.gpu_mem else "-")
        if not narrow:
            cells.append(row.gpu_ids or "-")
            if detailed:
                cells.append(row.gpu_kinds or "-")
        if detailed and not compact:
            cells.extend([str(row.threads), row.status[:1]])
        cells.append(row.command or row.name)
        table.add_row(*cells, style=style)

    total = snapshot.get("processes_total", len(rows))
    title = f"Top Processes | {len(rows)}/{total} shown"
    if state.filter_text:
        title += f" | filter: {state.filter_text}"
    if state.user_filter:
        title += f" | user: {state.user_filter}"
    if state.gpu_only:
        title += " | GPU only"
    title += f" | sort: {state.sort_key}"
    if not detailed:
        title += " | d for details"
    return Panel(table, title=literal_text(title), box=box.ROUNDED, padding=(0, 1))


def render_detail_panel(snapshot: dict[str, Any], state: UIState) -> Panel:
    rows: list[ProcessRow] = snapshot["processes"]
    if not rows:
        return Panel("No process selected", title=literal_text("Process Detail"), box=box.ROUNDED)
    row = rows[clamp_int(state.selected, 0, len(rows) - 1)]
    table = Table.grid(expand=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(ratio=1)
    table.add_row("PID", f"{row.pid}")
    table.add_row("User", row.user)
    table.add_row("Status", row.status)
    table.add_row("CPU/Mem", f"{row.cpu_pct:.1f}% / {row.mem_pct:.1f}%")
    table.add_row("RSS/VMS", f"{fmt_bytes(row.rss)} / {fmt_bytes(row.vms)}")
    table.add_row("IO", f"read {fmt_bps(row.read_bps)}  write {fmt_bps(row.write_bps)}")
    gpu_detail = f"{fmt_bytes(row.gpu_mem)} on GPU {row.gpu_ids} ({row.gpu_kinds or '?'})" if row.gpu_ids else "-"
    table.add_row("GPU", gpu_detail)
    try:
        proc = psutil.Process(row.pid)
        with proc.oneshot():
            actual_start_token = process_start_token(proc, snapshot["time"] - snapshot["uptime"])
            if row.start_token and not same_process_start(actual_start_token, row.start_token):
                table.add_row("Detail", "process exited and PID was reused")
                return Panel(table, title=literal_text("Process Detail"), box=box.ROUNDED, padding=(0, 1))
            actual_create_time = proc.create_time()
            table.add_row("Started", dt.datetime.fromtimestamp(actual_create_time).strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Exe", sanitize_text(proc.exe(), 4096))
            table.add_row("CWD", sanitize_text(proc.cwd(), 4096))
            cmd = sanitize_text(" ".join(proc.cmdline()), 8192)
            table.add_row("Command", Text(cmd, overflow="fold"))
            if state.show_env:
                env = proc.environ()
                visible = [f"{key}={sanitize_text(env[key], 4096)}" for key in GPU_ENV_KEYS if key in env]
                table.add_row(
                    "Env", Text("\n".join(visible) if visible else "(no selected GPU/ML env vars)", overflow="fold")
                )
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as exc:
        table.add_row("Detail", f"unavailable: {exc}")
    return Panel(table, title=literal_text("Process Detail"), box=box.ROUNDED, padding=(0, 1))


def render_help_panel() -> Panel:
    help_text = Table.grid(expand=True)
    help_text.add_column(ratio=1)
    help_text.add_column(ratio=1)
    help_text.add_row(
        Text.from_markup(
            "[bold]Navigation[/bold]\n"
            "up/down, PgUp/PgDn  move selection\n"
            "Home/End             jump top/bottom\n"
            "d                    summary/detail resources\n"
            "v                    selected process details\n"
            "e                    details + selected env vars"
        ),
        Text.from_markup(
            "[bold]Actions[/bold]\n"
            "space                pause/resume\n"
            "/                    filter processes\n"
            "u                    clear filter\n"
            "k                    terminate selected PID, asks first\n"
            "r                    reset rate history"
        ),
    )
    help_text.add_row(
        Text.from_markup(
            "[bold]Sorting[/bold]\nc CPU   m memory   g GPU memory\ni process IO       p PID       n name"
        ),
        Text.from_markup(
            "[bold]Display[/bold]\n"
            "1 per-core toggle in detail view\n"
            "a show quiet NICs / pseudo mounts\n"
            "+/- change refresh interval\n"
            "h or ? hide help    q quit"
        ),
    )
    return Panel(help_text, title=literal_text("Help"), box=box.ROUNDED, padding=(0, 1))


def render_footer(state: UIState) -> Panel:
    if not state.interactive:
        text = "snapshot mode  |  use --json for machine-readable output  |  omnitop --help for options"
        return Panel(Text(text, style="dim"), box=box.SIMPLE, padding=(0, 1))
    if state.input_mode == "filter":
        text = f"Filter: {state.filter_buffer}  Enter=apply  Esc=cancel  Backspace=edit"
        return Panel(Text(text, style="bold yellow"), box=box.SIMPLE, padding=(0, 1))
    if state.pending_kill_pid is not None:
        text = f"Terminate PID {state.pending_kill_pid}?  y=SIGTERM  K=SIGKILL  Esc/other=cancel"
        return Panel(Text(text, style="bold red"), box=box.SIMPLE, padding=(0, 1))
    text = "q quit  h help  d detail  / filter  c/m/g sort  v proc  k kill  space pause"
    return Panel(Text(text, style="dim"), box=box.SIMPLE, padding=(0, 1))


def read_key(timeout: float = 0.0) -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
    if not ready:
        return None

    ch = sys.stdin.read(1)
    if ch == "\x03":
        return "ctrl-c"
    if ch in ("\r", "\n"):
        return "enter"
    if ch in ("\x7f", "\b"):
        return "backspace"
    if ch == "\x1b":
        seq = ch
        end = time.monotonic() + 0.03
        while time.monotonic() < end and select.select([sys.stdin], [], [], 0.005)[0]:
            seq += sys.stdin.read(1)
            if seq.endswith("~") or seq[-1:] in {"A", "B", "C", "D", "F", "H"}:
                break
        return {
            "\x1b[A": "up",
            "\x1b[B": "down",
            "\x1b[C": "right",
            "\x1b[D": "left",
            "\x1b[5~": "pageup",
            "\x1b[6~": "pagedown",
            "\x1b[H": "home",
            "\x1b[F": "end",
            "\x1b[1~": "home",
            "\x1b[4~": "end",
        }.get(seq, "esc")
    return ch


def handle_key(key: str, state: UIState, snapshot: dict[str, Any], sampler: Sampler) -> bool:
    if state.input_mode == "filter":
        if key == "enter":
            state.filter_text = state.filter_buffer.strip()
            state.input_mode = None
            state.selected = 0
            state.selected_pid = None
            state.selected_start_token = None
            state.scroll = 0
            state.set_message("filter applied" if state.filter_text else "filter cleared")
        elif key == "esc":
            state.input_mode = None
            state.set_message("filter canceled")
        elif key == "backspace":
            state.filter_buffer = state.filter_buffer[:-1]
        elif len(key) == 1 and key.isprintable() and len(state.filter_buffer) < 256:
            state.filter_buffer += key
        return True

    if state.pending_kill_pid is not None:
        pid = state.pending_kill_pid
        start_token = state.pending_kill_start_token
        state.pending_kill_pid = None
        state.pending_kill_start_token = None
        if key == "y":
            return send_signal(pid, signal.SIGTERM, state, start_token)
        if key == "K":
            return send_signal(pid, signal.SIGKILL, state, start_token)
        state.set_message("kill canceled")
        return True

    rows: list[ProcessRow] = snapshot.get("processes", [])
    if key in {"q", "ctrl-c"}:
        return False
    if key in {"h", "?"}:
        state.show_help = not state.show_help
    elif key == " ":
        state.paused = not state.paused
        state.set_message("paused" if state.paused else "resumed")
    elif key in {"c", "m", "g", "i", "p", "n"}:
        state.sort_key = {"c": "cpu", "m": "mem", "g": "gpu", "i": "io", "p": "pid", "n": "name"}[key]
        state.reverse = state.sort_key != "name"
        state.selected = 0
        state.selected_pid = None
        state.selected_start_token = None
        state.scroll = 0
    elif key == "/":
        state.input_mode = "filter"
        state.filter_buffer = state.filter_text
    elif key == "u":
        state.filter_text = ""
        state.selected = 0
        state.selected_pid = None
        state.selected_start_token = None
        state.scroll = 0
        state.set_message("filter cleared")
    elif key == "r":
        sampler.reset_rates()
        state.set_message("rate history reset")
    elif key == "d":
        state.show_full = not state.show_full
        state.set_message("detail view" if state.show_full else "summary view")
    elif key == "1":
        state.show_per_core = not state.show_per_core
    elif key == "a":
        state.show_all_ifaces = not state.show_all_ifaces
        state.show_all_mounts = not state.show_all_mounts
    elif key == "v":
        state.show_details = not state.show_details
    elif key == "e":
        state.show_details = True
        state.show_env = not state.show_env
    elif key == "+":
        state.interval = min(MAX_INTERVAL, round(state.interval + 0.2, 1))
        state.set_message(f"interval {state.interval:.1f}s")
    elif key == "-":
        state.interval = max(MIN_INTERVAL, round(state.interval - 0.2, 1))
        state.set_message(f"interval {state.interval:.1f}s")
    elif key == "up":
        state.selected = max(0, state.selected - 1)
    elif key == "down":
        state.selected = min(max(0, len(rows) - 1), state.selected + 1)
    elif key == "pageup":
        state.selected = max(0, state.selected - state.page_size)
    elif key == "pagedown":
        state.selected = min(max(0, len(rows) - 1), state.selected + state.page_size)
    elif key == "home":
        state.selected = 0
    elif key == "end":
        state.selected = max(0, len(rows) - 1)
    elif key == "k":
        if rows:
            row = rows[clamp_int(state.selected, 0, len(rows) - 1)]
            state.pending_kill_pid = row.pid
            state.pending_kill_start_token = row.start_token
        else:
            state.set_message("no process selected")

    sync_scroll(state, len(rows))
    state.selected_pid = rows[state.selected].pid if rows else None
    state.selected_start_token = rows[state.selected].start_token if rows else None
    return True


def send_signal(
    pid: int,
    sig: signal.Signals,
    state: UIState,
    expected_start_token: float | None = None,
) -> bool:
    if pid <= 1:
        state.set_message(f"refusing to signal protected PID {pid}")
        return True
    if pid == os.getpid():
        state.set_message("refusing to signal OmniTop itself")
        return True
    try:
        proc = psutil.Process(pid)
        if expected_start_token and not same_process_start(process_start_token(proc), expected_start_token):
            state.set_message(f"refusing PID {pid}: process identity changed")
            return True
        os.kill(pid, sig)
        state.set_message(f"sent {sig.name} to PID {pid}")
    except ProcessLookupError:
        state.set_message(f"PID {pid} no longer exists")
    except PermissionError:
        state.set_message(f"permission denied for PID {pid}")
    except Exception as exc:
        state.set_message(f"failed to signal PID {pid}: {exc}")
    return True


def sync_scroll(state: UIState, row_count: int) -> None:
    if row_count <= 0:
        state.selected = 0
        state.scroll = 0
        return
    state.selected = clamp_int(state.selected, 0, row_count - 1)
    if state.selected < state.scroll:
        state.scroll = state.selected
    elif state.selected >= state.scroll + state.page_size:
        state.scroll = state.selected - state.page_size + 1
    max_scroll = max(0, row_count - state.page_size)
    state.scroll = clamp_int(state.scroll, 0, max_scroll)


def preserve_selection(state: UIState, snapshot: dict[str, Any]) -> None:
    """Keep the selected PID stable while rows re-sort between samples."""

    rows: list[ProcessRow] = snapshot.get("processes", [])
    if not rows:
        state.selected = 0
        state.selected_pid = None
        state.selected_start_token = None
        state.scroll = 0
        return
    if state.selected_pid is not None:
        for index, row in enumerate(rows):
            same_token = state.selected_start_token is None or same_process_start(
                row.start_token or 0.0, state.selected_start_token
            )
            if row.pid == state.selected_pid and same_token:
                state.selected = index
                sync_scroll(state, len(rows))
                return
    state.selected = clamp_int(state.selected, 0, len(rows) - 1)
    state.selected_pid = rows[state.selected].pid
    state.selected_start_token = rows[state.selected].start_token
    sync_scroll(state, len(rows))


def print_once(snapshot: dict[str, Any], state: UIState, console: Console, rows: int | None) -> None:
    if rows is not None:
        state.forced_rows = max(1, rows)
    console.print(render_dashboard(snapshot, state, console))


def interval_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or not MIN_INTERVAL <= parsed <= MAX_INTERVAL:
        raise argparse.ArgumentTypeError(f"must be finite and between {MIN_INTERVAL:g} and {MAX_INTERVAL:g}")
    return parsed


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="omnitop",
        description="Resilient CPU, memory, disk, network, process, and NVIDIA GPU monitor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Interactive keys: h help, q quit, d details, / filter, c/m/g/i/p/n sort, k signal process.",
    )
    parser.add_argument("--version", action="version", version=f"OmniTop {__version__}")
    parser.add_argument("--interval", "-i", type=interval_arg, default=1.0, help="Refresh interval in seconds")
    parser.add_argument("--sort", choices=SORT_KEYS, default="cpu", help="Initial process sort key")
    parser.add_argument("--filter", default="", help="Case-insensitive PID/user/name/command filter")
    parser.add_argument("--user", default="", help="Show only processes owned by this exact user")
    parser.add_argument("--gpu-only", action="store_true", help="Show only processes reported by NVML")
    parser.add_argument("--all-ifaces", action="store_true", help="Show quiet/down network interfaces")
    parser.add_argument("--all-mounts", action="store_true", help="Show pseudo/temp filesystems")
    parser.add_argument("--per-core", action="store_true", help="Show per-core CPU summary")
    parser.add_argument("--full", action="store_true", help="Start in detailed resource view")
    parser.add_argument("--no-gpu", action="store_true", help="Disable NVML/NVIDIA GPU collection")
    parser.add_argument("--no-processes", action="store_true", help="Skip process collection for lower overhead")
    parser.add_argument("--history", type=positive_int_arg, default=90, help="Samples retained for sparklines")
    parser.add_argument("--gpu-retry", type=interval_arg, default=30.0, help="Seconds between NVML reconnect attempts")
    parser.add_argument("--once", action="store_true", help="Render one dashboard snapshot and exit")
    parser.add_argument("--json", action="store_true", help="Emit a stable JSON snapshot (implies --once)")
    parser.add_argument(
        "--count", type=positive_int_arg, default=1, help="Number of JSON samples; values >1 emit JSONL"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print a single JSON result")
    parser.add_argument("--rows", type=positive_int_arg, default=None, help="Process rows in rich snapshot output")
    parser.add_argument("--diagnose", action="store_true", help="Print runtime/NVML diagnostics and exit")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument("--no-alt-screen", action="store_true", help="Keep interactive output in terminal scrollback")
    parser.add_argument("--debug-log", type=Path, default=None, help="Write rotating debug logs to this path")
    args = parser.parse_args(argv)
    if args.history > 3600:
        parser.error("--history cannot exceed 3600 samples")
    if args.count > 1 and not args.json:
        parser.error("--count > 1 requires --json")
    if args.diagnose and args.count > 1:
        parser.error("--diagnose emits one result and cannot be combined with --count > 1")
    if args.pretty and not args.json:
        parser.error("--pretty requires --json")
    if args.pretty and args.count > 1:
        parser.error("--pretty cannot be combined with --count > 1 (JSONL mode)")
    if args.gpu_only and args.no_processes:
        parser.error("--gpu-only cannot be combined with --no-processes")
    if args.gpu_only and args.no_gpu:
        parser.error("--gpu-only cannot be combined with --no-gpu")
    if args.rows is not None and args.json:
        parser.error("--rows applies to rich output and cannot be combined with --json")
    return args


def configure_logging(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(resolved, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.debug("OmniTop %s logging initialized", __version__)


def dependency_versions() -> dict[str, str]:
    result = {}
    for distribution in ("psutil", "rich", "nvidia-ml-py"):
        try:
            result[distribution] = package_version(distribution)
        except PackageNotFoundError:
            result[distribution] = "not installed"
    return result


def diagnostics_payload(sampler: Sampler) -> dict[str, Any]:
    gpu = sampler.gpu.sample()
    return {
        "schema_version": SCHEMA_VERSION,
        "omnitop_version": __version__,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "source": str(Path(__file__).resolve()),
        "dependencies": dependency_versions(),
        "terminal": {
            "stdin_tty": sys.stdin.isatty(),
            "stdout_tty": sys.stdout.isatty(),
            "encoding": sys.stdout.encoding,
        },
        "gpu": {
            "enabled": sampler.gpu.enabled,
            "available": gpu["available"],
            "degraded": gpu.get("degraded", False),
            "error": gpu.get("error", ""),
            "driver": gpu.get("driver", ""),
            "device_count": len(gpu.get("gpus", [])),
            "monitor_only": True,
            "occupancy_source": "NVML running-process queries, not open /dev/nvidia descriptors",
        },
    }


def render_diagnostics(payload: dict[str, Any], console: Console) -> None:
    table = Table(title=f"OmniTop {payload['omnitop_version']} diagnostics", box=box.ROUNDED)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Python", f"{payload['python']} ({payload['python_executable']})")
    table.add_row("Platform", payload["platform"])
    table.add_row("Source", payload["source"])
    dependencies = payload["dependencies"]
    table.add_row("Dependencies", ", ".join(f"{name}={value}" for name, value in dependencies.items()))
    terminal = payload["terminal"]
    table.add_row("Terminal", f"stdin_tty={terminal['stdin_tty']} stdout_tty={terminal['stdout_tty']}")
    gpu = payload["gpu"]
    gpu_value = f"available={gpu['available']} devices={gpu['device_count']} driver={gpu['driver'] or 'n/a'}"
    if gpu["error"]:
        gpu_value += f" error={gpu['error']}"
    table.add_row("NVIDIA", gpu_value)
    table.add_row("GPU semantics", gpu["occupancy_source"])
    console.print(table)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if hasattr(value, "_asdict"):
        return jsonable(value._asdict())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return sanitize_text(str(value), 8192)


def snapshot_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = dict(snapshot)
    gpu = dict(payload["gpu"])
    gpu.pop("proc_map", None)
    payload["gpu"] = gpu
    payload["generated_at"] = dt.datetime.fromtimestamp(snapshot["time"], tz=dt.timezone.utc).isoformat()
    return jsonable(payload)


def write_json(payload: dict[str, Any], pretty: bool = False) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=pretty,
    )
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def run_snapshot_mode(
    sampler: Sampler,
    state: UIState,
    console: Console,
    *,
    json_output: bool,
    count: int,
    pretty: bool,
    rows: int | None,
) -> int:
    state.interactive = False
    sampler.sample(state)  # Prime rate counters.
    time.sleep(min(1.0, state.interval))
    for index in range(count):
        sample_started = time.monotonic()
        snapshot = sampler.sample(state)
        preserve_selection(state, snapshot)
        if json_output:
            write_json(snapshot_payload(snapshot), pretty=pretty)
        else:
            print_once(snapshot, state, console, rows)
        if index + 1 < count:
            time.sleep(max(0.0, state.interval - (time.monotonic() - sample_started)))
    return 0


def run_interactive(
    sampler: Sampler,
    state: UIState,
    console: Console,
    *,
    use_alt_screen: bool,
) -> int:
    sample_started = time.monotonic()
    snapshot = sampler.sample(state)
    preserve_selection(state, snapshot)
    with (
        GracefulSignals(),
        RawTerminal(),
        Live(
            render_dashboard(snapshot, state, console),
            console=console,
            screen=use_alt_screen,
            auto_refresh=False,
        ) as live,
    ):
        next_sample = max(sample_started + state.interval, time.monotonic() + 0.05)
        last_size = console.size
        message_was_visible = bool(state.visible_message())
        running = True
        while running:
            now_mono = time.monotonic()
            wait = 0.25 if state.paused else min(0.25, max(0.0, next_sample - now_mono))
            key = read_key(wait)
            dirty = False
            while key is not None:
                running = handle_key(key, state, snapshot, sampler)
                dirty = True
                if not running:
                    break
                key = read_key(0.0)
            if not running:
                break

            now_mono = time.monotonic()
            if not state.paused and now_mono >= next_sample:
                sample_started = now_mono
                try:
                    snapshot = sampler.sample(state)
                    preserve_selection(state, snapshot)
                    if snapshot.get("warnings"):
                        state.set_message(snapshot["warnings"][0], ttl=max(4.0, state.interval))
                except Exception as exc:
                    state.set_message(f"sampling failed: {sanitize_text(str(exc), 120)}")
                    LOGGER.exception("interactive sampling failed")
                next_sample = max(sample_started + state.interval, time.monotonic() + 0.05)
                dirty = True
            elif dirty and not state.paused:
                next_sample = min(next_sample, time.monotonic() + state.interval)

            if console.size != last_size:
                last_size = console.size
                dirty = True
            message_is_visible = bool(state.visible_message())
            if message_is_visible != message_was_visible:
                message_was_visible = message_is_visible
                dirty = True
            if dirty:
                live.update(render_dashboard(snapshot, state, console), refresh=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        configure_logging(args.debug_log)
    except OSError as exc:
        print(f"omnitop: cannot initialize debug log: {exc}", file=sys.stderr)
        return 2

    state = UIState(
        interval=args.interval,
        sort_key=args.sort,
        reverse=args.sort != "name",
        show_all_ifaces=args.all_ifaces,
        show_all_mounts=args.all_mounts,
        show_per_core=args.per_core,
        show_full=args.full,
        filter_text=sanitize_text(args.filter, 256),
        user_filter=sanitize_text(args.user, 128),
        gpu_only=args.gpu_only,
    )
    console = Console(
        markup=False,
        color_system=None if args.no_color else "auto",
        force_terminal=False if args.no_color else None,
    )
    sampler: Sampler | None = None

    try:
        sampler = Sampler(
            gpu_enabled=not args.no_gpu,
            history_len=args.history,
            gpu_retry_seconds=args.gpu_retry,
            collect_processes=not args.no_processes,
        )
        if args.diagnose:
            payload = diagnostics_payload(sampler)
            if args.json:
                write_json(jsonable(payload), pretty=args.pretty)
            else:
                render_diagnostics(payload, console)
            return 0
        non_interactive_terminal = not (sys.stdin.isatty() and sys.stdout.isatty())
        if args.once or args.json or non_interactive_terminal:
            return run_snapshot_mode(
                sampler,
                state,
                console,
                json_output=args.json,
                count=args.count,
                pretty=args.pretty,
                rows=args.rows,
            )
        return run_interactive(sampler, state, console, use_alt_screen=not args.no_alt_screen)
    except BrokenPipeError:
        with suppress(OSError):
            sys.stdout.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except TerminationRequested as exc:
        return 128 + exc.signum
    except Exception as exc:
        LOGGER.exception("fatal OmniTop error")
        print(f"omnitop: {sanitize_text(str(exc), 300)}", file=sys.stderr)
        return 1
    finally:
        if sampler is not None:
            sampler.close()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def sanitize_text(value: Any, limit: int = 8192) -> str:
    """Remove terminal control characters and cap untrusted process text."""

    text = _decode(value)
    clean = "".join(" " if ord(char) < 32 or 127 <= ord(char) < 160 else char for char in text)
    clean = " ".join(clean.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)] + "…"


def same_process_start(actual: float, expected: float, tolerance: float = 0.01) -> bool:
    """Compare process creation timestamps without relying on PID alone."""

    return abs(float(actual) - float(expected)) <= tolerance


def process_start_token(proc: Any, boot_time: float | None = None) -> float:
    """Return a monotonic process-start token, with a portable fallback.

    On Linux, psutil's backend already parses ``/proc/<pid>/stat`` for CPU
    times. Asking that backend for its monotonic start time inside ``oneshot``
    reuses the same read and avoids re-reading the system boot time once per
    process. The public API remains the fallback for other psutil versions and
    platforms.
    """

    backend = getattr(proc, "_proc", None)
    backend_create_time = getattr(backend, "create_time", None)
    if backend_create_time is not None:
        try:
            return float(backend_create_time(monotonic=True))
        except TypeError:
            pass
    epoch_start = float(proc.create_time())
    epoch_boot = float(boot_time if boot_time is not None else psutil.boot_time())
    return epoch_start - epoch_boot


def _is_quiet_interface(name: str, is_up: bool, rx_bps: float, tx_bps: float) -> bool:
    if name == "lo" or name.startswith(("docker", "br-", "veth", "virbr", "tun", "tap")):
        return rx_bps + tx_bps <= 0
    return not is_up and rx_bps + tx_bps <= 0


def _process_matches(row: ProcessRow, needle: str) -> bool:
    haystack = f"{row.pid} {row.user} {row.name} {row.status} {row.gpu_ids} {row.gpu_kinds} {row.command}".casefold()
    return needle in haystack


def safe_call(func: Any, default: Any) -> Any:
    try:
        return func()
    except Exception:
        return default


def read_cpu_temperature() -> float | None:
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, OSError, NotImplementedError):
        return None
    preferred: list[float] = []
    fallback: list[float] = []
    preferred_groups = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz")
    excluded_groups = ("nvme", "gpu", "amdgpu", "nvidia", "battery")
    preferred_labels = ("package", "tctl", "tdie", "cpu")
    for group, entries in temps.items():
        group_name = group.casefold()
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None or not 0.0 < float(current) < 150.0:
                continue
            label = str(getattr(entry, "label", "")).casefold()
            value = float(current)
            if any(token in group_name for token in preferred_groups) or any(
                token in label for token in preferred_labels
            ):
                preferred.append(value)
            elif not any(token in group_name for token in excluded_groups):
                fallback.append(value)
    candidates = preferred or fallback
    return max(candidates) if candidates else None


def rate(current: float, previous: float, elapsed: float) -> float:
    values = (float(current), float(previous), float(elapsed))
    if not all(math.isfinite(value) for value in values) or values[0] < values[1]:
        return 0.0
    return max(0.0, (values[0] - values[1]) / max(0.001, values[2]))


def clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def style_for_pct(pct: float) -> str:
    if pct >= 90:
        return "bold red"
    if pct >= 75:
        return "yellow"
    if pct >= 50:
        return "cyan"
    return "green"


def pct_text(pct: float | None) -> Text:
    if pct is None:
        return Text("n/a", style="dim")
    return Text(f"{pct:5.1f}%", style=style_for_pct(float(pct)))


def pct_text_plain(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    return f"{float(pct):.1f}%"


def bar(pct: float | None, width: int = 20) -> Text:
    pct = 0.0 if pct is None else clamp(float(pct), 0.0, 100.0)
    filled = round(width * pct / 100.0)
    text = Text("[")
    text.append("#" * filled, style=style_for_pct(pct))
    text.append("-" * (width - filled), style="dim")
    text.append("]")
    return text


def meter(pct: float | None, width: int = 10) -> Text:
    if pct is None:
        return Text("n/a", style="dim")
    value = clamp(float(pct), 0.0, 100.0)
    text = Text(f"{value:5.1f}% ")
    text.append(bar(value, width=width))
    return text


def level_for_pct(pct: float | None, warn: float, crit: float) -> str:
    if pct is None:
        return "unknown"
    value = float(pct)
    if value >= crit:
        return "crit"
    if value >= warn:
        return "warn"
    return "ok"


def level_for_temp(temp_c: float | None) -> str:
    if temp_c is None:
        return "unknown"
    if temp_c >= 85.0:
        return "crit"
    if temp_c >= 75.0:
        return "warn"
    return "ok"


def worst_of(*levels: str) -> str:
    order = {"ok": 0, "warn": 1, "crit": 2, "error": 3}
    known = [level for level in levels if level in order]
    if not known:
        return "unknown"
    return max(known, key=lambda level: order[level])


def state_text(level: str) -> Text:
    labels = {
        "ok": ("OK", "green"),
        "warn": ("WARN", "yellow"),
        "crit": ("CRIT", "bold red"),
        "error": ("ERROR", "bold red"),
        "unknown": ("N/A", "dim"),
    }
    label, style = labels.get(level, labels["unknown"])
    return Text(label, style=style)


def spark(values: Iterable[float], unit_pct: bool) -> str:
    vals = [float(value) for value in values if math.isfinite(float(value))][-32:]
    if not vals:
        return ""
    charset = " .:-=+*#%@"
    if unit_pct:
        low, high = 0.0, 100.0
    else:
        low, high = 0.0, max(vals) or 1.0
    span = max(0.000001, high - low)
    chars = []
    for value in vals:
        idx = int(clamp((float(value) - low) / span, 0.0, 0.9999) * len(charset))
        chars.append(charset[idx])
    return "".join(chars)


def fmt_bytes(num: float | int | None) -> str:
    if num is None:
        return "n/a"
    value = float(num)
    if not math.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{sign}{value:.0f}{unit}"
            if value < 10:
                return f"{sign}{value:.1f}{unit}"
            return f"{sign}{value:.0f}{unit}"
        value /= 1024.0
    return f"{sign}{value:.0f}PiB"


def fmt_bps(num: float | int | None) -> str:
    return f"{fmt_bytes(num)}/s"


def fmt_pct(num: float | int | None) -> str:
    if num is None:
        return "n/a"
    return f"{float(num):5.1f}%"


def fmt_int(num: int | float) -> str:
    value = float(num)
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1000.0 or suffix == "T":
            return f"{value:.0f}{suffix}"
        value /= 1000.0
    return f"{value:.0f}T"


def fmt_duration(seconds: float) -> str:
    if not math.isfinite(float(seconds)):
        return "n/a"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def shorten_mount(path: str, width: int) -> str:
    if width <= 3:
        return path[: max(0, width)]
    if len(path) <= width:
        return path
    return "..." + path[-(width - 3) :]


if __name__ == "__main__":
    raise SystemExit(main())
