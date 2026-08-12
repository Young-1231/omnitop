from __future__ import annotations

from rich.console import Console

from omnitop.app import Sampler, UIState, render_dashboard


def test_dashboard_renders_in_narrow_terminal_without_control_markup() -> None:
    state = UIState(interval=1.0, interactive=False, forced_rows=2)
    sampler = Sampler(gpu_enabled=False, collect_processes=False)
    try:
        snapshot = sampler.sample(state)
    finally:
        sampler.close()
    console = Console(record=True, width=60, height=28, color_system=None, force_terminal=False)
    console.print(render_dashboard(snapshot, state, console))
    output = console.export_text()
    assert "OmniTop 2.0.0" in output
    assert "Overview" in output
    assert "Top Processes" in output
    assert "snapshot mode" in output
