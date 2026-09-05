import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from harness.sandbox import RUNNER, SUSPEND_IDLE, Agent, local

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PAUSE_S = 0.5

# an agent whose background thread keeps stepping; a suspension shows up as one long gap
SPINNER = textwrap.dedent(
    """
    import threading
    import time

    ticks = 0
    longest_gap = 0.0


    def spin() -> None:
        global ticks, longest_gap
        last = time.monotonic()
        while True:
            now = time.monotonic()
            longest_gap = max(longest_gap, now - last)
            last = now
            ticks += 1
            time.sleep(0.001)


    threading.Thread(target=spin, daemon=True).start()


    def get_move(fen: str, time_left_ms: int) -> str:
        seen = ticks
        while ticks == seen:  # let the spinner take one step after this request arrived
            time.sleep(0.001)
        return f"{ticks} {longest_gap:.3f}"
    """
)

posix = pytest.mark.skipif(not SUSPEND_IDLE, reason="no stop signal on this platform")


@pytest.fixture
def spinner(tmp_path: Path) -> Path:
    (tmp_path / "agent.py").write_text(SPINNER)
    return tmp_path


def _state(agent: Agent) -> str:
    process = agent._process
    assert process is not None
    out = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(process.pid)], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _longest_gap(reply: str) -> float:
    return float(reply.split()[1])


@posix
def test_idle_agent_is_stopped_between_moves(spinner: Path) -> None:
    agent = local(spinner)
    agent.start(init_budget_s=10.0)
    try:
        assert _state(agent).startswith("T"), "stopped from the ready line to its first request"
        agent.move(FEN, 10_000)
        assert _state(agent).startswith("T"), "stopped again once it has replied"
        time.sleep(PAUSE_S)
        assert _longest_gap(agent.move(FEN, 10_000)) >= PAUSE_S * 0.8, "no CPU while idle"
    finally:
        agent.stop()


@posix
def test_suspension_can_be_switched_off(spinner: Path) -> None:
    agent = Agent([sys.executable, str(RUNNER), str(spinner)], suspend_idle=False)
    agent.start(init_budget_s=10.0)
    try:
        agent.move(FEN, 10_000)
        assert not _state(agent).startswith("T")
        time.sleep(PAUSE_S)
        assert _longest_gap(agent.move(FEN, 10_000)) < PAUSE_S * 0.8, "kept running while idle"
    finally:
        agent.stop()


@posix
def test_stop_kills_a_stopped_agent(spinner: Path) -> None:
    agent = local(spinner)
    agent.start(init_budget_s=10.0)
    process = agent._process
    assert process is not None and _state(agent).startswith("T")
    agent.stop()
    assert process.poll() is not None
