import contextlib
import os
import sys

_DIR = os.environ.get("CHESS_TELEMETRY_DIR")


def _is_agent_runner() -> bool:
    return any("runner.py" in str(a) for a in sys.argv)


class _Tee:
    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, data):
        with contextlib.suppress(Exception):
            self._sink.write(data)
            self._sink.flush()
        return self._stream.write(data)

    def flush(self):
        with contextlib.suppress(Exception):
            self._sink.flush()
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


if _DIR and _is_agent_runner():
    try:
        agent = "unknown"
        for a in sys.argv[1:]:
            if "runner.py" not in str(a):
                agent = os.path.basename(str(a).rstrip("/\\")) or "unknown"
                break
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"agent_{agent}_{os.getpid()}.log")
        sink = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = _Tee(sys.stdout, sink)
        sys.stderr = _Tee(sys.stderr, sink)
    except Exception:
        pass
