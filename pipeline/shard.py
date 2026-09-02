from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from chessml.encoding import NUM_PLANES

MULTIPV = 4

ARRAYS: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "X": (np.dtype(np.int8), (NUM_PLANES, 8, 8)),
    "Y_policy": (np.dtype(np.int32), ()),
    "Y_value": (np.dtype(np.int8), ()),
    "Y_value_engine": (np.dtype(np.float16), ()),
    "Y_policy_engine": (np.dtype(np.int16), ()),
    "Y_value_engine4": (np.dtype(np.float16), (MULTIPV,)),
    "Y_policy_engine4": (np.dtype(np.int16), (MULTIPV,)),
    "Y_depth_engine4": (np.dtype(np.int8), (MULTIPV,)),
    "Y_value_lichess": (np.dtype(np.float16), ()),
    "tier": (np.dtype(np.int8), ()),
    "value_source": (np.dtype(np.int8), ()),
    "Z": (np.dtype(np.uint64), ()),
    "split": (np.dtype(np.int8), ()),
    "fen_offset": (np.dtype(np.uint64), ()),
}

SOURCE_NONE = 0
SOURCE_LICHESS = 1
SOURCE_STOCKFISH = 2

TIER_TOP = 0
TIER_MID = 1

SPLIT_TRAIN = 0
SPLIT_VAL = 1

BYTES_PER_ROW = sum(dt.itemsize * int(np.prod(shape, dtype=int)) for dt, shape in ARRAYS.values())


def _path(root: Path, name: str) -> Path:
    return root / f"{name}.bin"


@dataclass
class ShardWriter:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._fh = {name: open(_path(self.root, name), "wb") for name in ARRAYS}  # noqa: SIM115
        self._fen = open(self.root / "fen.txt", "wb")  # noqa: SIM115
        self._fen_bytes = 0
        self.n = 0

    def append(
        self,
        *,
        x: npt.NDArray[np.int8],
        y_policy: int,
        y_value: int,
        z: int,
        fen: str,
        split: int,
        y_value_engine: float = float("nan"),
        y_policy_engine: int = -1,
        y_value_lichess: float = float("nan"),
        tier: int = 0,
        value_source: int = SOURCE_NONE,
    ) -> None:
        line = fen.encode("ascii") + b"\n"
        self._write("fen_offset", self._fen_bytes)
        self._fen.write(line)
        self._fen_bytes += len(line)

        self._fh["X"].write(np.ascontiguousarray(x, dtype=np.int8).tobytes())
        self._write("Y_policy", y_policy)
        self._write("Y_value", y_value)
        self._write("Y_value_engine", y_value_engine)
        self._write("Y_policy_engine", y_policy_engine)
        self._write("Y_value_engine4", np.full(MULTIPV, np.nan))
        self._write("Y_policy_engine4", np.full(MULTIPV, -1))
        self._write("Y_depth_engine4", np.full(MULTIPV, -1))
        self._write("Y_value_lichess", y_value_lichess)
        self._write("tier", tier)
        self._write("value_source", value_source)
        self._write("Z", z)
        self._write("split", split)
        self.n += 1

    def _write(self, name: str, value: Any) -> None:
        dtype, _ = ARRAYS[name]
        self._fh[name].write(np.asarray(value, dtype=dtype).tobytes())

    def write_meta(self, meta: dict[str, Any]) -> None:
        (self.root / "meta.json").write_text(json.dumps({**meta, "n_samples": self.n}, indent=2))

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fen.close()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Shard:
    def __init__(self, root: Path, mode: Literal["r", "r+"] = "r") -> None:
        self.root = Path(root)
        self.meta: dict[str, Any] = json.loads((self.root / "meta.json").read_text())
        self.n = int(self.meta["n_samples"])
        self._mode = mode
        self._cache: dict[str, np.memmap[Any, Any]] = {}

    def __getattr__(self, name: str) -> np.memmap[Any, Any]:
        if name not in ARRAYS:
            raise AttributeError(name)
        if name not in self._cache:
            dtype, shape = ARRAYS[name]
            self._cache[name] = np.memmap(
                _path(self.root, name), dtype=dtype, mode=self._mode, shape=(self.n, *shape)
            )
        return self._cache[name]

    def ensure_columns(self) -> list[str]:
        added: list[str] = []
        for name, (dtype, shape) in ARRAYS.items():
            path = _path(self.root, name)
            want = self.n * int(np.prod(shape, dtype=int)) * dtype.itemsize
            if path.exists() and path.stat().st_size == want:
                continue
            arr = np.memmap(path, dtype=dtype, mode="w+", shape=(self.n, *shape))
            arr[:] = np.nan if dtype.kind == "f" else -1
            arr.flush()
            del arr
            added.append(name)
        self._cache.clear()
        return added

    def fen(self, i: int) -> str:
        with open(self.root / "fen.txt", "rb") as fh:
            fh.seek(int(self.fen_offset[i]))
            return fh.readline().decode("ascii").strip()

    def fens(self) -> list[str]:
        return (self.root / "fen.txt").read_text().splitlines()
