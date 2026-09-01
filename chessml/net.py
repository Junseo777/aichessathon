import json
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from chessml.encoding import encode_move, featurize, mirror_move, transposition_key

Evaluation = tuple[list[chess.Move], npt.NDArray[np.float32], float]

_NO_PRIORS: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)


class PolicyValueNet:
    def __init__(self, session: ort.InferenceSession, name: str, cache_size: int = 60_000) -> None:
        self.session = session
        self.name = name
        self.cache_size = cache_size
        self.cache: dict[object, Evaluation] = {}
        self.hits = 0
        self.forwards = 0

    def raw(self, x: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], float]:
        policy, value = self.session.run(None, {"x": x[np.newaxis]})
        self.forwards += 1
        return policy[0], float(value[0, 0])

    def evaluate(self, board: chess.Board) -> Evaluation:
        cacheable = not board.move_stack
        key: object = None
        if cacheable:
            key = (
                transposition_key(board),
                min(board.halfmove_clock, 100) // 2,
                min(board.fullmove_number, 200) // 2,
            )
            hit = self.cache.get(key)
            if hit is not None:
                self.hits += 1
                return hit

        moves = list(board.legal_moves)
        if not moves:
            return moves, _NO_PRIORS, 0.0
        logits, value = self.raw(featurize(board))
        rotate = board.turn == chess.BLACK
        indices = [encode_move(mirror_move(m) if rotate else m) for m in moves]
        legal_logits = logits[indices]
        legal_logits -= legal_logits.max()
        priors = np.exp(legal_logits)
        priors /= priors.sum()
        result: Evaluation = (moves, priors.astype(np.float32), value)

        if cacheable:
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[key] = result
        return result


def _make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _time_forward(session: ort.InferenceSession, runs: int = 15) -> float:
    x = np.zeros((1, 21, 8, 8), dtype=np.float32)
    for _ in range(5):
        session.run(None, {"x": x})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"x": x})
    return (time.perf_counter() - start) / runs


def load_fastest(weights_dir: Path) -> tuple[PolicyValueNet, dict[str, Any]]:
    manifest: dict[str, Any] = json.loads((weights_dir / "manifest.json").read_text())
    names = ("model.int8.onnx", "model.onnx")
    candidates = [name for name in names if (weights_dir / name).exists()]
    if not candidates:
        raise FileNotFoundError(f"no .onnx files in {weights_dir}")
    sessions = {name: _make_session(weights_dir / name) for name in candidates}
    timed = sorted((_time_forward(session), name) for name, session in sessions.items())
    best_ms, best_name = timed[0][0] * 1000.0, timed[0][1]
    manifest["forward_ms"] = round(best_ms, 3)
    manifest["chosen"] = best_name
    print(f"net: {[(n, round(t * 1000, 2)) for t, n in timed]} -> {best_name}")
    return PolicyValueNet(sessions[best_name], best_name), manifest
