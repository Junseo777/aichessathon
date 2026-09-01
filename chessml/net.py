import json
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from chessml.encoding import encode_move, featurize, mirror_move


class PolicyValueNet:
    def __init__(self, session: ort.InferenceSession, name: str) -> None:
        self.session = session
        self.name = name

    def raw(self, x: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], float]:
        policy, value = self.session.run(None, {"x": x[np.newaxis]})
        return policy[0], float(value[0, 0])

    def evaluate(
        self, board: chess.Board
    ) -> tuple[list[chess.Move], npt.NDArray[np.float32], float]:
        logits, value = self.raw(featurize(board))
        moves = list(board.legal_moves)
        rotate = board.turn == chess.BLACK
        indices = [encode_move(mirror_move(m) if rotate else m) for m in moves]
        legal_logits = logits[indices]
        legal_logits -= legal_logits.max()
        priors = np.exp(legal_logits)
        priors /= priors.sum()
        return moves, priors.astype(np.float32), value


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
