from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.model import NUM_MOVES  # noqa: E402
from train.train import policy_target  # noqa: E402


def reference(
    human: np.ndarray, moves: np.ndarray, values: np.ndarray, alpha: float, temperature: float
) -> np.ndarray:
    """The target, computed one row at a time."""
    out = np.zeros((len(human), NUM_MOVES), dtype=np.float64)
    for r, h in enumerate(human):
        lines = [(int(m), float(v)) for m, v in zip(moves[r], values[r], strict=True) if m >= 0]
        if not lines:
            out[r, h] = 1.0
            continue
        w = np.exp(np.array([v for _, v in lines]) / temperature)
        w /= w.sum()
        for (m, _), wk in zip(lines, w, strict=True):
            out[r, m] += (1.0 - alpha) * wk
        out[r, h] += alpha
    return out


def batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    human = np.array([5, 7, 9, 11, 13], dtype=np.int64)
    moves = np.array(
        [
            [5, 6, 7, 8],  # four lines, the human move is the best line
            [1, 7, 3, 4],  # four lines, the human move is the second line
            [-1, -1, -1, -1],  # no lines at all
            [20, 21, -1, -1],  # two lines, the human move outside them
            [13, 14, 15, -1],  # three lines
        ],
        dtype=np.int64,
    )
    values = np.array(
        [
            [0.5, 0.4, 0.1, -0.2],
            [0.9, 0.85, 0.0, -0.9],
            [np.nan] * 4,
            [0.0, -0.05, np.nan, np.nan],
            [-0.3, -0.31, -0.6, np.nan],
        ],
        dtype=np.float32,
    )
    return human, moves, values


def target(alpha: float, temperature: float) -> np.ndarray:
    human, moves, values = batch()
    soft = (torch.from_numpy(moves), torch.from_numpy(values))
    return policy_target(torch.from_numpy(human), soft, alpha, temperature).numpy()


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("temperature", [0.02, 0.05, 0.2])
def test_matches_the_row_by_row_reference(alpha: float, temperature: float) -> None:
    human, moves, values = batch()
    got = target(alpha, temperature)
    np.testing.assert_allclose(got, reference(human, moves, values, alpha, temperature), atol=1e-6)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-6)


def test_no_lines_means_one_hot_on_the_human_move() -> None:
    human, _, _ = batch()
    got = target(0.0, 0.05)
    assert got[2, 9] == 1.0 and got[2].sum() == 1.0
    plain = policy_target(torch.from_numpy(human), None, 0.0, 0.05).numpy()
    np.testing.assert_array_equal(plain, np.eye(NUM_MOVES)[human])


def test_absent_lines_carry_nothing_and_nan_never_leaks() -> None:
    got = target(0.0, 0.05)
    assert np.isfinite(got).all()
    assert got[3, 20] + got[3, 21] == pytest.approx(1.0)
    assert got[3, 11] == 0.0  # alpha 0: a human move outside the lines gets nothing
    assert got[4, 13] + got[4, 14] + got[4, 15] == pytest.approx(1.0)
    assert got[2, 0] == 0.0 and got[3, 0] == 0.0  # clamped -1 indices add nothing to move 0


def test_temperature_sets_the_sharpness() -> None:
    cold, warm = target(0.0, 0.02), target(0.0, 0.2)
    assert cold[0, 5] > warm[0, 5] > 0.25
    assert cold[1, 1] > warm[1, 1]
