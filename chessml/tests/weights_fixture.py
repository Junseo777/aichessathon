import json
from pathlib import Path

import pytest

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"


def require_weights() -> Path:
    if not (WEIGHTS / "manifest.json").exists():
        pytest.skip(
            f"no export in {WEIGHTS}: `./use-weights.sh <run>` for a trained net, "
            "`make weights` for random init",
            allow_module_level=True,
        )
    return WEIGHTS


def needs_trained_net() -> pytest.MarkDecorator:
    """Skip a strength test on a random-init export (manifest checkpoint null), as CI builds."""
    manifest: dict[str, object] = json.loads((WEIGHTS / "manifest.json").read_text())
    random_init = "checkpoint" in manifest and manifest["checkpoint"] is None
    return pytest.mark.skipif(
        random_init, reason=f"{WEIGHTS} is a random-init export; strength tests need a trained net"
    )
