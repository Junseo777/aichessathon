import json
from pathlib import Path

import pytest

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"


def require_weights() -> Path:
    if not (WEIGHTS / "manifest.json").exists():
        pytest.skip(f"no export in {WEIGHTS}; run `make weights`", allow_module_level=True)
    return WEIGHTS


def needs_trained_net() -> pytest.MarkDecorator:
    """Skips a strength test when weights/ is a random-init export, which is what CI
    builds. Exports record the checkpoint they came from; None means there was none."""
    manifest: dict[str, object] = json.loads((WEIGHTS / "manifest.json").read_text())
    random_init = "checkpoint" in manifest and manifest["checkpoint"] is None
    return pytest.mark.skipif(
        random_init, reason=f"{WEIGHTS} is a random-init export; strength tests need a trained net"
    )
