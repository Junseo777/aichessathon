from pathlib import Path

import pytest

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"


def require_weights() -> Path:
    if not (WEIGHTS / "manifest.json").exists():
        pytest.skip(f"no export in {WEIGHTS}; run `make weights`", allow_module_level=True)
    return WEIGHTS
