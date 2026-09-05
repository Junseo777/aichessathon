"""Build a sparring agent dir from a chessml snapshot with explicit search settings.

usage: make_agent.py <dir> <net-run> <snapshot-dir> --mcts "<kwargs>" [--temp T]
  <snapshot-dir> holds agent.py and chessml/ (a git archive of one commit)
  --mcts   the keyword arguments for MCTS(_NET, ...), e.g. "fpu_reduction=0.25, proofs=False"
  --temp   policy softmax temperature passed to load_fastest (default 1.0)
"""
import argparse, re, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path); ap.add_argument("net"); ap.add_argument("snapshot", type=Path)
ap.add_argument("--mcts", required=True); ap.add_argument("--temp", default="1.0")
a = ap.parse_args()
if a.dir.exists():
    shutil.rmtree(a.dir)
a.dir.mkdir(parents=True)
shutil.copytree(a.snapshot / "chessml", a.dir / "chessml", ignore=shutil.ignore_patterns("tests", "__pycache__"))
src = (a.snapshot / "agent.py").read_text()
src, n1 = re.subn(r'load_fastest\(\s*Path\(__file__\)\.resolve\(\)\.parent / "weights"[^)]*\)',
                  f'load_fastest(Path(__file__).resolve().parent / "weights", policy_temperature={a.temp})', src, flags=re.S)
src, n2 = re.subn(r"_MCTS = MCTS\(.*?\n\)\n|_MCTS = MCTS\([^\n]*\)\n", f"_MCTS = MCTS(_NET, {a.mcts})\n", src, flags=re.S)
assert n1 == 1 and n2 == 1, (n1, n2)
(a.dir / "agent.py").write_text(src)
w = a.dir / "weights"; w.mkdir()
store = Path("/Users/junseo/chess-master/weights") / a.net
for f in ("model.onnx", "manifest.json"):
    (w / f).symlink_to(store / f)
print(f"{a.dir}: net {a.net}, temp {a.temp}, MCTS({a.mcts})")
