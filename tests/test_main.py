import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

def run_cli(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["OFFLINE"] = "1"         # <--- чтобы не ходить в интернет
    cmd = [sys.executable, "-m", "src.main", *args]
    return subprocess.check_output(cmd, env=env).decode("utf-8")

def test_cli_price_ticks_runs():
    out = run_cli(["price-ticks", "--n", "2", "--delay", "0.1"])
    assert "values" in out and "count" in out

def test_binance_klines_function_import():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    sys.path.insert(0, str(SRC))

    from binance.api import get_klines
    assert callable(get_klines)
