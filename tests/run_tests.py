"""Run the tests without pytest: python tests/run_tests.py (pytest works too: pytest tests/)."""
import inspect
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from pathlib import Path  # noqa: E402

fails = 0
for mod in ["test_synthetic", "test_exactness", "test_exoj_reference"]:
    m = __import__(mod)
    for name, fn in inspect.getmembers(m, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d)) if "tmp_path" in inspect.signature(fn).parameters else fn()
                print(f"PASS {mod}.{name}")
            except Exception:
                fails += 1
                print(f"FAIL {mod}.{name}")
                traceback.print_exc()
print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
