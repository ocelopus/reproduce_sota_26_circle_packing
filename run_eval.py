"""Local evaluation wrapper.

The sandbox reports an empty ``sys.executable``, which makes evaluator.run_with_timeout
fail with PermissionError. We patch ``sys.executable`` to the real interpreter path
so the (unmodified) evaluator works. Usage:  python run_eval.py <program.py>
"""
import sys
import os
import shutil

if not sys.executable:
    sys.executable = shutil.which("python3") or shutil.which("python") or "/usr/local/bin/python"

import evaluator  # noqa: E402

if __name__ == "__main__":
    path = sys.argv[1]
    res = evaluator.evaluate(path)
    print("RESULT", res)
