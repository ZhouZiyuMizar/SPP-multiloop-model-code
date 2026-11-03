# -*- coding: utf-8 -*-
"""Run the model checks, the six manuscript cases, the parameter sweep and the harmonic scans."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
script = root / "source" / "spp_model.py"
out = root / "results"
for command in ("check", "cases", "sweep", "harmonics"):
    print("Running", command, flush=True)
    subprocess.run([sys.executable, str(script), command, "--out", str(out)], check=True)
print("Results written to", out)
