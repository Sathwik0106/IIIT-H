"""Run the full non-UI emotion recognition workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


COMMANDS = [
    ["src/dataset.py"],
    ["models/speech_pipeline/train.py"],
    ["models/speech_pipeline/test.py"],
    ["models/text_pipeline/train.py"],
    ["models/text_pipeline/test.py"],
    ["models/fusion_pipeline/train.py"],
    ["models/fusion_pipeline/test.py"],
    ["src/artifacts.py"],
]


def main() -> None:
    for command in COMMANDS:
        script = PROJECT_ROOT / command[0]
        print(f"\nRunning {script.relative_to(PROJECT_ROOT)}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
