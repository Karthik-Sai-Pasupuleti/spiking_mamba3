"""The output folder of one run: `runs/<run_name>-<timestamp>/` or a fixed folder.

`create_run_dir` makes a new timestamped folder. A fixed folder (`--run-dir`) is used
for runs that continue over several cluster jobs; then `train.log` and `metrics.csv` are
appended to, not overwritten. `setup_logging` sends log lines to `train.log` and to the
terminal. `write_config` saves the config plus the environment (seed, git commit,
hardware). `MetricsCSV` appends one row per logged step to `metrics.csv`.
"""

import csv
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

METRIC_COLUMNS = [
    "step",
    "tokens_seen",
    "train_loss",
    "val_loss",
    "val_ppl",
    "lr",
    "grad_norm",
    "tokens_per_s",
    # Spiking models only, on validation steps; empty otherwise.
    "spike_rate",
    "spike_rate_in_proj",
    "spike_rate_out_proj",
]


def create_run_dir(runs_dir: Path, run_name: str) -> Path:
    """Create and return `runs_dir/<run_name>-<YYYYmmdd-HHMMSS>/`."""
    run_dir = runs_dir / f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    return run_dir


def setup_logging(log_file: Path) -> None:
    """Log INFO and above to `log_file` and to the terminal, with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def git_commit() -> str:
    """Current git commit hash, or 'unknown' outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment_info() -> dict[str, str]:
    """Git commit, GPU, torch and Python versions, for the run record."""
    return {
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def write_config(path: Path, record: dict[str, Any]) -> None:
    """Write the run record (config plus derived values) as indented JSON."""
    path.write_text(json.dumps(record, indent=2, default=str))


class MetricsCSV:
    """Append-only `metrics.csv` with the columns `METRIC_COLUMNS`. Flushes every row.

    If the file already has rows (a resumed run), new rows go after them and the header
    is not written again.
    """

    def __init__(self, path: Path) -> None:
        is_new = not path.exists() or path.stat().st_size == 0
        self.file = path.open("a", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(METRIC_COLUMNS)

    def write(self, row: dict[str, float]) -> None:
        """Write one row. Floats keep 5 significant digits; integers stay as they are.

        A column missing from `row` is written as an empty field.
        """
        values = []
        for column in METRIC_COLUMNS:
            value = row.get(column)
            if value is None:
                values.append("")
            else:
                values.append(f"{value:.5g}" if isinstance(value, float) else str(value))
        self.writer.writerow(values)
        self.file.flush()

    def close(self) -> None:
        """Close the file."""
        self.file.close()
