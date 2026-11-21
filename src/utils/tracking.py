"""Experiment tracking with MLflow: one MLflow run per training run.

`MLflowTracker` records what is needed to repeat a run and its results:
- at the start: every config value as a param, the original YAML file and the command
  line, the git commit, branch and dirty flag, the environment (GPU, torch, CUDA, Python),
  and, if the working tree has uncommitted changes, a zip of the code as it is on disk;
- during training: the metrics of every log step;
- at the end: `config.json`, `train.log` and `metrics.csv`, and the run status.

Runs go to `sqlite:///<repo>/mlflow.db` with artifacts in `<repo>/mlartifacts/`, unless
the environment variable MLFLOW_TRACKING_URI names another store. View them with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root. A resumed run
folder (`--run-dir`) continues its MLflow run: the run id is kept in the folder.
"""

import logging
import math
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import mlflow
import torch

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
SNAPSHOT_PATHS = ["src", "configs", "pyproject.toml", "uv.lock"]
RUN_ID_FILE = "mlflow_run_id.txt"


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> flat dict with dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def artifact_location(uri: str) -> str | None:
    """Where a new experiment keeps its files.

    For a local SQLite store: `mlartifacts/` next to the database file. Without this,
    MLflow would use `./mlruns` in the current folder. For other stores (a tracking
    server) None, so the server decides.
    """
    if uri.startswith("sqlite:///"):
        db_file = Path(uri.removeprefix("sqlite:///")).resolve()
        return (db_file.parent / "mlartifacts").as_uri()
    return None


def git(*args: str) -> str:
    """Output of a git command in the repo, or '' if git fails."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_state() -> dict[str, str]:
    """Commit, branch and dirty flag ("true" if there are changed or untracked files)."""
    return {
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": "true" if git("status", "--porcelain") else "false",
    }


def write_code_snapshot(path: Path) -> int:
    """Zip the code in SNAPSHOT_PATHS as it is on disk; return the number of files.

    Takes tracked and untracked files but skips git-ignored ones (data, runs, caches),
    and tracked files that are deleted on disk.
    """
    listing = git(
        "ls-files", "--cached", "--others", "--exclude-standard", "--", *SNAPSHOT_PATHS
    )
    files = sorted({name for name in listing.splitlines() if (ROOT / name).is_file()})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, arcname=name)
    return len(files)


class MLflowTracker:
    """One MLflow run for one run folder. Only rank 0 should build it.

    Call `log_start` once after the run folder and `config.json` exist, `log_metrics`
    with every metrics row, and `finish` at the end (also after a crash).
    """

    def __init__(self, run_dir: Path, experiment: str) -> None:
        uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        mlflow.set_tracking_uri(uri)
        found = mlflow.get_experiment_by_name(experiment)
        if found is not None:
            experiment_id = found.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                experiment, artifact_location=artifact_location(uri)
            )
        self.run_dir = run_dir
        id_file = run_dir / RUN_ID_FILE
        self.resumed = id_file.exists()
        run_id = id_file.read_text().strip() if self.resumed else None
        run = mlflow.start_run(
            run_id=run_id, experiment_id=experiment_id, run_name=run_dir.name
        )
        id_file.write_text(run.info.run_id)
        logger.info(
            "MLflow run %s in experiment %r at %s", run.info.run_id, experiment, uri
        )

    def log_start(
        self, config_path: Path, record: dict[str, Any], command: list[str]
    ) -> None:
        """Log config, code version and environment. On a resumed run, only new tags.

        `record` is the content of `config.json`: "config", "environment" and the derived
        values (step counts, parameter count).
        """
        state = git_state()
        if self.resumed:
            mlflow.set_tags(
                {
                    "resumed_at_step": record["start_step"],
                    "resume_git_commit": state["git_commit"],
                    "resume_git_dirty": state["git_dirty"],
                }
            )
            return
        derived = {k: v for k, v in record.items() if k not in ("config", "environment")}
        derived.pop("start_step", None)
        mlflow.log_params({**flatten(record["config"]), **derived})
        mlflow.set_tags(
            {
                **state,
                "mlflow.source.git.commit": state["git_commit"],
                "mlflow.source.git.branch": state["git_branch"],
                "mlflow.source.name": "src/train.py",
                "command": " ".join(command),
                "config_file": str(config_path),
                "run_dir": str(self.run_dir),
                "gpu": record["environment"]["gpu"],
                "torch": record["environment"]["torch"],
                "cuda": str(torch.version.cuda),
                "python": record["environment"]["python"],
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="config")
        if state["git_dirty"] == "true":
            snapshot = self.run_dir / "source.zip"
            n_files = write_code_snapshot(snapshot)
            mlflow.log_artifact(str(snapshot), artifact_path="code")
            logger.warning(
                "uncommitted changes: the code (%d files) is saved as code/source.zip; "
                "commit before real runs so git_commit alone identifies the code",
                n_files,
            )

    def log_metrics(self, row: dict[str, float]) -> None:
        """Log one metrics row at its "step". Missing and NaN values are left out."""
        values = {
            key: float(value)
            for key, value in row.items()
            if key != "step" and value is not None and not math.isnan(value)
        }
        mlflow.log_metrics(values, step=int(row["step"]))

    def finish(self, status: str) -> None:
        """Upload config.json, train.log and metrics.csv, then end the run with `status`.

        `status` is "FINISHED", "FAILED" or "KILLED".
        """
        for name in ("config.json", "train.log", "metrics.csv"):
            path = self.run_dir / name
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="run")
        mlflow.end_run(status=status)
