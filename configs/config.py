"""Load an experiment config from `configs/<name>.yaml`.

A config has the sections `data`, `model`, `optim` and `train`, plus `run_name` and
`seed`. `--smoke` shrinks it to a 20-step run of a tiny model, to check the pipeline.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path, smoke: bool = False) -> dict[str, Any]:
    """Read the YAML config. With `smoke`, shrink the model, sequences and steps.

    A smoke run starts from random weights: `train.init_from` is dropped, because the
    shrunken model cannot load a full-size checkpoint.
    """
    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if smoke:
        cfg["run_name"] += "-smoke"
        cfg["data"].update(seq_len=256, val_samples=4)
        cfg["model"].update(d_model=128, d_intermediate=256, n_layer=2)
        cfg["train"].update(
            epochs=1,
            batch_size=12,  # a multiple of micro 2 x 1, 2, 3 or 6 GPUs
            micro_batch_size=2,
            max_steps=20,
            log_every=1,
            val_every=10,
            save_every=10,
            init_from=None,
        )
    return cfg
