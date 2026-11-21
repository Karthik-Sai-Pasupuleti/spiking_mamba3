"""Save and load checkpoints.

Our checkpoints are dicts with a `"model"` state dict (`last.pt` also has `"optimizer"`
and `"step"`). Upstream checkpoints such as `weights/mamba3-siso-187m/pytorch_model.bin`
are a bare state dict. `load_model_state` reads both.
"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write to a temporary file, then rename, so a crash never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str = "cuda") -> dict[str, Any]:
    """Load one of our checkpoints (`last.pt` or `final.pt`)."""
    return torch.load(path, map_location=device, weights_only=True)


def load_model_state(path: Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Return the model state dict of our checkpoint or of an upstream checkpoint."""
    state = torch.load(path, map_location=device, weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state
