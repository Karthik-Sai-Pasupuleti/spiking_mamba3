"""Validation: next-token loss and perplexity of a model on token samples.

`next_token_loss` is also the training loss. Run as a script, it loads a checkpoint and
reports the validation loss: `python eval.py --config <yaml> --checkpoint <pt>`.
"""

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def next_token_loss(
    model: nn.Module, inputs: Tensor, targets: Tensor, dtype: torch.dtype
) -> Tensor:
    """Cross-entropy of predicting token t+1 from tokens up to t.

    `inputs` and `targets` are int64 tensors (n, seq_len) on the GPU, as `LMDataset` gives
    them: `targets` is `inputs` moved one token to the left. The forward pass runs under
    autocast with `dtype`; the loss is computed in float32.
    """
    with torch.autocast("cuda", dtype=dtype):
        logits = model(inputs)
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    return F.cross_entropy(flat_logits, targets.reshape(-1))


@torch.no_grad()
def mean_loss(model: nn.Module, loader: DataLoader, dtype: torch.dtype) -> float:
    """Mean next-token loss over all micro-batches of `loader`."""
    was_training = model.training
    model.eval()
    losses = []
    for inputs, targets in loader:
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        losses.append(next_token_loss(model, inputs, targets, dtype).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    """Load a checkpoint and print the mean loss over the validation windows."""
    from configs.config import load_config
    from dataset.dataloader import build_val_loader
    from dataset.dataset import token_dataset
    from models.layers.spiking import track_spikes
    from models.model import MambaLMHeadModel, MambaConfig
    from utils.checkpoint import load_model_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(args.config)
    model = MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")
    model.load_state_dict(load_model_state(args.checkpoint))
    data = cfg["data"]
    val = token_dataset(data["val_bin"], data["seq_len"])
    loader = build_val_loader(
        val, data["val_samples"], cfg["train"]["micro_batch_size"], data["num_workers"]
    )
    with track_spikes(model) as rates:
        loss = mean_loss(model, loader, getattr(torch, cfg["train"]["dtype"]))
    logger.info("val loss %.4f ppl %.2f", loss, math.exp(loss))
    if rates:
        logger.info("spike rate %s", " ".join(f"{k} {v:.4f}" for k, v in rates.items()))


if __name__ == "__main__":
    main()
