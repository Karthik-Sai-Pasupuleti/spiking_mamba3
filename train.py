"""Pretraining: learning-rate schedule, optimizer, train loop, and the `train` command.

Run it with `python train.py --config configs/<name>.yaml [--smoke]
[--run-dir runs/<name>]`, or with `torchrun --standalone --nproc_per_node=N -m
train ...` for several GPUs. With `--run-dir` the run uses one fixed folder
and resumes from its `last.pt`.

The loop does no file I/O. It gets its micro-batches from an iterator and reports metrics
and checkpoints through functions it gets as arguments; `main` builds them from `dataset`
(FlashAttention's `LMDataset` and fault-tolerant sampler) and `utils`.

Every step checks the gradient norm. A non-finite gradient stops the run before
`optimizer.step()` can write NaN into the weights, and logs which parameters have it.
"""

import argparse
import logging
import math
import random
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from configs.config import load_config
from dataset.dataloader import (
    TrainPlan,
    build_loader,
    build_train_sampler,
    build_val_loader,
    micro_batches,
    plan_training,
)
from dataset.dataset import LMDataset, load_tokens
from eval import mean_loss, next_token_loss
from models.model import MambaLMHeadModel, MambaConfig
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.distributed import close_distributed, init_distributed
from utils.run_files import (
    MetricsCSV,
    create_run_dir,
    environment_info,
    setup_logging,
    write_config,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(
    step: int, total_steps: int, warmup_steps: int, peak: float, floor: float
) -> float:
    """Linear warmup to `peak`, then cosine decay down to `floor`."""
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, optim_cfg: dict[str, Any]) -> torch.optim.AdamW:
    """Fused AdamW with weight decay on 2-D weight matrices only.

    Norm weights, biases, `D`, `dt_bias` and all parameters marked `_no_weight_decay`
    get no decay. `B_bias` and `C_bias` are 3-D, so they get no decay either.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not getattr(param, "_no_weight_decay", False):
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": optim_cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=optim_cfg["peak_lr"], betas=tuple(optim_cfg["betas"]), fused=True
    )


def non_finite_grads(model: nn.Module) -> list[str]:
    """Names of the parameters whose gradient has a NaN or inf value."""
    return [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]


def pretrain(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    plan: TrainPlan,
    train_batches: Iterator[tuple[Tensor, Tensor]],
    val_loader: DataLoader,
    cfg: dict[str, Any],
    log_metrics: Callable[[dict[str, float]], None],
    save: Callable[[str, dict[str, Any]], None],
    start_step: int = 0,
    inspect: Callable[[nn.Module, int], None] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Train from `start_step` to `plan.total_steps`, with the config sections `data`,
    `optim` and `train`.

    Every `log_every` steps it calls `log_metrics`, with validation every `val_every`
    steps. If `inspect` is given, it is called on the same steps as `inspect(model, step)`
    right after the backward pass, so the gradients are still raw (not yet clipped).
    Every `save_every` steps it calls `save("last.pt", payload)` with model, optimizer and
    step, and at the end `save("final.pt", payload)`. Raises RuntimeError on the first
    step with a non-finite gradient norm, before the optimizer step.

    `train_batches` yields this GPU's micro-batches from `start_step` on (see
    `dataset.dataloader.micro_batches`); each step takes `batch_size // (micro_batch_size x
    world_size)` of them. `val_loader` gives the validation micro-batches.

    Multi-GPU: pass the DDP-wrapped model and this process's `rank` and `world_size`.
    Each rank trains on its share of every step's samples; DDP averages the gradients
    once per step. Only rank 0 validates, inspects, logs and saves. With
    `world_size == 1` and a plain model, this is ordinary single-GPU training.
    """
    seq_len = cfg["data"]["seq_len"]
    ocfg, tcfg = cfg["optim"], cfg["train"]
    batch_size, micro = tcfg["batch_size"], tcfg["micro_batch_size"]
    if batch_size % (micro * world_size) != 0:
        raise ValueError(
            f"batch_size {batch_size} must be a multiple of "
            f"micro_batch_size x world_size = {micro * world_size}"
        )
    accum = batch_size // (micro * world_size)
    dtype = getattr(torch, tcfg["dtype"])
    tokens_per_step = batch_size * seq_len
    is_ddp = isinstance(model, DistributedDataParallel)
    core = model.module if is_ddp else model  # the plain model, for eval and saving

    model.train()
    window_start = perf_counter()
    for step in range(start_step, plan.total_steps):
        lr = lr_at(
            step, plan.total_steps, plan.warmup_steps, ocfg["peak_lr"], ocfg["min_lr"]
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device="cuda")
        for i in range(accum):
            inputs, targets = next(train_batches)
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            # Average gradients across GPUs only after the last micro-batch.
            sync = not is_ddp or i == accum - 1
            with nullcontext() if sync else model.no_sync():
                loss = next_token_loss(model, inputs, targets, dtype) / accum
                loss.backward()
            step_loss += loss.detach()
        done = step + 1
        log_step = done % tcfg["log_every"] == 0 or done == plan.total_steps
        if inspect is not None and log_step and rank == 0:
            inspect(core, done)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
        if not torch.isfinite(grad_norm):
            bad = non_finite_grads(core)
            raise RuntimeError(
                f"non-finite gradient norm at step {step + 1} "
                f"(loss {step_loss.item():.4f}); "
                f"{len(bad)} parameters with NaN/inf gradients: {bad[:10]}"
            )
        optimizer.step()

        if log_step and world_size > 1:
            dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)  # all ranks, same steps
        if log_step and rank == 0:
            loss_value = step_loss.item()
            elapsed = perf_counter() - window_start
            tokens_per_s = tcfg["log_every"] * tokens_per_step / elapsed
            val_loss = float("nan")
            if done % tcfg["val_every"] == 0 or done == plan.total_steps:
                val_loss = mean_loss(core, val_loader, dtype)
            log_metrics(
                {
                    "step": done,
                    "tokens_seen": done * tokens_per_step,
                    "train_loss": loss_value,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(val_loss),
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                    "tokens_per_s": tokens_per_s,
                }
            )
            logger.info(
                "step %d/%d loss %.4f val %.4f lr %.2e gnorm %.2f %.0f tok/s",
                done, plan.total_steps, loss_value, val_loss, lr, grad_norm.item(),
                tokens_per_s,
            )
            window_start = perf_counter()
        if done % tcfg["save_every"] == 0 and done < plan.total_steps and rank == 0:
            payload = {
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": done,
            }
            save("last.pt", payload)

    if rank == 0:
        save("final.pt", {"model": core.state_dict(), "step": plan.total_steps})


def main() -> None:
    """Parse the arguments, build model, data and run folder, and call `pretrain`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="20-step run of a tiny model")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    info = init_distributed()
    if info.world_size > 1 and args.run_dir is None:
        raise SystemExit("several GPUs need --run-dir, so all processes share one folder")
    try:
        run_dir = args.run_dir
        if run_dir is None:
            run_dir = create_run_dir(Path("runs"), cfg["run_name"])
        elif info.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
        if info.is_main:
            setup_logging(run_dir / "train.log")

        set_seed(cfg["seed"])
        data = cfg["data"]
        tcfg, ocfg = cfg["train"], cfg["optim"]
        train = LMDataset(load_tokens(Path(data["train_bin"])), data["seq_len"])
        val = LMDataset(load_tokens(Path(data["val_bin"])), data["seq_len"])
        sampler = build_train_sampler(train, info.rank, info.world_size, cfg["seed"])
        per_rank_step = tcfg["batch_size"] // info.world_size
        plan = plan_training(
            sampler.num_samples, per_rank_step, tcfg["epochs"],
            tcfg["max_steps"], ocfg["warmup_fraction"],
        )
        micro, workers = tcfg["micro_batch_size"], data["num_workers"]
        train_loader = build_loader(train, micro, workers, sampler)
        val_loader = build_val_loader(val, data["val_samples"], micro, workers)

        model = MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")
        optimizer = build_optimizer(model, ocfg)
        start_step = 0
        last = run_dir / "last.pt"
        if last.exists():
            state = load_checkpoint(last)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"]
        if info.is_main:
            write_config(
                run_dir / "config.json",
                {
                    "config": cfg,
                    "environment": environment_info(),
                    "world_size": info.world_size,
                    "train_samples": len(train),
                    "steps_per_epoch": plan.steps_per_epoch,
                    "total_steps": plan.total_steps,
                    "start_step": start_step,
                    "n_params": sum(p.numel() for p in model.parameters()),
                },
            )
            logger.info("%s: %d steps, starting at %d", cfg["run_name"], plan.total_steps, start_step)
        wrapped: nn.Module = model
        if info.world_size > 1:
            wrapped = DistributedDataParallel(model, device_ids=[info.local_rank])

        metrics = MetricsCSV(run_dir / "metrics.csv") if info.is_main else None
        train_batches = micro_batches(train_loader, sampler, plan, start_step, per_rank_step)
        pretrain(
            wrapped, optimizer, plan, train_batches, val_loader, cfg,
            log_metrics=metrics.write if metrics else (lambda row: None),
            save=lambda name, payload: save_checkpoint(run_dir / name, payload),
            start_step=start_step,
            rank=info.rank,
            world_size=info.world_size,
        )
        if metrics:
            metrics.close()
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
