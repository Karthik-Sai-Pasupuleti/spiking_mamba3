"""Spiking Mamba-3 SISO on WikiText-2: PLIF neurons before in_proj and out_proj, T steps per token.

This is the one file the autoresearch agent edits (see program.md). It defines the spiking
layer, trains it for the fixed 740-step WikiText-2 budget and prints a summary block with
the metric (`val_ppl`) and the firing rate (`spike_rate`, capped at RATE_CAP).

Design (baseline):
- `PLIFRate`: a SpikingJelly ParametricLIFNode in multi-step mode. Each token's activation
  is fed T times; the membrane starts at 0 for every token and integrates over the T steps.
  The output is the spike count / T, so it takes the values 0, 1/T, ..., 1.
- `SpikingMamba3`: the upstream Mamba3 layer with a PLIF in front of in_proj and out_proj.
  The projections keep their weight tensors and names, so the SSM and its kernels are the
  dense Mamba-3 ones; only their inputs are spikes.
- Fixed parts (model size, data, seq_len, batch, steps, evaluation) come from
  configs/mamba3-siso-61m-wikitext2.yaml and eval.py, which the agent must not change.

Run: .venv/bin/python spiking_mamba3.py > run.log 2>&1   (add --smoke for a 20-step check)
"""

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from torch import Tensor, nn

from configs.config import load_config
from dataset.dataloader import build_loader, build_train_sampler, build_val_loader, plan_training
from dataset.dataloader import micro_batches
from dataset.dataset import LMDataset, load_tokens
from eval import mean_loss
from models.backbone import SSM_LAYERS
from models.layers.mamba3 import Mamba3
from models.model import MambaConfig, MambaLMHeadModel
from train import build_optimizer, lr_at, non_finite_grads, set_seed
from utils.checkpoint import load_model_state

logger = logging.getLogger(__name__)

# ---- Knobs: the agent may change these (and any code in this file that uses them). ----
T = 4                          # PLIF steps per token (allowed: 1-8)
PLIF_INIT_TAU = 2.0            # initial membrane time constant; tau = 1 / sigmoid(w), w learnable
PLIF_V_THRESHOLD = 1.0
PEAK_LR = 5.0e-3               # 5x the teacher's 1e-3; 6e-3 with 5% warmup diverged
MIN_LR = 5.0e-4
WARMUP_FRACTION = 0.50
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
KD_WEIGHT = 0.5                # 0 = no distillation; loss = (1 - w) * CE + w * KD
KD_TEMPERATURE = 1.0
SURROGATE_ALPHA = 1.5          # ATan surrogate sharpness (SpikingJelly default 2.0)

# ---- Fixed: these define the comparison. Do not change. ----
BASE_CONFIG = Path("configs/mamba3-siso-61m-wikitext2.yaml")
TEACHER_CHECKPOINT = Path("runs/teacher-dense-61m-wikitext2/final.pt")
RATE_CAP = 0.20
TOTAL_STEPS = 740
OUTPUT_CHECKPOINT = Path("runs/autoresearch/latest.pt")


class PLIFRate(nn.Module):
    """PLIF neuron that sees each token T times and returns its spike count / T.

    Input and output have the same shape (..., features). The membrane is reset before every
    call, so tokens do not share neuron state; the SSM carries the memory along the sequence.
    While `track` is True, the layer adds up its spikes and neuron-steps for `firing_rate`.
    """

    def __init__(self, steps: int, init_tau: float, v_threshold: float) -> None:
        super().__init__()
        self.steps = steps
        self.neuron = neuron.ParametricLIFNode(
            init_tau=init_tau,
            v_threshold=v_threshold,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=SURROGATE_ALPHA),
            detach_reset=True,
            step_mode="m",
        )
        self.track = False
        self.spikes = 0.0
        self.neuron_steps = 0
        self.last_rate = 0.0

    def forward(self, x: Tensor) -> Tensor:
        """Return the spike count / T of each element of `x`, in the dtype of `x`."""
        flat = x.reshape(-1, x.shape[-1]).float()
        self.neuron.reset()
        spikes = self.neuron(flat.unsqueeze(0).expand(self.steps, *flat.shape))  # (T, N, D)
        rate = spikes.mean(dim=0)
        self.last_rate = float(rate.detach().mean())
        if self.track:
            self.spikes += float(spikes.detach().sum())
            self.neuron_steps += spikes.numel()
        return rate.reshape(x.shape).to(x.dtype)


class SpikingLinear(nn.Module):
    """A linear layer whose input first passes through a `PLIFRate` neuron.

    It takes over the `weight` (and `bias`) tensors of an existing nn.Linear, so parameter
    names and shapes stay those of the dense layer.
    """

    def __init__(self, linear: nn.Linear, plif: PLIFRate) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.plif = plif

    def forward(self, x: Tensor) -> Tensor:
        """Spike the input, then apply the linear map to the spike rates."""
        return F.linear(self.plif(x), self.weight, self.bias)


class SpikingMamba3(Mamba3):
    """Mamba-3 SISO layer with PLIF neurons before in_proj and before out_proj.

    Takes the arguments of Mamba3 plus `plif_steps`, `plif_init_tau` and `plif_v_threshold`.
    """

    def __init__(
        self,
        d_model: int,
        *args: Any,
        plif_steps: int = T,
        plif_init_tau: float = PLIF_INIT_TAU,
        plif_v_threshold: float = PLIF_V_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(d_model, *args, **kwargs)
        if self.is_mimo:
            raise ValueError("SpikingMamba3 supports SISO only")
        device = self.in_proj.weight.device
        for name in ("in_proj", "out_proj"):
            plif = PLIFRate(plif_steps, plif_init_tau, plif_v_threshold).to(device)
            setattr(self, name, SpikingLinear(getattr(self, name), plif))


SSM_LAYERS["SpikingMamba3"] = SpikingMamba3


def plif_layers(model: nn.Module) -> dict[str, PLIFRate]:
    """All PLIFRate modules of `model`, by name."""
    return {name: m for name, m in model.named_modules() if isinstance(m, PLIFRate)}


def build_config(smoke: bool) -> dict[str, Any]:
    """The fixed 61M WikiText-2 config with the spiking layer and the knobs above applied."""
    cfg = load_config(BASE_CONFIG)
    cfg["model"]["ssm_cfg"].update(
        layer="SpikingMamba3",
        plif_steps=T,
        plif_init_tau=PLIF_INIT_TAU,
        plif_v_threshold=PLIF_V_THRESHOLD,
    )
    cfg["optim"].update(
        peak_lr=PEAK_LR,
        min_lr=MIN_LR,
        warmup_fraction=WARMUP_FRACTION,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
    )
    if smoke:
        cfg["train"]["max_steps"] = 20
    return cfg


def load_teacher(cfg: dict[str, Any]) -> nn.Module:
    """The frozen dense Mamba-3 61M teacher, in eval mode."""
    teacher_model_cfg = load_config(BASE_CONFIG)["model"]
    teacher = MambaLMHeadModel(MambaConfig(**teacher_model_cfg), device="cuda")
    teacher.load_state_dict(load_model_state(TEACHER_CHECKPOINT))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def train_loss(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    dtype: torch.dtype,
    teacher: nn.Module | None,
) -> Tensor:
    """Next-token cross-entropy, mixed with logit distillation when a teacher is given."""
    with torch.autocast("cuda", dtype=dtype):
        logits = model(inputs)
    flat = logits.float().reshape(-1, logits.shape[-1])
    ce = F.cross_entropy(flat, targets.reshape(-1))
    if teacher is None:
        return ce
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        teacher_logits = teacher(inputs).float().reshape(-1, logits.shape[-1])
    t = KD_TEMPERATURE
    kd = F.kl_div(
        F.log_softmax(flat / t, dim=-1), F.log_softmax(teacher_logits / t, dim=-1),
        log_target=True, reduction="batchmean",
    ) * t * t
    return (1 - KD_WEIGHT) * ce + KD_WEIGHT * kd


def firing_rate_and_val_loss(
    model: nn.Module, val_loader: Any, dtype: torch.dtype
) -> tuple[float, float, dict[str, float]]:
    """Validation loss (fixed eval.mean_loss) and the firing rate measured in the same pass.

    The rate is total spikes / total neuron-steps over all PLIF neurons and all validation
    tokens. Also returns the rate of the in_proj and of the out_proj neurons separately.
    """
    layers = plif_layers(model)
    for layer in layers.values():
        layer.track, layer.spikes, layer.neuron_steps = True, 0.0, 0
    val_loss = mean_loss(model, val_loader, dtype)
    by_site: dict[str, list[float]] = {"in_proj": [0.0, 0.0], "out_proj": [0.0, 0.0]}
    for name, layer in layers.items():
        site = "in_proj" if ".in_proj." in name else "out_proj"
        by_site[site][0] += layer.spikes
        by_site[site][1] += layer.neuron_steps
        layer.track = False
    total_spikes = sum(v[0] for v in by_site.values())
    total_steps = sum(v[1] for v in by_site.values())
    rates = {site: v[0] / v[1] for site, v in by_site.items()}
    return val_loss, total_spikes / total_steps, rates


def main() -> None:
    """Train the spiking model for the fixed budget and print the summary block."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 steps, to check the code runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = time.perf_counter()

    cfg = build_config(args.smoke)
    data, tcfg, ocfg = cfg["data"], cfg["train"], cfg["optim"]
    set_seed(cfg["seed"])
    train_set = LMDataset(load_tokens(Path(data["train_bin"])), data["seq_len"])
    val_set = LMDataset(load_tokens(Path(data["val_bin"])), data["seq_len"])
    sampler = build_train_sampler(train_set, 0, 1, cfg["seed"])
    batch, micro = tcfg["batch_size"], tcfg["micro_batch_size"]
    plan = plan_training(
        sampler.num_samples, batch, tcfg["epochs"], tcfg["max_steps"], ocfg["warmup_fraction"]
    )
    if not args.smoke and plan.total_steps != TOTAL_STEPS:
        raise RuntimeError(f"budget changed: {plan.total_steps} steps, expected {TOTAL_STEPS}")
    loader = build_loader(train_set, micro, data["num_workers"], sampler)
    val_loader = build_val_loader(val_set, data["val_samples"], micro, data["num_workers"])
    batches = micro_batches(loader, sampler, plan, 0, batch)

    model = MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")
    optimizer = build_optimizer(model, ocfg)
    teacher = load_teacher(cfg) if KD_WEIGHT > 0 else None
    dtype = getattr(torch, tcfg["dtype"])
    accum = batch // micro
    logger.info("T=%d, %d steps, %d params, KD %.2f", T, plan.total_steps,
                sum(p.numel() for p in model.parameters()), KD_WEIGHT)

    model.train()
    train_start = time.perf_counter()
    for step in range(plan.total_steps):
        lr = lr_at(step, plan.total_steps, plan.warmup_steps, ocfg["peak_lr"], ocfg["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            inputs, targets = next(batches)
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            loss = train_loss(model, inputs, targets, dtype, teacher) / accum
            loss.backward()
            step_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step + 1}: {non_finite_grads(model)[:10]}")
        optimizer.step()
        if (step + 1) % 20 == 0 or step == 0:
            rates = [m.last_rate for m in plif_layers(model).values()]
            logger.info("step %d loss %.4f lr %.2e gnorm %.2f rate %.4f", step + 1, step_loss,
                        lr, grad_norm.item(), sum(rates) / len(rates))
    training_seconds = time.perf_counter() - train_start

    val_loss, rate, site_rates = firing_rate_and_val_loss(model, val_loader, dtype)
    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": plan.total_steps}, OUTPUT_CHECKPOINT)
    print("---")
    print(f"val_ppl:            {math.exp(val_loss):.6f}")
    print(f"val_loss:           {val_loss:.6f}")
    print(f"spike_rate:         {rate:.6f}")
    print(f"rate_ok:            {'yes' if rate <= RATE_CAP else 'no'}")
    print(f"rate_in_proj:       {site_rates['in_proj']:.6f}")
    print(f"rate_out_proj:      {site_rates['out_proj']:.6f}")
    print(f"T:                  {T}")
    print(f"num_steps:          {plan.total_steps}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"total_seconds:      {time.perf_counter() - start:.1f}")
    print(f"peak_vram_mb:       {torch.cuda.max_memory_allocated() / 2**20:.1f}")


if __name__ == "__main__":
    main()
