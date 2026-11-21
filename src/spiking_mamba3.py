"""Spiking Mamba-3 SISO layer: PLIF neurons before in_proj and out_proj, T steps per token.

Importing this module registers the layer as "SpikingMamba3" in `models.backbone.SSM_LAYERS`,
so a config with `ssm_cfg.layer: SpikingMamba3` builds it. `src/train.py` and `src/eval.py`
import it for that reason; training and evaluation live there, not here.

Design:
- `PLIFRate`: a SpikingJelly ParametricLIFNode in multi-step mode. Each token's activation
  is fed T times; the membrane starts at 0 for every token and integrates over the T steps.
  The output is the spike count / T. With a constant input and a hard reset, the count can
  only be T // k (k = steps to the first spike), so T = 4 gives the values 0, 1/4, 1/2, 1.
- `SpikingMamba3`: the upstream Mamba3 layer with a PLIF in front of in_proj and out_proj.
  The projections keep their weight tensors and names, so the SSM and its kernels are the
  dense Mamba-3 ones; only their inputs are spikes.

Run the smoke check from the repo root: .venv/bin/python src/spiking_mamba3.py [--config ...]
"""

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from torch import Tensor, nn

from configs.config import load_config
from models.backbone import SSM_LAYERS
from models.layers.mamba3 import Mamba3
from models.model import MambaConfig, MambaLMHeadModel

# Defaults of the neuron; a config overrides them through ssm_cfg.
T = 4                          # PLIF steps per token
PLIF_INIT_TAU = 2.0            # initial membrane time constant; tau = 1 / sigmoid(w), w learnable
PLIF_V_THRESHOLD = 1.0

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "mamba3-siso-61m-wikitext103-spiking.yaml"
)


class PLIFRate(nn.Module):
    """PLIF neuron that sees each token T times and returns its spike count / T.

    Input and output have the same shape (..., features). The membrane is reset before every
    call, so tokens do not share neuron state; the SSM carries the memory along the sequence.
    `last_rate` holds the mean output of the last call. While `track` is True, the layer
    also adds up its spikes and neuron-steps, to measure a firing rate over many calls.
    """

    def __init__(self, steps: int, init_tau: float, v_threshold: float) -> None:
        super().__init__()
        self.steps = steps
        self.neuron = neuron.ParametricLIFNode(
            init_tau=init_tau,
            v_threshold=v_threshold,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
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


def start_spike_count(model: nn.Module) -> None:
    """Reset the spike counters of all PLIF neurons of `model` and start counting."""
    for layer in plif_layers(model).values():
        layer.track, layer.spikes, layer.neuron_steps = True, 0.0, 0


def spike_rates(model: nn.Module) -> dict[str, float]:
    """Stop counting and return spikes / neuron-steps since `start_spike_count`.

    Keys: "spike_rate" (all neurons), "spike_rate_in_proj" and "spike_rate_out_proj".
    Empty for a model without PLIF neurons (a dense model).
    """
    counts = {"in_proj": [0.0, 0], "out_proj": [0.0, 0]}
    for name, layer in plif_layers(model).items():
        site = "in_proj" if ".in_proj." in name else "out_proj"
        counts[site][0] += layer.spikes
        counts[site][1] += layer.neuron_steps
        layer.track = False
    total_steps = counts["in_proj"][1] + counts["out_proj"][1]
    if total_steps == 0:
        return {}
    rates = {"spike_rate": (counts["in_proj"][0] + counts["out_proj"][0]) / total_steps}
    for site, (spikes, steps) in counts.items():
        rates[f"spike_rate_{site}"] = spikes / steps
    return rates


def main() -> MambaLMHeadModel:
    """Parse `--config` and build the model it describes on the GPU, with random weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    return MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")


if __name__ == "__main__":
    # Smoke check: random tokens of shape (2, 2048) through the model; print the shapes,
    # the next-token loss (about ln(vocab) for random weights) and the spike rates.
    torch.manual_seed(0)
    lm = main().eval()
    vocab = lm.config.vocab_size
    tokens = torch.randint(0, vocab, (2, 2049), device="cuda")
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    start_spike_count(lm)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = lm(inputs)
    measured = spike_rates(lm)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    n_params = sum(p.numel() for p in lm.parameters())
    print(f"params: {n_params:,}, PLIF neurons: {len(plif_layers(lm))}")
    print(f"inputs {tuple(inputs.shape)} -> logits {tuple(logits.shape)}")
    ln_vocab = torch.log(torch.tensor(float(vocab))).item()
    print(f"loss {loss.item():.3f} (ln vocab = {ln_vocab:.3f})")
    for key, value in measured.items():
        print(f"{key}: {value:.4f}")
