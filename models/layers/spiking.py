"""Spiking Mamba-3 SISO layer: PLIF neurons before in_proj and out_proj, T steps per token.

The neuron design of the autoresearch run `autoresearch/nov6` (best commit ab9fa2c), moved here
so that `train.py` can build it from a config (`ssm_cfg.layer: SpikingMamba3`) and train it on
several GPUs. `spiking_mamba3.py` keeps its own copy for the autoresearch loop.

- `PLIFRate`: SpikingJelly `ParametricLIFNode` in multi-step mode. Each token's activation is
  fed T times; the membrane starts at 0 for every token. The output is spike count / T.
- `SpikingLinear`: a PLIFRate in front of an existing nn.Linear (same weight tensors and
  names), with an optional learnable per-channel gain on the neuron input.
- `SpikingMamba3`: upstream `Mamba3` with a SpikingLinear for in_proj and for out_proj. The
  SSM and its Triton kernels are untouched, so a dense Mamba-3 checkpoint loads into it; only
  the neuron parameters (`*.plif.neuron.w`, `*.gain`) are new.

`spike_rates` and `track_spikes` measure firing rates for logging and validation.
"""

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from torch import Tensor, nn

from models.layers.mamba3 import Mamba3


class PLIFRate(nn.Module):
    """PLIF neuron that sees each token T times and returns its spike count / T.

    Input and output have the same shape (..., features). The membrane is reset before every
    call. While `track` is True, the layer adds up its spikes and neuron-steps.
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

    It takes over the `weight` (and `bias`) of an existing nn.Linear. With `gain_init` set,
    the neuron input is first multiplied by a learnable per-channel gain (initial value
    `gain_init`), which moves the channels onto the threshold.
    """

    def __init__(self, linear: nn.Linear, plif: PLIFRate, gain_init: float | None = None) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.plif = plif
        self.gain = None
        if gain_init is not None:
            self.gain = nn.Parameter(
                torch.full((linear.in_features,), gain_init, device=linear.weight.device)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Spike the (optionally scaled) input, then apply the linear map to the spike rates."""
        if self.gain is not None:
            x = x * self.gain.to(x.dtype)
        return F.linear(self.plif(x), self.weight, self.bias)


class SpikingMamba3(Mamba3):
    """Mamba-3 SISO layer with PLIF neurons before in_proj and before out_proj.

    Takes the arguments of Mamba3 plus the neuron settings `plif_steps` (T), `plif_init_tau`,
    `plif_v_threshold`, and `out_proj_gain_init` (None = no gain). Defaults are the nov6 best:
    T=4, tau 2.0, threshold 1.0, out_proj gain 1.5. out_proj gets the gain because its input,
    the gated SSM output, has no norm in front; in_proj's input is RMSNorm output.
    """

    def __init__(
        self,
        d_model: int,
        *args: Any,
        plif_steps: int = 4,
        plif_init_tau: float = 2.0,
        plif_v_threshold: float = 1.0,
        out_proj_gain_init: float | None = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(d_model, *args, **kwargs)
        if self.is_mimo:
            raise ValueError("SpikingMamba3 supports SISO only")
        device = self.in_proj.weight.device
        for name, gain_init in (("in_proj", None), ("out_proj", out_proj_gain_init)):
            plif = PLIFRate(plif_steps, plif_init_tau, plif_v_threshold).to(device)
            setattr(self, name, SpikingLinear(getattr(self, name), plif, gain_init))


def plif_layers(model: nn.Module) -> dict[str, PLIFRate]:
    """All PLIFRate modules of `model`, by name (empty for a dense model)."""
    return {name: m for name, m in model.named_modules() if isinstance(m, PLIFRate)}


def _site(name: str) -> str:
    """"in_proj" or "out_proj": which projection the PLIF module `name` feeds."""
    return "in_proj" if "in_proj" in name.split(".") else "out_proj"


def spike_rates(model: nn.Module) -> dict[str, float]:
    """Mean firing rate of the last forward pass, for the in_proj and the out_proj neurons."""
    sites: dict[str, list[float]] = {"in_proj": [], "out_proj": []}
    for name, layer in plif_layers(model).items():
        sites[_site(name)].append(layer.last_rate)
    return {site: sum(v) / len(v) for site, v in sites.items() if v}


@contextmanager
def track_spikes(model: nn.Module) -> Iterator[dict[str, float]]:
    """Count spikes during the `with` block; afterwards the yielded dict holds the rates.

    The rate is total spikes / total neuron-steps (all neurons, all tokens), overall
    ("total") and per site. The dict stays empty for a dense model.
    """
    layers = plif_layers(model)
    for layer in layers.values():
        layer.track, layer.spikes, layer.neuron_steps = True, 0.0, 0
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        counts: dict[str, list[float]] = {}
        for name, layer in layers.items():
            for key in (_site(name), "total"):
                counts.setdefault(key, [0.0, 0.0])
                counts[key][0] += layer.spikes
                counts[key][1] += layer.neuron_steps
            layer.track = False
        result.update({k: s / n for k, (s, n) in counts.items() if n})


if __name__ == "__main__":
    # Smoke check: one small spiking layer on the GPU, forward + backward, firing rates.
    torch.manual_seed(0)
    layer = SpikingMamba3(d_model=128, d_state=64, headdim=64, chunk_size=64, device="cuda")
    u = torch.randn(2, 64, 128, device="cuda", requires_grad=True)
    with track_spikes(layer) as rates, torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(u)
    y.float().sum().backward()
    print(tuple(y.shape), "finite", bool(torch.isfinite(y).all()), "rates", rates,
          "grad gain", layer.out_proj.gain.grad is not None)
