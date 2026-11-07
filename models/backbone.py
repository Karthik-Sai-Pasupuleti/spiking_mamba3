"""Mamba-3 backbone: token embedding, a stack of blocks, final norm.

Ports upstream `mamba_ssm/models/mixer_seq_simple.py` (`create_block`, `_init_weights`,
`MixerModel`) at commit e9594ce, for the Mamba-3 mixers only. Module order, parameter names
and init calls are the same as upstream, so upstream checkpoints load and the same seed gives
the same initial weights. `models.model` puts the output head on top.
"""

import math
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from models.kernels.triton.layer_norm import RMSNorm, layer_norm_fn
from models.layers.block import Block
from models.layers.mamba3 import Mamba3
from models.layers.mlp import GatedMLP
from models.layers.spiking import SpikingMamba3

Device = torch.device | str | None

# Mixer classes by the upstream `ssm_cfg["layer"]` name.
SSM_LAYERS: dict[str, type[Mamba3]] = {"Mamba3": Mamba3, "SpikingMamba3": SpikingMamba3}


def create_block(
    d_model: int,
    d_intermediate: int,
    ssm_cfg: dict[str, Any],
    norm_epsilon: float,
    rms_norm: bool,
    residual_in_fp32: bool,
    fused_add_norm: bool,
    layer_idx: int,
    device: Device = None,
    dtype: torch.dtype | None = None,
) -> Block:
    """Build one block: norm, mixer, then norm and MLP if `d_intermediate > 0`.

    The mixer class is `SSM_LAYERS[ssm_cfg["layer"]]` (default "Mamba3").
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = dict(ssm_cfg)
    mixer_type = SSM_LAYERS[ssm_cfg.pop("layer", "Mamba3")]
    mixer_cls = partial(mixer_type, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        RMSNorm if rms_norm else nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls: Any = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
) -> None:
    """Upstream GPT-2 style init: zero biases, normal embeddings, scaled out projections.

    `out_proj.weight` and `fc2.weight` are re-drawn and divided by
    sqrt(n_residuals_per_layer * n_layer), so the residual stream does not grow with
    depth.
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    """Embedding, `n_layer` blocks and final norm. Port of upstream `MixerModel`.

    Takes token ids (batch, seqlen) and returns hidden states (batch, seqlen, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg: dict[str, Any],
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run all blocks. Each adds its input to the residual stream, then norms."""
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if not self.fused_add_norm:
            if residual is not None:
                residual = hidden_states + residual
            else:
                residual = hidden_states
            return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=isinstance(self.norm_f, RMSNorm),
        )
