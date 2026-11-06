"""Mamba-3 language model: backbone plus a (tied) output head.

Ports upstream `MambaLMHeadModel` without generation. Only SISO (Triton kernels) is
supported: the TileLang MIMO kernels are not copied, so `ssm_cfg["is_mimo"] = True` fails
with an assert. Parameter names match upstream, so checkpoints such as
`weights/mamba3-siso-187m` load with `load_state_dict`. Training lives in
`train`.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from models.backbone import SSM_LAYERS, Device, MixerModel, _init_weights


@dataclass
class MambaConfig:
    """Model size and options. Port of upstream `MambaConfig`, same field names and defaults.

    The attention fields of upstream are left out. `ssm_cfg` holds the keyword arguments
    of the mixer (d_state, headdim, chunk_size, ...). Its `"layer"` key picks the mixer
    from `SSM_LAYERS`; only "Mamba3" (the default) exists.
    """

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict[str, Any] = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class MambaLMHeadModel(nn.Module):
    """Mamba-3 language model. Port of upstream `MambaLMHeadModel`, without generation.

    Build it with a `MambaConfig` and call it on token ids (batch, seqlen). It returns
    logits (batch, seqlen, padded vocab size). The vocabulary is padded up to a multiple
    of `pad_vocab_size_multiple`.
    """

    def __init__(
        self,
        config: MambaConfig,
        device: Device = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        ssm_cfg = dict(config.ssm_cfg)
        layer = ssm_cfg.get("layer", "Mamba3")
        if layer not in SSM_LAYERS:
            raise ValueError(
                f"unknown ssm layer {layer!r}; choose from {list(SSM_LAYERS)}"
            )
        vocab_size = config.vocab_size
        multiple = config.pad_vocab_size_multiple
        if vocab_size % multiple != 0:
            vocab_size += multiple - (vocab_size % multiple)
        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=config.rms_norm,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            config.d_model, vocab_size, bias=False, device=device, dtype=dtype
        )
        # Upstream applies the init a second time here, with n_residuals_per_layer=1.
        # This replaces the backbone's 1/sqrt(2 * n_layer) scale with 1/sqrt(n_layer).
        # Kept as is, so the same seed gives the same weights as upstream.
        self.apply(partial(_init_weights, n_layer=config.n_layer))
        if config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return next-token logits for every position."""
        return self.lm_head(self.backbone(input_ids))


if __name__ == "__main__":
    # Smoke check: tiny SISO model, one forward pass on the GPU (Triton needs CUDA).
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=128,
        d_intermediate=256,
        n_layer=2,
        vocab_size=1000,
        ssm_cfg={"d_state": 64, "headdim": 64, "chunk_size": 64},
        pad_vocab_size_multiple=16,
    )
    model = MambaLMHeadModel(cfg, device="cuda")
    ids = torch.randint(0, cfg.vocab_size, (2, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    n_params = sum(p.numel() for p in model.parameters())
    print(tuple(logits.shape), f"{n_params:,} params", bool(torch.isfinite(logits).all()))
