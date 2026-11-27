"""Every way this study computes the spikes of a soft-reset PLIF neuron over the token axis.

All methods share one interface, `spikes, v_last = method(x, tau, theta, v0=None, **kwargs)`,
with `x` and `spikes` of shape [L, B, D], and are listed in `METHODS`. `bench_neuron.py`
imports this module, runs one method on the fixed inputs and compares it with the exact
serial reference (`sj_reference`, SpikingJelly's fused kernels). This file only defines
the methods; timing, the fidelity gate and the result rows live in `bench_neuron.py`.

The neuron (see METHODS.md, "Reference neuron"), with q = 1/tau = sigmoid(w), w = -log(tau - 1):

    h_t = v_{t-1} + (x_t - v_{t-1}) * q      if decay_input      (the repo's setting)
    h_t = v_{t-1} - v_{t-1} * q + x_t        if not decay_input  (used by the counting input)
    s_t = 1 if h_t >= theta else 0           backward: ATan surrogate, alpha = 2
    v_t = h_t - theta * s_t                  soft reset, the spike is detached in the reset
"""

import linecache
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from spikingjelly._ops.plif import triton as sj_triton
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import surrogate as sj_surrogate
from torch import Tensor

logger = logging.getLogger("neuron")

# A method may leave one short remark about its last call here (e.g. a measured number of
# rounds); `bench_neuron.py` appends it to the notes column and clears it.
LAST_NOTE: list[str] = []

ALPHA = 2.0        # ATan surrogate width, the SpikingJelly default used by the repo
ATAN_ID = 1        # SpikingJelly's id of the ATan surrogate in its fused kernels
SJ_BLOCK = 256     # neurons per program in SpikingJelly's Triton kernels


def _per_channel_leak(**_config: object) -> bool:
    """Default of `Method.scalar_leak`: the method takes a leak per channel."""
    return False


@dataclass
class Method:
    """One registered method.

    `seq_depth(L, **config)` is the number of sequential steps it needs.
    `scalar_leak(**config)` is True when it can only take one leak for the whole layer.
    """

    fn: Callable[..., tuple[Tensor, ...]]
    seq_depth: Callable[..., int]
    scalar_leak: Callable[..., bool] = _per_channel_leak


METHODS: dict[str, Method] = {}


def register(
    name: str, seq_depth: Callable[..., int], scalar_leak: Callable[..., bool] = _per_channel_leak
) -> Callable:
    """Decorator: add a method to `METHODS` under `name`."""

    def wrap(fn: Callable[..., tuple[Tensor, ...]]) -> Callable[..., tuple[Tensor, ...]]:
        METHODS[name] = Method(fn, seq_depth, scalar_leak)
        return fn

    return wrap


def leak_w(tau: Tensor) -> Tensor:
    """The repo's leak parameter w for a time constant tau: 1/tau = sigmoid(w)."""
    return -torch.log(tau - 1.0)


def leak_q(tau: Tensor) -> Tensor:
    """q = 1/tau, computed as sigmoid(w) like SpikingJelly does, so all methods share its bits.

    tau = inf gives q = 0 exactly (no leak, used by the counting input).
    """
    return torch.sigmoid(leak_w(tau))


def charge(v: Tensor, x: Tensor, q: Tensor, decay_input: bool) -> Tensor:
    """One charge step h_t from v_{t-1} and x_t, in SpikingJelly's order of operations."""
    if decay_input:
        return v + (x - v) * q
    return v - v * q + x


class _FirstOfUniform(torch.autograd.Function):
    """Return element 0 of a vector whose elements are all equal; spread the gradient evenly.

    SpikingJelly's PLIF takes one scalar leak per layer. This turns the per-channel
    vector into that scalar without changing its bits (a mean could round).
    """

    @staticmethod
    def forward(ctx, w: Tensor) -> Tensor:  # type: ignore[override]
        ctx.n = w.numel()
        return w.reshape(-1)[0].clone()

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:  # type: ignore[override]
        return (grad / ctx.n).expand(ctx.n)


def _uniform_scalar(w: Tensor) -> Tensor:
    """The single value of a uniform vector `w`; raises if the elements differ."""
    if not bool((w == w.reshape(-1)[0]).all()):
        raise ValueError("SpikingJelly's PLIF has one scalar leak per layer; tau must be uniform")
    return _FirstOfUniform.apply(w)


# --------------------------------------------------------------------------------------
# The exact reference: SpikingJelly's fused Triton PLIF kernels
# --------------------------------------------------------------------------------------


def sj_reference(
    x: Tensor,
    tau: Tensor,
    theta: float,
    r: Tensor | None = None,
    v0: Tensor | None = None,
    decay_input: bool = True,
    hard: bool = False,
    detach: bool = True,
) -> dict[str, Tensor]:
    """Exact serial result from SpikingJelly's fused Triton kernels, launched as it launches them.

    Returns `spikes`, `v_seq` (membrane after reset, [L, B, D]), `v_last` and `h`. With `r`
    (the gradient of the loss w.r.t. the spikes) it also returns `gx` and `gtau`, the
    gradients w.r.t. `x` and w.r.t. a per-channel `tau`. SpikingJelly sums its per-neuron
    leak gradient over all neurons because its leak is one scalar; here the same kernel
    output is summed over the batch only, which gives the per-channel gradient.
    """
    w = _uniform_scalar(leak_w(tau.detach()))
    v_init = zero_state(x) if v0 is None else v0
    reset = 0.0 if hard else None
    spikes, v_seq, h = sj_triton._forward_impl(
        x.detach(), v_init, w, decay_input, theta, reset, detach, ALPHA, True, ATAN_ID
    )
    out = {"spikes": spikes, "v_seq": v_seq, "v_last": v_seq[-1], "h": h}
    if r is None:
        return out
    q = w.float().sigmoid()
    gx = torch.empty_like(h, dtype=r.dtype)
    gv_init = torch.empty_like(v_init)
    gq = torch.empty_like(v_init)
    neurons = v_init.numel()
    sj_triton._backward_kernel[(triton.cdiv(neurons, SJ_BLOCK),)](
        r, torch.zeros_like(v_init), x.detach(), v_init, h, gx, gv_init, gq,
        x.shape[0], neurons, q, theta, 0.0, ALPHA,
        decay_input, not hard, detach, ATAN_ID, False, SJ_BLOCK, None,
        enable_fp_fusion=False,
    )  # fmt: skip
    q_ch = leak_q(tau.detach())
    out["gx"] = gx
    out["gtau"] = -gq.sum(dim=0) * q_ch * q_ch  # dq/dtau = -q^2
    return out


# --------------------------------------------------------------------------------------
# A. Serial methods
# --------------------------------------------------------------------------------------

_SJ_NODES: dict[tuple, sj_neuron.ParametricLIFNode] = {}


def _sj_node(
    theta: float, decay_input: bool, store_v_seq: bool, hard: bool = False, detach: bool = True
) -> sj_neuron.ParametricLIFNode:
    """The repo's PLIF module (soft reset unless `hard`), cached per setting."""
    key = (theta, decay_input, store_v_seq, hard, detach)
    if key not in _SJ_NODES:
        _SJ_NODES[key] = sj_neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=decay_input,
            v_threshold=theta,
            v_reset=0.0 if hard else None,
            surrogate_function=sj_surrogate.ATan(),
            detach_reset=detach,
            step_mode="m",
            store_v_seq=store_v_seq,
        ).cuda()
    return _SJ_NODES[key]


def _sj_module(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor | None, decay_input: bool, return_v_seq: bool,
    hard: bool = False, detach: bool = True,
) -> tuple[Tensor, ...]:  # fmt: skip
    """Call SpikingJelly's ParametricLIFNode in multi-step mode, as the repo does."""
    node = _sj_node(float(theta), decay_input, return_v_seq, hard, detach)
    node.reset()
    if v0 is not None:
        node.v = v0
    w = _uniform_scalar(leak_w(tau))
    spikes = torch.func.functional_call(node, {"w": w}, (x,))
    if return_v_seq:
        return spikes, node.v, node.v_seq
    return spikes, node.v


@register("serial_fused", lambda L, **_: L, scalar_leak=lambda **_: True)
def serial_fused(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    hard: int = 0,
    detach: int = 1,
) -> tuple[Tensor, ...]:
    """The reference: SpikingJelly's multi-step PLIF, one fused kernel that loops over L.

    SpikingJelly 2.0 has no cupy backend; its fused kernel is the Triton provider, which
    it selects by itself on this machine. The leak is one scalar per layer.
    """
    return _sj_module(x, tau, theta, v0, decay_input, return_v_seq, bool(hard), bool(detach))


@register("serial_python", lambda L, **_: L, scalar_leak=lambda impl="sj", **_: impl == "sj")
def serial_python(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    impl: str = "sj",
) -> tuple[Tensor, ...]:
    """The same neuron as a Python loop over L.

    `impl="sj"`: SpikingJelly's Torch provider (needs SJ_PLIF_CUDA_IMPLEMENTATION=torch
    in the environment before spikingjelly is imported; `bench_neuron.py` sets it).
    `impl="own"`: a plain PyTorch loop with autograd and a per-channel leak.
    """
    if impl == "sj":
        return _sj_module(x, tau, theta, v0, decay_input, return_v_seq)
    q = leak_q(tau)
    spike_fn = sj_surrogate.ATan(alpha=ALPHA)
    v = zero_state(x) if v0 is None else v0
    spikes, v_seq = [], []
    for x_t in x:
        h = charge(v, x_t, q, decay_input)
        s = spike_fn(h - theta)
        v = h - theta * s.detach()
        spikes.append(s)
        if return_v_seq:
            v_seq.append(v)
    if return_v_seq:
        return torch.stack(spikes), v, torch.stack(v_seq)
    return torch.stack(spikes), v


# --------------------------------------------------------------------------------------
# Own fused Triton kernels (used by serial_triton and by every chunked method)
# --------------------------------------------------------------------------------------
#
# One forward and one backward kernel, generated as source text for an unroll factor U.
# Triton has no arrays of tensors, so "load U time steps, then compute U steps, then store
# U steps" has to be written out as U named variables; the generator does that.
# A program handles BLOCK neurons of one chunk of C time steps. The grid is
# (neuron blocks, chunks). With one chunk of C = L steps this is the plain serial loop.

TAPE_NONE, TAPE_H32, TAPE_H16, TAPE_U8 = 0, 1, 2, 3
_TAPE_DTYPE = {TAPE_H32: torch.float32, TAPE_H16: torch.float16, TAPE_U8: torch.int32}
_KERNELS: dict[tuple[str, int], object] = {}

# The 8-bit tape stores h - theta as a small float: 1 sign bit, 4 exponent bits
# (2^-9 .. 2^6) and 3 mantissa bits, i.e. about 4 % relative precision. Four time steps
# are packed into one int32, so the tape is [L/4, B, D] int32.
_U8_EXP_SHIFT = 118  # float32 exponent field of 2^-9


def _jit(name: str, source: str) -> object:
    """Compile generated Triton source. `linecache` lets Triton read the source back."""
    filename = f"<generated {name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"triton": triton, "tl": tl}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - our own generated kernel text
    return namespace[name]


def _step_lines(indent: str, j: int, track: bool) -> list[str]:
    """Source of one neuron step for unrolled slot j: charge, fire, reset.

    With `track` it also narrows the interval [lo, hi) of entry states that give the same
    spikes, using h_t(x) = h_t(guess) + beta^t (x - guess) (soft reset only).
    """
    lines = [
        *_charge_lines(indent, f"h{j}", "v", f"x{j}"),
        f"{indent}s{j} = (h{j} >= theta).to(tl.float32)",
        f"{indent}if HARD:",
        f"{indent}    v = (1.0 - s{j}) * h{j}",
        f"{indent}else:",
        f"{indent}    v = h{j} - s{j} * theta",
        f"{indent}v{j} = v",
    ]
    if track:
        lines += [
            f"{indent}if VALID:",
            f"{indent}    p = p - p * q",
            f"{indent}    bound = (theta - h{j}) / p",
            f"{indent}    bound = tl.where(p > 0.0, bound, tl.where(s{j} > 0.0, -3.0e38, 3.0e38))",
            f"{indent}    lo = tl.where(s{j} > 0.0, tl.maximum(lo, bound), lo)",
            f"{indent}    hi = tl.where(s{j} > 0.0, hi, tl.minimum(hi, bound))",
        ]
    return lines


def _store_lines(indent: str, unroll: int, word_base: str) -> list[str]:
    """Source that stores spikes, tape and membrane of the `unroll` slots just computed."""
    steps = range(unroll)
    lines = [f"{indent}if SPIKES:"]
    lines += [f"{indent}    tl.store(s_ptr + base + {j} * stride, s{j}, mask=mask)" for j in steps]
    lines.append(f"{indent}if TAPE == 1:")
    lines += [f"{indent}    tl.store(tape_ptr + base + {j} * stride, h{j}, mask=mask)" for j in steps]
    lines.append(f"{indent}if TAPE == 2:")
    lines += [f"{indent}    tl.store(tape_ptr + base + {j} * stride, h{j}.to(tl.float16), mask=mask)" for j in steps]
    if unroll % 4 == 0:
        lines.append(f"{indent}if TAPE == 3:")
        for j in steps:
            lines += [
                f"{indent}    bits = (h{j} - theta).to(tl.int32, bitcast=True)",
                f"{indent}    e = ((bits >> 23) & 255) - {_U8_EXP_SHIFT}",
                f"{indent}    m = (bits >> 20) & 7",
                f"{indent}    m = tl.where(e < 0, 0, tl.where(e > 15, 7, m))",
                f"{indent}    e = tl.minimum(tl.maximum(e, 0), 15)",
                f"{indent}    c{j} = (((bits >> 31) & 1) << 7) | (e << 3) | m",
            ]
        lines.append(f"{indent}    wbase = {word_base}")
        lines += [
            f"{indent}    tl.store(tape_ptr + wbase + {w} * stride, "
            f"c{4 * w} | (c{4 * w + 1} << 8) | (c{4 * w + 2} << 16) | (c{4 * w + 3} << 24), mask=mask)"
            for w in range(unroll // 4)
        ]
    lines.append(f"{indent}if STORE_V:")
    lines += [f"{indent}    tl.store(vseq_ptr + base + {j} * stride, v{j}, mask=mask)" for j in steps]
    return lines


def _load_x_lines(indent: str, unroll: int) -> list[str]:
    """Source that loads the input of `unroll` consecutive time steps (any float dtype)."""
    return [
        f"{indent}x{j} = tl.load(x_ptr + base + {j} * stride, mask=mask, other=0.0).to(tl.float32)"
        for j in range(unroll)
    ]


def _fwd_source(unroll: int) -> str:
    """Source of the forward kernel that handles `unroll` time steps per loop iteration."""
    ind = "        "
    warm_steps = [line for j in range(unroll) for line in _step_lines(ind, j, track=False)]
    main_steps = [line for j in range(unroll) for line in _step_lines(ind, j, track=True)]
    lines = [
        "@triton.jit",
        "def lif_fwd(x_ptr, s_ptr, tape_ptr, vseq_ptr, q_ptr, vin_ptr, vstart_ptr, vout_ptr, lo_ptr, hi_ptr,",
        "            C, W, N, D, theta,",
        "            DECAY: tl.constexpr, TAPE: tl.constexpr, SPIKES: tl.constexpr, STORE_V: tl.constexpr,",
        "            VALID: tl.constexpr, HARD: tl.constexpr, BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    chunk = tl.program_id(1)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    v = tl.load(vin_ptr + chunk * stride + n, mask=mask, other=0.0)",
        "    # Warm-up: chunks after the first start W steps early (never before t = 0), store nothing there.",
        f"    warm = tl.minimum(chunk * C, W) // {unroll}",
        f"    start = chunk * C - warm * {unroll}",
        "    for g in range(warm):",
        f"        base = n + (start + g * {unroll}) * stride",
        *_load_x_lines(ind, unroll),
        *warm_steps,
        "    tl.store(vstart_ptr + chunk * stride + n, v, mask=mask)",
        "    if VALID:",
        "        p = tl.full((BLOCK,), 1.0, tl.float32)",
        "        lo = tl.full((BLOCK,), -3.0e38, tl.float32)",
        "        hi = tl.full((BLOCK,), 3.0e38, tl.float32)",
        f"    for g in range(C // {unroll}):",
        f"        base = n + (chunk * C + g * {unroll}) * stride",
        *_load_x_lines(ind, unroll),
        *main_steps,
        *_store_lines(ind, unroll, f"n + ((chunk * C + g * {unroll}) // 4) * stride"),
        "    tl.store(vout_ptr + chunk * stride + n, v, mask=mask)",
        "    if VALID:",
        "        tl.store(lo_ptr + chunk * stride + n, lo, mask=mask)",
        "        tl.store(hi_ptr + chunk * stride + n, hi, mask=mask)",
    ]
    return "\n".join(lines) + "\n"


def _grad_step_lines(indent: str, h: str, gs: str) -> list[str]:
    """Source of one backward step: surrogate at `h`, then `gh` from `gs` and the carry.

    With a detached soft reset the carry passes unchanged. Otherwise it is scaled by the
    derivative of the reset, as in SpikingJelly's backward kernel.
    """
    return [
        f"{indent}z = ({h} - theta) * (1.5707963267948966 * alpha)",
        f"{indent}sg = (0.5 * alpha) / (1.0 + z * z)",
        f"{indent}if HARD:",
        f"{indent}    rg = 1.0 - ({h} >= theta).to(tl.float32)",
        f"{indent}    if not DETACH:",
        f"{indent}        rg = rg + (0.0 - {h}) * sg",
        f"{indent}    gh = {gs} * sg + carry * rg",
        f"{indent}else:",
        f"{indent}    if DETACH:",
        f"{indent}        gh = {gs} * sg + carry",
        f"{indent}    else:",
        f"{indent}        gh = {gs} * sg + carry * (1.0 - theta * sg)",
        f"{indent}carry = gh - gh * q",
    ]


def _bwd_source(unroll: int) -> str:
    """Source of the backward kernel (reverse loop over one chunk, `unroll` steps at a time)."""
    steps = range(unroll)
    ind = "        "
    words = unroll // 4

    decode = _decode_u8
    lines = [
        "@triton.jit",
        "def lif_bwd(gs_ptr, tape_ptr, x_ptr, gx_ptr, q_ptr, vin_ptr, cin_ptr, cout_ptr, gacc_ptr,",
        "            C, W, NCHUNK, N, D, theta, alpha,",
        "            DECAY: tl.constexpr, TAPE: tl.constexpr, USE_X: tl.constexpr, STORE: tl.constexpr,",
        "            HARD: tl.constexpr, DETACH: tl.constexpr, BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    alpha = tl.cast(alpha, tl.float32)",
        "    chunk = tl.program_id(1)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    vin = tl.load(vin_ptr + chunk * stride + n, mask=mask, other=0.0)",
        "    carry = tl.load(cin_ptr + chunk * stride + n, mask=mask, other=0.0)",
        "    gacc = tl.zeros((BLOCK,), tl.float32)",
        "    first = chunk * C",
        "    last = first + C - 1",
        "    # Warm-up of the carry: chunks before the last start W steps later (never past L), store nothing.",
        "    # (float tapes only; the launcher refuses a warm-up with the 8-bit tape)",
        f"    warm = tl.minimum((NCHUNK - 1 - chunk) * C, W) // {unroll}",
        "    for g in range(warm):",
        f"        base = n + (last + (warm - g) * {unroll}) * stride",
        *[f"{ind}gw{j} = tl.load(gs_ptr + base - {j} * stride, mask=mask, other=0.0).to(tl.float32)" for j in steps],
        *[f"{ind}hw{j} = tl.load(tape_ptr + base - {j} * stride, mask=mask, other=0.0).to(tl.float32)" for j in steps],
        *[line for j in steps for line in _grad_step_lines(ind, f"hw{j}", f"gw{j}")],
        "    if TAPE != 3:",
        "        h_cur = tl.load(tape_ptr + n + last * stride, mask=mask, other=0.0).to(tl.float32)",
        f"    for g in range(C // {unroll}):",
        f"        t_hi = last - g * {unroll}",
        "        base = n + t_hi * stride",
    ]
    lines += [f"{ind}gs{j} = tl.load(gs_ptr + base - {j} * stride, mask=mask, other=0.0).to(tl.float32)" for j in steps]
    # hc{j}: h at time t_hi - j; hq{j}: h at time t_hi - j - 1 (not used at the first step of the chunk)
    lines.append(f"{ind}if TAPE == 3:")
    if unroll % 4 == 0:
        lines.append(f"{ind}    wbase = n + (t_hi // 4) * stride")
        lines += [f"{ind}    wd{i} = tl.load(tape_ptr + wbase - {i} * stride, mask=mask, other=0)" for i in range(words)]
        lines.append(
            f"{ind}    wd{words} = tl.load(tape_ptr + wbase - {words} * stride, "
            f"mask=mask & (t_hi - {unroll - 1} > first), other=0)"
        )
        for j in steps:
            lines.append(f"{ind}    hc{j} = {decode(f'wd{j // 4}', 3 - j % 4)}")
            lines.append(f"{ind}    hq{j} = {decode(f'wd{(j + 1) // 4}', 3 - (j + 1) % 4)}")
    else:
        lines += [f"{ind}    hc{j} = carry" for j in steps] + [f"{ind}    hq{j} = carry" for j in steps]
    lines.append(f"{ind}else:")
    for j in steps:
        lines.append(
            f"{ind}    hq{j} = tl.load(tape_ptr + base - {j + 1} * stride, mask=mask & (t_hi - {j} > first), "
            "other=0.0).to(tl.float32)"
        )
    lines.append(f"{ind}    hc0 = h_cur")
    lines += [f"{ind}    hc{j} = hq{j - 1}" for j in steps if j > 0]
    lines.append(f"{ind}    h_cur = hq{unroll - 1}")
    lines.append(f"{ind}if USE_X:")
    lines += [f"{ind}    x{j} = tl.load(x_ptr + base - {j} * stride, mask=mask, other=0.0).to(tl.float32)" for j in steps]
    for j in steps:
        lines += _grad_step_lines(ind, f"hc{j}", f"gs{j}")
        lines += [
            f"{ind}if DECAY:",
            f"{ind}    gx{j} = gh * q",
            f"{ind}else:",
            f"{ind}    gx{j} = gh",
            f"{ind}if HARD:",
            f"{ind}    vp = (1.0 - (hq{j} >= theta).to(tl.float32)) * hq{j}",
            f"{ind}else:",
            f"{ind}    vp = hq{j} - (hq{j} >= theta).to(tl.float32) * theta",
            f"{ind}vp = tl.where(t_hi - {j} > first, vp, vin)",
            f"{ind}if DECAY:",
            f"{ind}    if USE_X:",
            f"{ind}        gacc += gh * (x{j} - vp)",
            f"{ind}    else:",
            f"{ind}        gacc += gh * (hc{j} - vp)",
            f"{ind}else:",
            f"{ind}    gacc -= gh * vp",
        ]
    lines.append(f"{ind}if STORE:")
    lines += [f"{ind}    tl.store(gx_ptr + base - {j} * stride, gx{j}, mask=mask)" for j in steps]
    lines += [
        "    tl.store(cout_ptr + chunk * stride + n, carry, mask=mask)",
        "    if STORE:",
        "        tl.store(gacc_ptr + chunk * stride + n, gacc, mask=mask)",
    ]
    return "\n".join(lines) + "\n"


def _kernel(kind: str, unroll: int) -> object:
    """The compiled forward ("fwd") or backward ("bwd") kernel for an unroll factor."""
    key = (kind, unroll)
    if key not in _KERNELS:
        source = _fwd_source(unroll) if kind == "fwd" else _bwd_source(unroll)
        _KERNELS[key] = _jit(f"lif_{kind}", source)
    return _KERNELS[key]


@dataclass(frozen=True)
class KernelOpts:
    """Launch options of the own kernels.

    unroll: time steps loaded together (U). block: neurons per program. warps: warps per
    program. tape: what the forward pass saves for the backward pass. use_x: the backward
    pass re-reads x for the leak gradient (SpikingJelly's formula) instead of taking
    x_t - v_{t-1} = (h_t - v_{t-1}) / q from the tape. hard: reset to 0 instead of
    subtracting theta. detach: the spike is detached in the reset (the repo's setting).
    """

    unroll: int = 8
    block: int = 256
    warps: int = 4
    tape: int = TAPE_H32
    use_x: bool = False
    store_v: bool = False
    hard: bool = False
    detach: bool = True


def zero_state(x: Tensor) -> Tensor:
    """A zero membrane [B, D] in float32 for an input [L, B, D] of any float dtype."""
    return torch.zeros(x.shape[1:], dtype=torch.float32, device=x.device)


def _tape_shape(x: Tensor, tape_mode: int) -> tuple[int, ...]:
    """Shape of the tape for input `x`: one entry per step, or one int32 per 4 steps (8-bit)."""
    if tape_mode == TAPE_U8:
        return (x.shape[0] // 4, *x.shape[1:])
    return tuple(x.shape)


def _tape_mode_of(tape: Tensor) -> int:
    """The tape code that belongs to the dtype of a tape tensor."""
    return {torch.float16: TAPE_H16, torch.int32: TAPE_U8}.get(tape.dtype, TAPE_H32)


def launch_fwd(
    x: Tensor, q: Tensor, vin: Tensor, theta: float, decay_input: bool, opts: KernelOpts,
    chunk: int, warmup: int = 0, spikes: bool = True, tape: int | None = None,
    bounds: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """Run the forward kernel over all chunks of `chunk` steps.

    `vin` is [n_chunks, B*D], the membrane each chunk starts from. With `warmup` W, every
    chunk after the first starts W steps early and stores nothing for those steps.
    Returns (spikes, tape, v_seq, vstart, vout): `vstart` is the membrane entering the
    stored part of each chunk, `vout` the membrane leaving it. Tensors that were not
    asked for are empty. If a list is passed as `bounds`, the kernel also tracks, per
    chunk and neuron, the interval [lo, hi) of offsets from `vstart` that give the same
    spikes, and appends `lo` and `hi` to the list.
    """
    seq_len, batch, width = x.shape
    neurons = batch * width
    n_chunks = seq_len // chunk
    tape_mode = opts.tape if tape is None else tape
    if seq_len % chunk or chunk % opts.unroll or warmup % opts.unroll:
        raise ValueError(f"L={seq_len}, C={chunk}, W={warmup} must be multiples of C and of U={opts.unroll}")
    if tape_mode == TAPE_U8 and (opts.unroll % 4 or chunk % 4):
        raise ValueError("the 8-bit tape needs U and C to be multiples of 4")
    if bounds is not None and opts.hard:
        raise ValueError("the same-spike interval is only defined for the soft reset")
    empty = x.new_empty(0)
    s_out = torch.empty_like(x) if spikes else empty
    tape_out = empty
    if tape_mode:
        tape_out = torch.empty(_tape_shape(x, tape_mode), dtype=_TAPE_DTYPE[tape_mode], device=x.device)
    v_seq = torch.empty(x.shape, dtype=torch.float32, device=x.device) if opts.store_v else empty
    vstart = torch.empty_like(vin)
    vout = torch.empty_like(vin)
    track = bounds is not None
    lo = torch.empty_like(vin) if track else vin
    hi = torch.empty_like(vin) if track else vin
    _kernel("fwd", opts.unroll)[(triton.cdiv(neurons, opts.block), n_chunks)](
        x, s_out, tape_out, v_seq, q, vin, vstart, vout, lo, hi, chunk, warmup, neurons, width, theta,
        decay_input, tape_mode, spikes, opts.store_v, track, opts.hard, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
    if bounds is not None:
        bounds += [lo, hi]
    return s_out, tape_out, v_seq, vstart, vout


def launch_bwd(
    gs: Tensor, tape: Tensor, x: Tensor, q: Tensor, vin: Tensor, cin: Tensor, theta: float,
    decay_input: bool, opts: KernelOpts, chunk: int, warmup: int = 0, store: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    """Run the backward kernel over all chunks. `cin` is the carry entering each chunk from the future.

    `vin` is the membrane that entered the stored part of each chunk in the forward pass.
    With `warmup` W, every chunk before the last starts W steps later and only builds up
    its carry there. Returns (gx, cout, gq): the gradient w.r.t. x, the carry leaving each
    chunk towards the past ([n_chunks, B*D]) and the gradient w.r.t. q per chunk and
    neuron. With `store=False` only `cout` is computed.
    """
    seq_len, batch, width = gs.shape
    neurons = batch * width
    n_chunks = seq_len // chunk
    tape_mode = _tape_mode_of(tape)
    if tape_mode == TAPE_U8 and warmup:
        raise ValueError("the 8-bit tape does not support a warmed-up (truncated) backward pass")
    gx = torch.empty_like(gs) if store else gs.new_empty(0)
    cout = torch.empty_like(cin)
    gacc = torch.empty_like(cin)
    _kernel("bwd", opts.unroll)[(triton.cdiv(neurons, opts.block), n_chunks)](
        gs, tape, x, gx, q, vin, cin, cout, gacc, chunk, warmup, n_chunks, neurons, width, theta, ALPHA,
        decay_input, tape_mode, opts.use_x, store, opts.hard, opts.detach, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
    if not store:
        return gx, cout, gacc
    if decay_input and not opts.use_x:
        # The kernel summed gh * (h_t - v_{t-1}); x_t - v_{t-1} is that divided by q.
        q_row = q.repeat(batch)
        gacc = torch.where(q_row > 0, gacc / q_row, torch.zeros_like(gacc))
    return gx, cout, gacc


class _SerialKernel(torch.autograd.Function):
    """The exact serial neuron as one forward and one backward kernel launch."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, opts: KernelOpts
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len, batch, width = x.shape
        vin = v0.reshape(1, batch * width).contiguous()
        spikes, tape, v_seq, _vstart, vout = launch_fwd(x, q, vin, theta, decay_input, opts, chunk=seq_len)
        ctx.save_for_backward(x, q, vin)
        ctx.tape, ctx.theta, ctx.decay_input, ctx.opts = tape, theta, decay_input, opts
        ctx.mark_non_differentiable(v_seq)
        return spikes, vout.view(batch, width), v_seq

    @staticmethod
    def backward(ctx, gs: Tensor, gv_last: Tensor, _gv_seq: Tensor) -> tuple:  # type: ignore[override]
        x, q, vin = ctx.saved_tensors
        seq_len, batch, width = x.shape
        cin = gv_last.reshape(1, batch * width).contiguous()
        gx, cout, gq = launch_bwd(gs, ctx.tape, x, q, vin, cin, ctx.theta, ctx.decay_input, ctx.opts, seq_len)
        return gx, gq.view(batch, width).sum(0), cout.view(batch, width), None, None, None


def _tape_mode(name: str) -> int:
    """"h32", "h16" or "u8" -> the kernel's tape code."""
    return {"h32": TAPE_H32, "h16": TAPE_H16, "u8": TAPE_U8}[name]


@register("serial_triton", lambda L, **_: L)
def serial_triton(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    U: int = 8,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h32",
    usex: int = 0,
    hard: int = 0,
    detach: int = 1,
) -> tuple[Tensor, ...]:
    """Exact serial neuron in an own fused kernel that loads U time steps ahead.

    tape: what the forward pass saves of the charged membrane h for the backward pass:
    "h32" float32, "h16" float16, "u8" an 8-bit code of h - theta (the forward pass is
    float32 and exact in all three). `usex=1` makes the backward pass read x again, as
    SpikingJelly does. `hard=1` resets to 0; `detach=0` keeps the spike in the reset
    differentiable. x may be float32, float16 or bfloat16; the spikes have its dtype.
    """
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), bool(usex), return_v_seq, bool(hard), bool(detach))
    start = zero_state(x) if v0 is None else v0
    spikes, v_last, v_seq = _SerialKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts)
    if return_v_seq:
        return spikes, v_last, v_seq
    return spikes, v_last


def _repair_source(unroll: int) -> str:
    """Source of the repair kernel of `spec_verify`.

    One program per neuron block walks over the chunks in order and keeps the true
    membrane. Where the true membrane entering a chunk is not certified to give the stored
    spikes (bit-equal state, or state inside the same-spike interval), the chunk is run
    again for the whole block.
    """
    ind = "            "
    lines = [
        "@triton.jit",
        "def lif_repair(x_ptr, s_ptr, tape_ptr, vseq_ptr, q_ptr, vstart_ptr, vout_ptr, nfix_ptr,",
        "               lo_ptr, hi_ptr, ac_ptr, C, NCHUNK, N, D, theta,",
        "               DECAY: tl.constexpr, TAPE: tl.constexpr, STORE_V: tl.constexpr, CERT: tl.constexpr,",
        "               HARD: tl.constexpr, SPIKES: tl.constexpr, BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    v = tl.load(vout_ptr + n, mask=mask, other=0.0)",
        "    fixed = 0",
        "    for c in range(1, NCHUNK):",
        "        guess = tl.load(vstart_ptr + c * stride + n, mask=mask, other=0.0)",
        "        if CERT == 1:",
        "            # interval certificate: same spikes if the offset from the guess is in [lo, hi)",
        "            off = v - guess",
        "            lo = tl.load(lo_ptr + c * stride + n, mask=mask, other=0.0)",
        "            hi = tl.load(hi_ptr + c * stride + n, mask=mask, other=0.0)",
        "            wrong = (off < lo) | (off >= hi)",
        "        else:",
        "            # bit certificate: same future if the state is bit-equal to the guess",
        "            wrong = v != guess",
        "        bad = tl.minimum(tl.sum((wrong & mask).to(tl.int32)), 1)",
        "        fixed += bad",
        "        v_stored = tl.load(vout_ptr + c * stride + n, mask=mask, other=0.0)",
        "        entry = v",
        f"        for g in range(bad * (C // {unroll})):",
        f"            base = n + (c * C + g * {unroll}) * stride",
        *_load_x_lines(ind, unroll),
        *[line for j in range(unroll) for line in _step_lines(ind, j, track=False)],
        *_store_lines(ind, unroll, f"n + ((c * C + g * {unroll}) // 4) * stride"),
        "        # Re-run: v holds the new end state. No re-run: the stored end state (bit certificate),",
        "        # or the stored end state moved by beta^C times the offset (interval certificate).",
        "        if CERT == 1:",
        "            v_stored = v_stored + tl.load(ac_ptr + n % D, mask=mask, other=0.0) * off",
        "        v = tl.where(bad > 0, v, v_stored)",
        "        tl.store(vstart_ptr + c * stride + n, entry, mask=mask)",
        "        tl.store(vout_ptr + c * stride + n, v, mask=mask)",
        "    tl.store(nfix_ptr + tl.program_id(0), fixed)",
    ]
    return "\n".join(lines) + "\n"


def launch_repair(
    x: Tensor, q: Tensor, theta: float, decay_input: bool, opts: KernelOpts, chunk: int,
    spikes: Tensor, tape: Tensor, v_seq: Tensor, vstart: Tensor, vout: Tensor,
    bounds: list[Tensor] | None = None,
) -> Tensor:  # fmt: skip
    """Make a speculative chunked forward pass exact, in place. Returns re-run chunks per neuron block.

    Without `bounds` a chunk is kept only if its start state is bit-equal to the true
    state. With `bounds = [lo, hi]` (from `launch_fwd`) it is kept if the true state lies
    in the interval that gives the same spikes.
    """
    seq_len, batch, width = x.shape
    neurons = batch * width
    key = ("repair", opts.unroll)
    if key not in _KERNELS:
        _KERNELS[key] = _jit("lif_repair", _repair_source(opts.unroll))
    blocks = triton.cdiv(neurons, opts.block)
    n_fixed = torch.empty(blocks, dtype=torch.int32, device=x.device)
    lo, hi = (vstart, vstart) if bounds is None else bounds
    slope = _chunk_slope(q, chunk)
    _KERNELS[key][(blocks,)](
        x, spikes, tape, v_seq, q, vstart, vout, n_fixed, lo, hi, slope,
        chunk, seq_len // chunk, neurons, width, theta,
        decay_input, _tape_mode_of(tape), opts.store_v, int(bounds is not None), opts.hard, True, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
    return n_fixed


def _decode_u8(word: str, byte: int) -> str:
    """Source expression for h from one byte of a packed 8-bit tape word."""
    code = f"(({word} >> {8 * byte}) & 255)"
    bits = (
        f"(((({code} >> 3) & 15) + {_U8_EXP_SHIFT}) << 23) | (({code} & 7) << 20) | 524288 "
        f"| (({code} >> 7) << 31)"
    )
    return f"({bits}).to(tl.float32, bitcast=True) + theta"


def _ticks_fwd_source(unroll: int, ticks: int) -> str:
    """Source of the tick-aware forward kernel: `unroll` tokens per iteration, `ticks` steps per token.

    Per token it reads x once, runs `ticks` neuron steps on it, and stores the spike count
    and one int32 that holds the 8-bit tape codes of the ticks.
    """
    ind = "        "
    lines = [
        "@triton.jit",
        "def lif_ticks_fwd(x_ptr, c_ptr, tape_ptr, vseq_ptr, q_ptr, vin_ptr, vout_ptr, L, N, D, theta,",
        "                  DECAY: tl.constexpr, STORE_V: tl.constexpr, HARD: tl.constexpr, BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    v = tl.load(vin_ptr + n, mask=mask, other=0.0)",
        f"    for g in range(L // {unroll}):",
        f"        base = n + (g * {unroll}) * stride",
        *_load_x_lines(ind, unroll),
    ]
    for j in range(unroll):
        for k in range(ticks):
            lines += _charge_lines(ind, "h", "v", f"x{j}")
            lines += [
                f"{ind}s = (h >= theta).to(tl.float32)",
                f"{ind}if HARD:",
                f"{ind}    v = (1.0 - s) * h",
                f"{ind}else:",
                f"{ind}    v = h - s * theta",
                f"{ind}cnt{j} = s" if k == 0 else f"{ind}cnt{j} = cnt{j} + s",
                f"{ind}bits = (h - theta).to(tl.int32, bitcast=True)",
                f"{ind}e = ((bits >> 23) & 255) - {_U8_EXP_SHIFT}",
                f"{ind}m = (bits >> 20) & 7",
                f"{ind}m = tl.where(e < 0, 0, tl.where(e > 15, 7, m))",
                f"{ind}e = tl.minimum(tl.maximum(e, 0), 15)",
                f"{ind}code = (((bits >> 31) & 1) << 7) | (e << 3) | m",
                f"{ind}w{j} = code" if k == 0 else f"{ind}w{j} = w{j} | (code << {8 * k})",
            ]
        lines.append(f"{ind}v{j} = v")
    lines += [f"{ind}tl.store(c_ptr + base + {j} * stride, cnt{j}, mask=mask)" for j in range(unroll)]
    lines += [f"{ind}tl.store(tape_ptr + base + {j} * stride, w{j}, mask=mask)" for j in range(unroll)]
    lines.append(f"{ind}if STORE_V:")
    lines += [f"{ind}    tl.store(vseq_ptr + base + {j} * stride, v{j}, mask=mask)" for j in range(unroll)]
    lines.append("    tl.store(vout_ptr + n, v, mask=mask)")
    return "\n".join(lines) + "\n"


def _ticks_bwd_source(unroll: int, ticks: int) -> str:
    """Source of the tick-aware backward kernel (reverse over tokens, and over the ticks of a token)."""
    ind = "        "
    lines = [
        "@triton.jit",
        "def lif_ticks_bwd(gc_ptr, tape_ptr, gx_ptr, q_ptr, vin_ptr, cin_ptr, cout_ptr, gacc_ptr,",
        "                  L, N, D, theta, alpha,",
        "                  DECAY: tl.constexpr, HARD: tl.constexpr, DETACH: tl.constexpr, BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    alpha = tl.cast(alpha, tl.float32)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    vin = tl.load(vin_ptr + n, mask=mask, other=0.0)",
        "    carry = tl.load(cin_ptr + n, mask=mask, other=0.0)",
        "    gacc = tl.zeros((BLOCK,), tl.float32)",
        f"    for g in range(L // {unroll}):",
        f"        t_hi = L - 1 - g * {unroll}",
        "        base = n + t_hi * stride",
        *[f"{ind}gs{j} = tl.load(gc_ptr + base - {j} * stride, mask=mask, other=0.0).to(tl.float32)" for j in range(unroll)],
        *[f"{ind}wd{j} = tl.load(tape_ptr + base - {j} * stride, mask=mask, other=0)" for j in range(unroll)],
        f"{ind}wd{unroll} = tl.load(tape_ptr + base - {unroll} * stride, mask=mask & (t_hi - {unroll - 1} > 0), other=0)",
    ]
    for j in range(unroll):
        lines.append(f"{ind}acc = tl.zeros((BLOCK,), tl.float32)")
        for k in reversed(range(ticks)):
            previous = _decode_u8(f"wd{j}", k - 1) if k > 0 else _decode_u8(f"wd{j + 1}", ticks - 1)
            lines += [f"{ind}hc = {_decode_u8(f'wd{j}', k)}", f"{ind}hq = {previous}"]
            lines += _grad_step_lines(ind, "hc", f"gs{j}")
            lines += [
                f"{ind}if DECAY:",
                f"{ind}    acc += gh * q",
                f"{ind}else:",
                f"{ind}    acc += gh",
                f"{ind}if HARD:",
                f"{ind}    vp = (1.0 - (hq >= theta).to(tl.float32)) * hq",
                f"{ind}else:",
                f"{ind}    vp = hq - (hq >= theta).to(tl.float32) * theta",
            ]
            if k == 0:
                lines.append(f"{ind}vp = tl.where(t_hi - {j} > 0, vp, vin)")
            lines += [
                f"{ind}if DECAY:",
                f"{ind}    gacc += gh * (hc - vp)",
                f"{ind}else:",
                f"{ind}    gacc -= gh * vp",
            ]
        lines.append(f"{ind}gx{j} = acc")
    lines += [f"{ind}tl.store(gx_ptr + base - {j} * stride, gx{j}, mask=mask)" for j in range(unroll)]
    lines += [
        "    tl.store(cout_ptr + n, carry, mask=mask)",
        "    tl.store(gacc_ptr + n, gacc, mask=mask)",
    ]
    return "\n".join(lines) + "\n"


class _TickKernel(torch.autograd.Function):
    """T ticks per token with the membrane carried over all ticks; input and output per token."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, opts: KernelOpts, ticks: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len, batch, width = x.shape
        neurons = batch * width
        if seq_len % opts.unroll or not 1 <= ticks <= 4:
            raise ValueError(f"L={seq_len} must be a multiple of U={opts.unroll}, and 1 <= T <= 4")
        key = (f"ticks-fwd-{ticks}", opts.unroll)
        if key not in _KERNELS:
            _KERNELS[key] = _jit("lif_ticks_fwd", _ticks_fwd_source(opts.unroll, ticks))
        vin = v0.reshape(1, neurons).contiguous()
        counts = torch.empty_like(x)
        tape = torch.empty(x.shape, dtype=torch.int32, device=x.device)
        v_seq = torch.empty(x.shape, dtype=torch.float32, device=x.device) if opts.store_v else x.new_empty(0)
        vout = torch.empty_like(vin)
        _KERNELS[key][(triton.cdiv(neurons, opts.block),)](
            x, counts, tape, v_seq, q, vin, vout, seq_len, neurons, width, theta,
            decay_input, opts.store_v, opts.hard, opts.block, num_warps=opts.warps, enable_fp_fusion=False,
        )  # fmt: skip
        ctx.save_for_backward(q, vin)
        ctx.tape, ctx.theta, ctx.decay_input, ctx.opts, ctx.ticks = tape, theta, decay_input, opts, ticks
        ctx.mark_non_differentiable(v_seq)
        return counts, vout.view(batch, width), v_seq

    @staticmethod
    def backward(ctx, gc: Tensor, gv_last: Tensor, _gv_seq: Tensor) -> tuple:  # type: ignore[override]
        """Gradients w.r.t. the token input, the leak q per channel and the start membrane."""
        q, vin = ctx.saved_tensors
        seq_len, batch, width = gc.shape
        neurons = batch * width
        opts = ctx.opts
        key = (f"ticks-bwd-{ctx.ticks}", opts.unroll)
        if key not in _KERNELS:
            _KERNELS[key] = _jit("lif_ticks_bwd", _ticks_bwd_source(opts.unroll, ctx.ticks))
        gx = torch.empty_like(gc)
        cin = gv_last.reshape(1, neurons).contiguous()
        cout = torch.empty_like(cin)
        gacc = torch.empty_like(cin)
        _KERNELS[key][(triton.cdiv(neurons, opts.block),)](
            gc, ctx.tape, gx, q, vin, cin, cout, gacc, seq_len, neurons, width, ctx.theta, ALPHA,
            ctx.decay_input, opts.hard, opts.detach, opts.block, num_warps=opts.warps, enable_fp_fusion=False,
        )  # fmt: skip
        gq = gacc.view(batch, width).sum(0)
        if ctx.decay_input:  # the kernel summed gh * (h - v_prev); x - v_prev is that divided by q
            gq = torch.where(q > 0, gq / q, torch.zeros_like(gq))
        return gx, gq, cout.view(batch, width), None, None, None, None


@register("serial_ticks", lambda L, T=4, **_: L * T)
def serial_ticks(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    T: int = 4,
    U: int = 8,
    BLOCK: int = 256,
    nw: int = 4,
    hard: int = 0,
    detach: int = 1,
) -> tuple[Tensor, ...]:
    """T ticks per token, membrane carried over all L*T ticks, one kernel pass over the tokens.

    `x` is per token, [L, B, D]. Returns the spike COUNT per token (0..T), not the spikes,
    then the last membrane and, if asked, the membrane after the last tick of every token.
    """
    opts = KernelOpts(U, BLOCK, nw, TAPE_U8, False, return_v_seq, bool(hard), bool(detach))
    start = zero_state(x) if v0 is None else v0
    counts, v_last, v_seq = _TickKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts, T)
    if return_v_seq:
        return counts, v_last, v_seq
    return counts, v_last


def _chunk_slope(q: Tensor, chunk: int) -> Tensor:
    """beta^C per channel: the slope of the map from a chunk's entry state to its end state."""
    return ((1.0 - q.double()) ** chunk).float()


def _chunk_powers(q: Tensor, chunk: int, n_chunks: int, upper: bool) -> Tensor:
    """[D, n, n] matrix of powers of a = beta^C that carries a state over whole chunks.

    Lower form (forward): entry [i, j] = a^(i-1-j) for j < i. Upper form (backward):
    entry [i, j] = a^(j-i) for j >= i.
    """
    a = ((1.0 - q.double()) ** chunk).view(-1, 1, 1)
    i = torch.arange(n_chunks, device=q.device).view(1, n_chunks, 1)
    j = torch.arange(n_chunks, device=q.device).view(1, 1, n_chunks)
    if upper:
        return torch.where(j >= i, a ** (j - i).clamp_min(0), torch.zeros_like(a)).float()
    return torch.where(j < i, a ** (i - 1 - j).clamp_min(0), torch.zeros_like(a)).float()


# --------------------------------------------------------------------------------------
# A. Chunked methods: every chunk of C tokens is its own program in one kernel launch
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPlan:
    """How a chunked method finds the state that enters each chunk.

    fwd "cold": start `warmup` tokens early from zero. "picard": `passes` passes, each
    chunk starting from the end state its neighbour had in the pass before. "parareal":
    the same with the affine coarse correction. "verify": cold start, then the repair
    kernel makes the result exact.
    bwd "same": the same rule for the gradient carry ("cold" and "picard"). "exact": one
    pass for the carries, the exact linear carry recursion, one pass that stores.
    "serial": one serial reverse pass.
    """

    chunk: int
    fwd: str
    warmup: int = 0
    passes: int = 1
    bwd: str = "same"
    bwd_warmup: int = -1  # -1: use `warmup`
    cert: str = "bit"  # fwd "verify": "bit" (bit-equal states) or "interval" (same-spike interval)


def _first_row(row: Tensor, n_chunks: int) -> Tensor:
    """[n_chunks, N] tensor that is `row` for chunk 0 and zero for the other chunks."""
    out = row.new_zeros(n_chunks, row.numel())
    out[0] = row.reshape(-1)
    return out


def _last_row(row: Tensor, n_chunks: int) -> Tensor:
    """[n_chunks, N] tensor that is `row` for the last chunk and zero for the others."""
    out = row.new_zeros(n_chunks, row.numel())
    out[-1] = row.reshape(-1)
    return out


class _ChunkedKernel(torch.autograd.Function):
    """Chunked forward and backward with the own kernels, driven by a `ChunkPlan`."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, opts: KernelOpts, plan: ChunkPlan
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len, batch, width = x.shape
        n_chunks = seq_len // plan.chunk
        vin = _first_row(v0, n_chunks)
        if plan.fwd in ("picard", "parareal"):
            powers = _chunk_powers(q, plan.chunk, n_chunks, upper=False) if plan.fwd == "parareal" else None
            for _ in range(plan.passes - 1):
                ends = launch_fwd(x, q, vin, theta, decay_input, opts, plan.chunk, spikes=False, tape=TAPE_NONE)[4]
                shifted = torch.cat((vin[:1], ends[:-1]))
                if powers is None:
                    vin = shifted
                else:
                    # defect r[c] = F(U_old[c]) - U_old[c+1]; correction = triangular solve in closed form
                    defect = torch.cat(((shifted - vin)[1:], vin.new_zeros(1, vin.shape[1])))
                    delta = torch.einsum("dcj,jbd->cbd", powers, defect.view(n_chunks, batch, width))
                    vin = vin + delta.reshape(n_chunks, -1)
        bounds: list[Tensor] | None = [] if (plan.fwd == "verify" and plan.cert == "interval") else None
        spikes, tape, v_seq, vstart, vout = launch_fwd(
            x, q, vin, theta, decay_input, opts, plan.chunk, plan.warmup, bounds=bounds
        )
        if plan.fwd == "verify":
            n_fixed = launch_repair(
                x, q, theta, decay_input, opts, plan.chunk, spikes, tape, v_seq, vstart, vout, bounds
            )
            if opts.store_v:  # fidelity runs only: reading the counter forces a synchronisation
                total = n_fixed.numel() * (n_chunks - 1)
                LAST_NOTE.append(f"rerun {int(n_fixed.sum())} of {total} (block,chunk) pairs")
        ctx.save_for_backward(x, q)
        ctx.tape, ctx.vstart, ctx.theta, ctx.decay_input, ctx.opts, ctx.plan = (
            tape, vstart, theta, decay_input, opts, plan,
        )  # fmt: skip
        ctx.mark_non_differentiable(v_seq)
        return spikes, vout[-1].view(batch, width), v_seq

    @staticmethod
    def backward(ctx, gs: Tensor, gv_last: Tensor, _gv_seq: Tensor) -> tuple:  # type: ignore[override]
        x, q = ctx.saved_tensors
        seq_len, batch, width = x.shape
        plan, opts = ctx.plan, ctx.opts
        n_chunks = seq_len // plan.chunk
        mode = plan.fwd if plan.bwd == "same" else plan.bwd
        if mode == "serial":
            vin, cin = ctx.vstart[:1].contiguous(), gv_last.reshape(1, -1).contiguous()
            gx, cout, gq = launch_bwd(gs, ctx.tape, x, q, vin, cin, ctx.theta, ctx.decay_input, opts, seq_len)
            return gx, gq.view(batch, width).sum(0), cout.view(batch, width), None, None, None, None
        args = (gs, ctx.tape, x, q, ctx.vstart)
        cin = _last_row(gv_last, n_chunks)
        warmup = 0
        if mode == "cold":
            warmup = plan.warmup if plan.bwd_warmup < 0 else plan.bwd_warmup
        if mode == "picard":
            for _ in range(plan.passes - 1):
                cout = launch_bwd(*args, cin, ctx.theta, ctx.decay_input, opts, plan.chunk, store=False)[1]
                cin = torch.cat((cout[1:], cin[-1:]))
        elif mode == "exact":
            # The backward pass is linear: carry-out = carry-out for zero carry-in + beta^C * carry-in.
            local = launch_bwd(*args, cin.new_zeros(cin.shape), ctx.theta, ctx.decay_input, opts, plan.chunk, store=False)[1]
            source = torch.cat((local[1:], gv_last.reshape(1, -1)))
            powers = _chunk_powers(q, plan.chunk, n_chunks, upper=True)
            cin = torch.einsum("dcj,jbd->cbd", powers, source.view(n_chunks, batch, width)).reshape(n_chunks, -1)
        gx, cout, gq = launch_bwd(*args, cin, ctx.theta, ctx.decay_input, opts, plan.chunk, warmup)
        return gx, gq.sum(0).view(batch, width).sum(0), cout[0].view(batch, width), None, None, None, None


def _chunked(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor | None, decay_input: bool, return_v_seq: bool,
    opts: KernelOpts, plan: ChunkPlan,
) -> tuple[Tensor, ...]:  # fmt: skip
    """Shared wrapper of the chunked methods."""
    start = zero_state(x) if v0 is None else v0
    spikes, v_last, v_seq = _ChunkedKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts, plan)
    if return_v_seq:
        return spikes, v_last, v_seq
    return spikes, v_last


@register("chunk_cold", lambda L, C=256, W=64, **_: C + W)
def chunk_cold(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 256,
    W: int = 64,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """Chunks of C tokens, each warmed up over the W tokens before it, starting from v = 0."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    return _chunked(x, tau, theta, v0, decay_input, return_v_seq, opts, ChunkPlan(C, "cold", warmup=W))


@register("chunk_warm", lambda L, C=256, p=2, **_: p * C)
def chunk_warm(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 256,
    p: int = 2,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """Chunk-level Picard: p passes, each chunk starts from its predecessor's previous end state."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    return _chunked(x, tau, theta, v0, decay_input, return_v_seq, opts, ChunkPlan(C, "picard", passes=p))


@register("spec_verify", lambda L, C=64, W=64, bwd="serial", **_: L if bwd == "serial" else C + W)
def spec_verify(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 64,
    W: int = 64,
    bwd: str = "serial",
    Wb: int = -1,
    cert: str = "bit",
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
    hard: int = 0,
    detach: int = 1,
) -> tuple[Tensor, ...]:
    """Speculate like `chunk_cold`, prove each chunk correct, repair the rest.

    cert: "bit" keeps a chunk if its start state is bit-equal to the true state (result
    bit-identical to serial); "interval" keeps it if the true state lies in the interval
    that gives the same spikes (spikes identical, membrane equal up to beta^W).
    bwd: "serial" (one serial reverse pass), "exact" (chunked, two passes, exact) or
    "cold" (chunked, carry warmed up over Wb tokens, truncated).
    """
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq, bool(hard), bool(detach))
    plan = ChunkPlan(C, "verify", warmup=W, bwd=bwd, bwd_warmup=Wb, cert=cert)
    return _chunked(x, tau, theta, v0, decay_input, return_v_seq, opts, plan)


@register("parareal", lambda L, C=256, iters=2, **_: iters * C)
def parareal(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 256,
    iters: int = 2,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """Parareal on the chunk entry states: exact chunks (fine), reset-free affine chunk (coarse)."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    plan = ChunkPlan(C, "parareal", passes=iters, bwd="exact")
    return _chunked(x, tau, theta, v0, decay_input, return_v_seq, opts, plan)


# --------------------------------------------------------------------------------------
# A/B. Sweep-based methods as a cascade: K iterates advanced together in one serial pass
# --------------------------------------------------------------------------------------
#
# Picard (FPT, PMBC, Jacobi waveform relaxation) and Newton (FidLIF, DEER) sweep over the
# whole sequence K times. Iterate k at time t needs iterate k-1 only at times t and t-1.
# So one program can hold K membranes per neuron and advance all K iterates step by step.


def _charge_lines(indent: str, h: str, v: str, x: str) -> list[str]:
    """Source lines that set `h` to the charged membrane from state `v` and input `x`."""
    return [
        f"{indent}if DECAY:",
        f"{indent}    {h} = {v} + ({x} - {v}) * q",
        f"{indent}else:",
        f"{indent}    {h} = {v} - {v} * q + {x}",
    ]


def _cascade_source(unroll: int, stages: int, kind: str) -> str:
    """Source of a forward kernel that advances `stages` iterates per time step.

    kind "picard": stage k resets with the spike of stage k-1 at the same step
    (stage 1 with the constant INIT, or with a stored iterate when EXT is set).
    kind "newton": Newton step on the sigmoid-relaxed neuron, zero trajectory as start.
    The output spike is always the Heaviside of the last stage's charged membrane.
    """
    steps = range(unroll)
    ks = range(1, stages + 1)
    ind = "        "
    slopes = ", ".join(f"a{k}" for k in ks)
    lines = [
        "@triton.jit",
        "def lif_cascade(x_ptr, s_ptr, tape_ptr, vseq_ptr, q_ptr, vin_ptr, vout_ptr, ext_ptr,",
        f"                L, N, D, theta, shift, {slopes},",
        "                DECAY: tl.constexpr, TAPE: tl.constexpr, SPIKES: tl.constexpr, STORE_V: tl.constexpr,",
        "                RELAX: tl.constexpr, INIT: tl.constexpr, EXT: tl.constexpr, OUT_SOFT: tl.constexpr,",
        "                BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    shift = tl.cast(shift, tl.float32)",
        *[f"    a{k} = tl.cast(a{k}, tl.float32)" for k in ks],
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    v1 = tl.load(vin_ptr + n, mask=mask, other=0.0)",
        *[f"    v{k} = v1" for k in ks if k > 1],
        f"    for g in range(L // {unroll}):",
        f"        base = n + (g * {unroll}) * stride",
        *[f"{ind}x{j} = tl.load(x_ptr + base + {j} * stride, mask=mask, other=0.0)" for j in steps],
        f"{ind}if EXT:",
        *[f"{ind}    e{j} = tl.load(ext_ptr + base + {j} * stride, mask=mask, other=0.0)" for j in steps],
    ]
    for j in steps:
        if kind == "picard":
            lines += [f"{ind}if EXT:", f"{ind}    g0 = e{j}", f"{ind}else:", f"{ind}    g0 = INIT * 1.0"]
            for k in ks:
                lines += _charge_lines(ind, f"h{k}", f"v{k}", f"x{j}")
                lines += [
                    f"{ind}if RELAX:",
                    f"{ind}    g{k} = tl.sigmoid(a{k} * (h{k} - shift))",
                    f"{ind}else:",
                    f"{ind}    g{k} = (h{k} >= theta).to(tl.float32)",
                    f"{ind}v{k} = h{k} - theta * g{k - 1}",
                ]
            lines += [
                f"{ind}ho{j} = h{stages}",
                f"{ind}if OUT_SOFT:",
                f"{ind}    so{j} = g{stages}",
                f"{ind}else:",
                f"{ind}    so{j} = (h{stages} >= theta).to(tl.float32)",
                f"{ind}vo{j} = v{stages}",
            ]
        else:
            lines += _charge_lines(ind, f"ho{j}", f"v{stages}", f"x{j}")
            lines.append(f"{ind}so{j} = (ho{j} >= theta).to(tl.float32)")
            for k in reversed(ks):
                lower = f"v{k - 1}" if k > 1 else "zero"
                if k == 1:
                    lines.append(f"{ind}zero = tl.zeros((BLOCK,), tl.float32)")
                lines += _charge_lines(ind, "hn", lower, f"x{j}")
                lines += [
                    f"{ind}sig = tl.sigmoid(a{k} * (hn - theta))",
                    f"{ind}v{k} = (hn - theta * sig) + (1.0 - q) * (1.0 - theta * a{k} * sig * (1.0 - sig)) "
                    f"* (v{k} - {lower})",
                ]
            lines.append(f"{ind}vo{j} = v{stages}")
    lines.append(f"{ind}if SPIKES:")
    lines += [f"{ind}    tl.store(s_ptr + base + {j} * stride, so{j}, mask=mask)" for j in steps]
    lines.append(f"{ind}if TAPE == 1:")
    lines += [f"{ind}    tl.store(tape_ptr + base + {j} * stride, ho{j}, mask=mask)" for j in steps]
    lines.append(f"{ind}if TAPE == 2:")
    lines += [f"{ind}    tl.store(tape_ptr + base + {j} * stride, ho{j}.to(tl.float16), mask=mask)" for j in steps]
    lines.append(f"{ind}if STORE_V:")
    lines += [f"{ind}    tl.store(vseq_ptr + base + {j} * stride, vo{j}, mask=mask)" for j in steps]
    lines.append(f"    tl.store(vout_ptr + n, v{stages}, mask=mask)")
    return "\n".join(lines) + "\n"


def _prf_source(unroll: int) -> str:
    """Source of the reset loop of `decoupled_threshold`: it reads the membrane without reset."""
    steps = range(unroll)
    ind = "        "
    lines = [
        "@triton.jit",
        "def lif_prf(k_ptr, s_ptr, tape_ptr, vseq_ptr, q_ptr, vout_ptr, L, N, D, theta,",
        "            TAPE: tl.constexpr, STORE_V: tl.constexpr, BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    r = tl.zeros((BLOCK,), tl.float32)",
        "    v = r",
        f"    for g in range(L // {unroll}):",
        f"        base = n + (g * {unroll}) * stride",
        *[f"{ind}k{j} = tl.load(k_ptr + base + {j} * stride, mask=mask, other=0.0)" for j in steps],
    ]
    for j in steps:
        lines += [
            f"{ind}h{j} = k{j} - r",
            f"{ind}s{j} = (h{j} >= theta).to(tl.float32)",
            f"{ind}v = h{j} - s{j} * theta",
            f"{ind}v{j} = v",
            f"{ind}r = (r + s{j} * theta) - (r + s{j} * theta) * q",
        ]
    lines += [f"{ind}tl.store(s_ptr + base + {j} * stride, s{j}, mask=mask)" for j in steps]
    lines.append(f"{ind}if TAPE == 1:")
    lines += [f"{ind}    tl.store(tape_ptr + base + {j} * stride, h{j}, mask=mask)" for j in steps]
    lines.append(f"{ind}if TAPE == 2:")
    lines += [f"{ind}    tl.store(tape_ptr + base + {j} * stride, h{j}.to(tl.float16), mask=mask)" for j in steps]
    lines.append(f"{ind}if STORE_V:")
    lines += [f"{ind}    tl.store(vseq_ptr + base + {j} * stride, v{j}, mask=mask)" for j in steps]
    lines.append("    tl.store(vout_ptr + n, v, mask=mask)")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CascadeSpec:
    """Which sweep-based method the cascade kernel runs.

    kind: "picard" or "newton". stages: K. slopes: sigmoid slope per stage. shift: the
    point where the sigmoid is 0.5. relax: sigmoid (True) or Heaviside (False) feedback.
    init: the constant iterate 0 (0 or 1). passes: run one kernel pass per iterate
    instead of all iterates in one pass. sample: draw the spikes from the last sigmoid.
    """

    kind: str
    stages: int
    slopes: tuple[float, ...]
    shift: float
    relax: bool = False
    init: int = 0
    passes: bool = False
    sample: bool = False


def launch_cascade(
    x: Tensor, q: Tensor, vin: Tensor, theta: float, decay_input: bool, opts: KernelOpts, spec: CascadeSpec,
    ext: Tensor | None = None, out_soft: bool = False, tape: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """Run the cascade kernel once over the whole sequence. Returns (spikes, tape, v_seq, vout)."""
    seq_len, batch, width = x.shape
    neurons = batch * width
    if seq_len % opts.unroll:
        raise ValueError(f"L={seq_len} must be a multiple of U={opts.unroll}")
    key = (f"cascade-{spec.kind}-{spec.stages}", opts.unroll)
    if key not in _KERNELS:
        _KERNELS[key] = _jit("lif_cascade", _cascade_source(opts.unroll, spec.stages, spec.kind))
    tape_mode = opts.tape if tape is None else tape
    empty = x.new_empty(0)
    s_out = torch.empty_like(x)
    tape_out = torch.empty(x.shape, dtype=_TAPE_DTYPE[tape_mode], device=x.device) if tape_mode else empty
    v_seq = torch.empty_like(x) if opts.store_v else empty
    vout = torch.empty_like(vin)
    _KERNELS[key][(triton.cdiv(neurons, opts.block),)](
        x, s_out, tape_out, v_seq, q, vin, vout, x if ext is None else ext,
        seq_len, neurons, width, theta, spec.shift, *spec.slopes,
        decay_input, tape_mode, True, opts.store_v, spec.relax, spec.init, ext is not None, out_soft, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
    return s_out, tape_out, v_seq, vout


def _serial_backward(ctx: object, gs: Tensor, gv_last: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The standard backward pass (one serial reverse kernel) on a saved tape.

    Uses `ctx.saved_tensors = (x, q, vin)`, `ctx.tape`, `ctx.theta`, `ctx.decay_input`,
    `ctx.opts`. Returns the gradients w.r.t. x, q (per channel) and v0.
    """
    x, q, vin = ctx.saved_tensors  # type: ignore[attr-defined]
    seq_len, batch, width = x.shape
    cin = gv_last.reshape(1, batch * width).contiguous()
    gx, cout, gq = launch_bwd(
        gs, ctx.tape, x, q, vin, cin, ctx.theta, ctx.decay_input, ctx.opts, seq_len  # type: ignore[attr-defined]
    )
    return gx, gq.view(batch, width).sum(0), cout.view(batch, width)


class _CascadeKernel(torch.autograd.Function):
    """A sweep-based method: cascade forward, standard serial backward on the last iterate."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, opts: KernelOpts, spec: CascadeSpec
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len, batch, width = x.shape
        vin = v0.reshape(1, batch * width).contiguous()
        if spec.passes:
            iterate = None
            for k in range(spec.stages):
                last = k == spec.stages - 1
                one = CascadeSpec(spec.kind, 1, (spec.slopes[k],), spec.shift, spec.relax, spec.init)
                soft = spec.relax and not last
                iterate, tape, v_seq, vout = launch_cascade(
                    x, q, vin, theta, decay_input, opts, one, iterate, soft, None if last else TAPE_NONE
                )
            spikes = iterate
        else:
            spikes, tape, v_seq, vout = launch_cascade(x, q, vin, theta, decay_input, opts, spec, None, spec.sample)
        if spec.sample:
            generator = torch.Generator(device=x.device).manual_seed(0)
            spikes = torch.bernoulli(spikes, generator=generator)
        ctx.save_for_backward(x, q, vin)
        ctx.tape, ctx.theta, ctx.decay_input, ctx.opts = tape, theta, decay_input, opts
        ctx.mark_non_differentiable(v_seq)
        return spikes, vout.view(batch, width), v_seq

    @staticmethod
    def backward(ctx, gs: Tensor, gv_last: Tensor, _gv_seq: Tensor) -> tuple:  # type: ignore[override]
        return (*_serial_backward(ctx, gs, gv_last), None, None, None, None)


def _cascade(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor | None, decay_input: bool, return_v_seq: bool,
    opts: KernelOpts, spec: CascadeSpec,
) -> tuple[Tensor, ...]:  # fmt: skip
    """Shared wrapper of the cascade methods."""
    start = zero_state(x) if v0 is None else v0
    spikes, v_last, v_seq = _CascadeKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts, spec)
    if return_v_seq:
        return spikes, v_last, v_seq
    return spikes, v_last


def _fpt_slopes(stages: int, top: float = 12.0) -> tuple[float, ...]:
    """Rising sigmoid slopes from 1 to `top`, as in the FPT paper (1, 3, 12 for K = 3)."""
    if stages == 1:
        return (top,)
    return tuple(round(top ** (k / (stages - 1)), 3) for k in range(stages))


@register("fpt", lambda L, K=3, **_: K)
def fpt(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    K: int = 3,
    relax: str = "hard",
    impl: str = "cascade",
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """Fixed-point (Picard) iteration on the spike train, K iterates, hard or sigmoid feedback."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    spec = CascadeSpec("picard", K, _fpt_slopes(K), float(theta), relax == "sigmoid", 0, impl == "passes")
    return _cascade(x, tau, theta, v0, decay_input, return_v_seq, opts, spec)


@register("pmbc", lambda L, iters=4, **_: iters)
def pmbc(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    iters: int = 4,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """Lower bound of the spike train after `iters` rounds of max-min boundary compression.

    The lower bound after M rounds is Picard iterate M started from all-zeros (M even) or
    from all-ones (M odd), so this is the Picard cascade with that start.
    """
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    spec = CascadeSpec("picard", iters, (1.0,) * iters, float(theta), False, iters % 2)
    return _cascade(x, tau, theta, v0, decay_input, return_v_seq, opts, spec)


@register("newton_soft", lambda L, K=3, **_: K)
def newton_soft(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    K: int = 3,
    alpha: float = 4.0,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """K Newton steps on the membrane trajectory of the sigmoid-relaxed neuron (slope `alpha`)."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    spec = CascadeSpec("newton", K, (float(alpha),) * K, float(theta))
    return _cascade(x, tau, theta, v0, decay_input, return_v_seq, opts, spec)


@register("psn", lambda L, **_: 1)
def psn(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """The neuron without reset: spike = H(leaky sum of the input - theta)."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    return _cascade(x, tau, theta, v0, decay_input, return_v_seq, opts, CascadeSpec("picard", 1, (1.0,), float(theta)))


@register("spsn", lambda L, **_: 1)
def spsn(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    alpha: float = 4.0,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """No reset, stochastic spikes: s ~ Bernoulli(sigmoid(alpha (membrane - theta))), fixed seed."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    spec = CascadeSpec("picard", 1, (float(alpha),), float(theta), relax=True, sample=True)
    return _cascade(x, tau, theta, v0, decay_input, return_v_seq, opts, spec)


@register("psu", lambda L, **_: 2)
def psu(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """One reset estimate: the reset uses sigmoid(membrane without reset), then a hard threshold."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    spec = CascadeSpec("picard", 2, (1.0, 1.0), 0.0, relax=True)
    return _cascade(x, tau, theta, v0, decay_input, return_v_seq, opts, spec)


class _PrfKernel(torch.autograd.Function):
    """decoupled_threshold: reset-free scan, then a loop over the reset trace only."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, opts: KernelOpts
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len, batch, width = x.shape
        neurons = batch * width
        vin = v0.reshape(1, neurons).contiguous()
        free = CascadeSpec("picard", 1, (1.0,), theta)
        membrane = launch_cascade(x, q, vin, theta, decay_input, opts, free, tape=TAPE_H32)[1]
        key = ("prf", opts.unroll)
        if key not in _KERNELS:
            _KERNELS[key] = _jit("lif_prf", _prf_source(opts.unroll))
        spikes = torch.empty_like(x)
        tape = torch.empty(x.shape, dtype=_TAPE_DTYPE[opts.tape], device=x.device)
        v_seq = torch.empty_like(x) if opts.store_v else x.new_empty(0)
        vout = torch.empty_like(vin)
        _KERNELS[key][(triton.cdiv(neurons, opts.block),)](
            membrane, spikes, tape, v_seq, q, vout, seq_len, neurons, width, theta,
            opts.tape, opts.store_v, opts.block, num_warps=opts.warps, enable_fp_fusion=False,
        )  # fmt: skip
        ctx.save_for_backward(x, q, vin)
        ctx.tape, ctx.theta, ctx.decay_input, ctx.opts = tape, theta, decay_input, opts
        ctx.mark_non_differentiable(v_seq)
        return spikes, vout.view(batch, width), v_seq

    @staticmethod
    def backward(ctx, gs: Tensor, gv_last: Tensor, _gv_seq: Tensor) -> tuple:  # type: ignore[override]
        return (*_serial_backward(ctx, gs, gv_last), None, None, None)


@register("decoupled_threshold", lambda L, **_: L)
def decoupled_threshold(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """PRF-style: membrane without reset by one scan, then the reset as a moving threshold."""
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    start = zero_state(x) if v0 is None else v0
    spikes, v_last, v_seq = _PrfKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts)
    if return_v_seq:
        return spikes, v_last, v_seq
    return spikes, v_last


# --------------------------------------------------------------------------------------
# Methods written with PyTorch operations; they get the standard backward pass on a tape
# --------------------------------------------------------------------------------------


class _TapeGrad(torch.autograd.Function):
    """Attach the standard backward pass to spikes that were computed outside autograd.

    `tape` is the charged membrane h that goes with the spikes. The backward pass is the
    same serial reverse kernel that `serial_triton` uses.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, spikes: Tensor, tape: Tensor, v_last: Tensor,
        theta: float, decay_input: bool, opts: KernelOpts,
    ) -> tuple[Tensor, Tensor]:  # fmt: skip
        batch, width = x.shape[1:]
        ctx.save_for_backward(x, q, v0.reshape(1, batch * width).contiguous())
        ctx.tape, ctx.theta, ctx.decay_input, ctx.opts = tape.contiguous(), theta, decay_input, opts
        return spikes.view_as(spikes), v_last.view_as(v_last)

    @staticmethod
    def backward(ctx, gs: Tensor, gv_last: Tensor) -> tuple:  # type: ignore[override]
        return (*_serial_backward(ctx, gs, gv_last), None, None, None, None, None, None)


def _with_tape_grad(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor, decay_input: bool, return_v_seq: bool,
    spikes: Tensor, tape: Tensor, v_seq: Tensor,
) -> tuple[Tensor, ...]:  # fmt: skip
    """Return (spikes, v_last[, v_seq]) with the standard backward pass attached."""
    opts = KernelOpts(32, 256, 4, TAPE_H32, False, False)
    out, v_last = _TapeGrad.apply(x, leak_q(tau), v0, spikes, tape, v_seq[-1], float(theta), decay_input, opts)
    if return_v_seq:
        return out, v_last, v_seq
    return out, v_last


def _drive(x: Tensor, q: Tensor, decay_input: bool) -> Tensor:
    """The input term u_t of `h_t = beta v_{t-1} + u_t`."""
    return x * q if decay_input else x


def _free_membrane(x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool) -> Tensor:
    """Membrane without any reset, `k_t = beta k_{t-1} + u_t`, by one fused scan."""
    opts = KernelOpts(32, 256, 4, TAPE_H32, False, False)
    vin = v0.reshape(1, -1).contiguous()
    return launch_cascade(x, q, vin, theta, decay_input, opts, CascadeSpec("picard", 1, (1.0,), theta))[1]


@register("cumsum_closed_form", lambda L, **_: 1)
def cumsum_closed_form(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    mode: str = "cummax",
) -> tuple[Tensor, ...]:
    """Spike count as a closed form of a prefix sum (exact only without leak, see METHODS.md)."""
    start = zero_state(x) if v0 is None else v0
    with torch.no_grad():
        q = leak_q(tau)
        if mode == "preleak":
            prefix = _free_membrane(x, q, start, theta, decay_input)
        else:
            prefix = start + torch.cumsum(_drive(x, q, decay_input), dim=0)
        count = torch.floor(prefix / theta)
        if mode in ("cummax", "preleak"):
            count = torch.cummax(count, dim=0).values.clamp_min(0.0)
        before = torch.cat((torch.zeros_like(count[:1]), count[:-1]))
        low = -1.0 if mode == "ternary" else 0.0
        spikes = (count - before).clamp(low, 1.0)
        fired = torch.cumsum(spikes, dim=0)
        tape = prefix - theta * (fired - spikes)
        v_seq = prefix - theta * fired
    return _with_tape_grad(x, tau, theta, start, decay_input, return_v_seq, spikes, tape, v_seq)


@register("sliding_psn", lambda L, k=32, **_: k)
def sliding_psn(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    k: int = 32,
) -> tuple[Tensor, ...]:
    """No reset and only the last k inputs: membrane = sum_j q beta^j x[t-j], j < k."""
    q = leak_q(tau)
    lags = torch.arange(k, device=x.device, dtype=x.dtype)
    taps = (1.0 - q).unsqueeze(1) ** lags  # [D, k], tap j = beta^j
    if decay_input:
        taps = taps * q.unsqueeze(1)
    signal = F.pad(x.permute(1, 2, 0), (k - 1, 0))  # [B, D, k-1+L]
    membrane = F.conv1d(signal, taps.flip(1).unsqueeze(1), groups=x.shape[2]).permute(2, 0, 1)
    spikes = sj_surrogate.ATan(alpha=ALPHA)(membrane - theta)
    if return_v_seq:
        return spikes, membrane[-1], membrane
    return spikes, membrane[-1]


@register("mpe_psn", lambda L, **_: 1)
def mpe_psn(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
) -> tuple[Tensor, ...]:
    """Membrane-potential estimation: the previous membrane is guessed from the previous input alone."""
    q = leak_q(tau)
    drive = _drive(x, q, decay_input)
    generator = torch.Generator(device=x.device).manual_seed(0)
    keep = 1.0 - torch.bernoulli(torch.sigmoid(drive.detach()), generator=generator)
    start = zero_state(x) if v0 is None else v0
    previous = torch.cat((start.unsqueeze(0), (keep * drive)[:-1]))
    membrane = (1.0 - q) * previous + drive
    spikes = sj_surrogate.ATan(alpha=ALPHA)(membrane - theta)
    v_seq = membrane * (1.0 - spikes.detach())
    if return_v_seq:
        return spikes, v_seq[-1], v_seq
    return spikes, v_seq[-1]


_SDN_NETS: dict[tuple[int, int], torch.nn.Module] = {}


def _sdn_net() -> torch.nn.Module:
    """The 175-parameter causal CNN of `sdn_predictor` (the padding is applied by the caller)."""
    return torch.nn.Sequential(
        torch.nn.Conv1d(1, 6, 8), torch.nn.ReLU(), torch.nn.Conv1d(6, 6, 3), torch.nn.ReLU(), torch.nn.Conv1d(6, 1, 1)
    ).cuda()


def _sdn_apply(net: torch.nn.Module, membrane: Tensor) -> Tensor:
    """Predicted reset term for a batch of membrane sequences [n, 1, L]; causal, same length."""
    return net(F.pad(membrane, (9, 0)))  # receptive field 8 + 3 - 1 = 10 tokens


def _sdn_train(membrane: Tensor, target: Tensor, steps: int) -> torch.nn.Module:
    """Fit the CNN to the true reset term. `membrane` and `target` are [L, N]."""
    torch.manual_seed(0)
    net = _sdn_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    generator = torch.Generator(device=membrane.device).manual_seed(0)
    inputs, targets = membrane.t().unsqueeze(1), target.t().unsqueeze(1)
    loss = torch.zeros(())
    for _ in range(steps):
        rows = torch.randint(0, inputs.shape[0], (2048,), device=inputs.device, generator=generator)
        loss = F.mse_loss(_sdn_apply(net, inputs[rows]), targets[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    n_params = sum(p.numel() for p in net.parameters())
    LAST_NOTE.append(f"sdn: {n_params} params, {steps} Adam steps on the real input, final mse {loss.item():.4f}")
    return net.requires_grad_(False)


@register("sdn_predictor", lambda L, **_: 4)
def sdn_predictor(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    steps: int = 600,
) -> tuple[Tensor, ...]:
    """A tiny causal CNN predicts the reset term from the membrane without reset.

    The CNN is trained at the first call for a given width, on the input of that call
    (the benchmark calls the real input first), and reused afterwards.
    """
    start = zero_state(x) if v0 is None else v0
    seq_len, batch, width = x.shape
    with torch.no_grad():
        q = leak_q(tau)
        membrane = _free_membrane(x, q, start, theta, decay_input).reshape(seq_len, -1)
    key = (width, steps)
    if key not in _SDN_NETS:
        with torch.no_grad():
            reference = sj_reference(x, tau, theta, None, v0, decay_input)["h"].reshape(seq_len, -1)
        with torch.enable_grad():
            _SDN_NETS[key] = _sdn_train(membrane, membrane - reference, steps)
    with torch.no_grad():
        reset = _sdn_apply(_SDN_NETS[key], membrane.t().unsqueeze(1)).squeeze(1).t()
        tape = (membrane - reset).reshape(x.shape)
        spikes = (tape >= theta).to(x.dtype)
        v_seq = tape - theta * spikes
    return _with_tape_grad(x, tau, theta, start, decay_input, return_v_seq, spikes, tape, v_seq)


@register("speculative_chunk", lambda L, **_: L)
def speculative_chunk(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 64,
) -> tuple[Tensor, ...]:
    """Scan a chunk without reset, reset every neuron at its first crossing, rescan from there.

    Inside a chunk the membrane is `k_t = S_t + beta^t A`, with S the membrane from a zero
    start and A one number per neuron. A reset at step t* only changes A by
    `-theta beta^-t*`, so a round is a handful of whole-chunk operations.
    """
    start = zero_state(x) if v0 is None else v0
    seq_len, batch, width = x.shape
    rounds = 0
    with torch.no_grad():
        q = leak_q(tau)
        beta = (1.0 - q).double()
        steps = torch.arange(C, device=x.device, dtype=torch.float64)
        grow = (beta.unsqueeze(0) ** steps.unsqueeze(1)).unsqueeze(1)  # [C, 1, D], beta^t
        shrink = 1.0 / grow
        index = torch.arange(C, device=x.device).view(C, 1, 1)
        drive = _drive(x, q, decay_input).double()
        v = start.double()
        spike_parts, tape_parts = [], []
        for first in range(0, seq_len, C):
            free = grow * torch.cumsum(drive[first : first + C] * shrink, dim=0)  # S_t
            offset = beta * v  # A at the chunk start
            position = torch.zeros_like(v, dtype=torch.long)  # first step not yet decided
            spikes = torch.zeros_like(free)
            while True:
                rounds += 1
                crossing = ((free + grow * offset) >= theta) & (index >= position)
                hit = crossing.any(dim=0)
                if not bool(hit.any()):
                    break
                where = crossing.to(torch.uint8).argmax(dim=0)  # first crossing per neuron
                spikes.scatter_add_(0, where.unsqueeze(0), hit.to(spikes.dtype).unsqueeze(0))
                offset = offset - theta * hit * torch.gather(shrink.expand_as(free), 0, where.unsqueeze(0))[0]
                position = torch.where(hit, where + 1, position)
            earlier = torch.cumsum(spikes * shrink, dim=0) - spikes * shrink  # resets before step t
            tape = free + grow * (beta * v - theta * earlier)
            v = tape[-1] - theta * spikes[-1]
            spike_parts.append(spikes)
            tape_parts.append(tape)
        spikes = torch.cat(spike_parts).to(x.dtype)
        tape = torch.cat(tape_parts).to(x.dtype)
        v_seq = tape - theta * spikes
    LAST_NOTE.append(f"rounds={rounds}")
    return _with_tape_grad(x, tau, theta, start, decay_input, return_v_seq, spikes, tape, v_seq)


# --------------------------------------------------------------------------------------
# C. Carry-chain methods: every chunk is run from K candidate entry states
# --------------------------------------------------------------------------------------


@dataclass
class ChunkMaps:
    """Result of pass 1 of the candidate methods, all tensors [K, n_chunks, N].

    guess: the candidate entry states. end: the end state of the chunk for each candidate.
    lo, hi: offsets from the candidate for which the chunk has the same spikes.
    slope: beta^C per neuron, [N].
    """

    guess: Tensor
    end: Tensor
    lo: Tensor
    hi: Tensor
    slope: Tensor


def _candidate_maps(
    x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, opts: KernelOpts,
    chunk: int, warmup: int, n_cand: int, spread: float,
) -> ChunkMaps:  # fmt: skip
    """Pass 1: run every chunk from `n_cand` candidates around a warm-up guess; no spikes stored.

    Candidate k is the warm-up guess plus `spread * theta * (k - centre)`.
    """
    n_chunks = x.shape[0] // chunk
    centre = (n_cand - 1) // 2
    bounds: list[Tensor] = []
    quiet = {"spikes": False, "tape": TAPE_NONE}
    first = launch_fwd(x, q, _first_row(v0, n_chunks), theta, decay_input, opts, chunk, warmup, bounds=bounds, **quiet)
    guesses, ends = [first[3]], [first[4]]
    order = [centre]
    for k in range(n_cand):
        if k == centre:
            continue
        shifted = first[3] + spread * theta * (k - centre)
        run = launch_fwd(x, q, shifted, theta, decay_input, opts, chunk, 0, bounds=bounds, **quiet)
        guesses.append(run[3])
        ends.append(run[4])
        order.append(k)
    rank = torch.tensor(order).argsort().tolist()  # put the candidates back in offset order
    stack = lambda parts: torch.stack([parts[i] for i in rank])  # noqa: E731
    slope = _chunk_slope(q, chunk).repeat(x.shape[1])
    return ChunkMaps(stack(guesses), stack(ends), stack(bounds[0::2]), stack(bounds[1::2]), slope)


def _replay(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor, decay_input: bool, return_v_seq: bool,
    opts: KernelOpts, chunk: int, entries: Tensor, repair: bool,
) -> tuple[Tensor, ...]:  # fmt: skip
    """Pass 2: run all chunks from the given entry states, optionally repair, attach the backward pass."""
    with torch.no_grad():
        q = leak_q(tau)
        bounds: list[Tensor] | None = [] if repair else None
        spikes, tape, v_seq, vstart, vout = launch_fwd(x, q, entries, theta, decay_input, opts, chunk, bounds=bounds)
        if repair:
            n_fixed = launch_repair(x, q, theta, decay_input, opts, chunk, spikes, tape, v_seq, vstart, vout, bounds)
            if opts.store_v:
                LAST_NOTE[-1:] = [(LAST_NOTE[-1] + ", " if LAST_NOTE else "") + f"rerun {int(n_fixed.sum())} blocks"]
        v_last = vout[-1].view_as(v0)
    grad_opts = KernelOpts(32, 256, 4, opts.tape, False, False)
    out, v_last = _TapeGrad.apply(x, leak_q(tau), v0, spikes, tape, v_last, float(theta), decay_input, grad_opts)
    if return_v_seq:
        return out, v_last, v_seq
    return out, v_last


def _nearest(points: Tensor, guesses: Tensor) -> Tensor:
    """Index of the candidate in `guesses` [K, ...] that is nearest to `points` [...]."""
    return (points.unsqueeze(0) - guesses).abs().argmin(dim=0)


def _pick(table: Tensor, index: Tensor) -> Tensor:
    """table[index] along dim 0, for `table` [K, ...] and `index` [...]."""
    return table.gather(0, index.unsqueeze(0)).squeeze(0)


@register("carry_select", lambda L, C=256, W=32, **_: W + C + L // C + C)
def carry_select(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 256,
    K: int = 3,
    W: int = 32,
    spread: float = 0.5,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """K candidate carries per chunk, then one sweep over the chunks that selects the nearest."""
    start = zero_state(x) if v0 is None else v0
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    with torch.no_grad():
        maps = _candidate_maps(x, leak_q(tau), start, theta, decay_input, opts, C, W, K, spread)
        state = start.reshape(-1)
        entries = [state]
        for c in range(maps.guess.shape[1] - 1):
            k = _nearest(state, maps.guess[:, c])
            state = _pick(maps.end[:, c], k) + maps.slope * (state - _pick(maps.guess[:, c], k))
            entries.append(state)
    return _replay(x, tau, theta, start, decay_input, return_v_seq, opts, C, torch.stack(entries), repair=False)


@register("carry_lookahead", lambda L, C=256, W=32, **_: W + C + math.ceil(math.log2(max(L // C, 2))) + C)
def carry_lookahead(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 256,
    K: int = 3,
    W: int = 32,
    spread: float = 0.5,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """K candidates per chunk; candidate-index tables composed by a prefix scan over the chunks."""
    start = zero_state(x) if v0 is None else v0
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    with torch.no_grad():
        maps = _candidate_maps(x, leak_q(tau), start, theta, decay_input, opts, C, W, K, spread)
        n_chunks = maps.guess.shape[1]
        # table[k, c] = candidate of chunk c+1 that is nearest to the end state of candidate k of chunk c
        gaps = maps.end[:, :-1].unsqueeze(1) - maps.guess[:, 1:].unsqueeze(0)  # [K(out), K(next), n-1, N]
        table = gaps.abs().argmin(dim=0) if K == 1 else gaps.abs().permute(1, 0, 2, 3).argmin(dim=0)
        step = 1
        while step < n_chunks - 1:  # Hillis-Steele scan over function composition
            merged = table.clone()
            merged[:, step:] = table[:, step:].gather(0, table[:, :-step])
            table, step = merged, step * 2
        first = _nearest(start.reshape(-1), maps.guess[:, 0])
        index = table.gather(0, first.view(1, 1, -1).expand(1, n_chunks - 1, -1)).squeeze(0)  # [n-1, N]
        later = maps.guess[:, 1:].gather(0, index.unsqueeze(0)).squeeze(0)
        entries = torch.cat((start.reshape(1, -1), later))
    return _replay(x, tau, theta, start, decay_input, return_v_seq, opts, C, entries, repair=False)


def _compose(first: ChunkMaps, second: ChunkMaps) -> tuple[ChunkMaps, Tensor]:
    """Pieces of "first, then second". Also returns which pieces found no valid piece of `second`.

    A piece is the affine map `end + slope * (v - guess)` on `guess + [lo, hi)`. The image
    of the anchor of a piece of `first` must fall into a piece of `second`; the new
    interval is the part of the old one whose image stays inside that piece.
    """
    point = first.end.unsqueeze(0) - second.guess.unsqueeze(1)  # [K2, K1, ...]: offset of the image in piece j
    inside = (point >= second.lo.unsqueeze(1)) & (point < second.hi.unsqueeze(1))
    found = inside.any(dim=0)
    j = torch.where(found, inside.to(torch.uint8).argmax(dim=0), point.abs().argmin(dim=0))  # [K1, ...]
    offset = point.gather(0, j.unsqueeze(0)).squeeze(0)
    lo2, hi2, end2 = (t.gather(0, j) for t in (second.lo, second.hi, second.end))
    steep = first.slope > 0
    safe = torch.where(steep, first.slope, torch.ones_like(first.slope))
    lo = torch.where(steep, torch.maximum(first.lo, (lo2 - offset) / safe), first.lo)
    hi = torch.where(steep, torch.minimum(first.hi, (hi2 - offset) / safe), first.hi)
    lo = torch.where(found, lo, torch.full_like(lo, 3.0e38))  # empty interval: this piece is not certified
    end = end2 + second.slope * offset
    return ChunkMaps(first.guess, end, lo, hi, first.slope * second.slope), ~found


@register("map_scan", lambda L, C=256, W=32, **_: W + C + math.ceil(math.log2(max(L // C, 2))) + C)
def map_scan(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    C: int = 256,
    K: int = 1,
    W: int = 32,
    spread: float = 0.5,
    repair: int = 1,
    U: int = 16,
    BLOCK: int = 256,
    nw: int = 4,
    tape: str = "h16",
) -> tuple[Tensor, ...]:
    """Piecewise-affine chunk maps with validity intervals, composed by a prefix scan over chunks."""
    start = zero_state(x) if v0 is None else v0
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), False, return_v_seq)
    with torch.no_grad():
        maps = _candidate_maps(x, leak_q(tau), start, theta, decay_input, opts, C, W, K, spread)
        n_chunks = maps.guess.shape[1]
        maps.slope = maps.slope.expand(n_chunks, -1)
        step = 1
        while step < n_chunks:  # after the scan, piece set c is the map of chunks 0..c
            head = ChunkMaps(*(t[:, :-step] for t in (maps.guess, maps.end, maps.lo, maps.hi)), maps.slope[:-step])
            tail = ChunkMaps(*(t[:, step:] for t in (maps.guess, maps.end, maps.lo, maps.hi)), maps.slope[step:])
            joined, _ = _compose(head, tail)
            maps = ChunkMaps(
                *(torch.cat((old[:, :step], new), dim=1) for old, new in (
                    (maps.guess, joined.guess), (maps.end, joined.end), (maps.lo, joined.lo), (maps.hi, joined.hi)
                )),
                torch.cat((maps.slope[:step], joined.slope)),
            )  # fmt: skip
            step *= 2
        offset = start.reshape(1, 1, -1) - maps.guess  # the true first state, seen from every piece
        inside = (offset >= maps.lo) & (offset < maps.hi)
        k = torch.where(inside.any(dim=0), inside.to(torch.uint8).argmax(dim=0), offset.abs().argmin(dim=0))
        ends = _pick(maps.end, k) + maps.slope * _pick(offset, k)  # [n_chunks, N]: true end state of chunk c
        entries = torch.cat((start.reshape(1, -1), ends[:-1]))
        if return_v_seq:
            uncertified = int((~inside.any(dim=0)).sum())
            LAST_NOTE.append(f"uncertified entries {uncertified} of {inside[0].numel()}")
    return _replay(x, tau, theta, start, decay_input, return_v_seq, opts, C, entries, repair=bool(repair))


if __name__ == "__main__":
    # Smoke check: every registered method on a small random input, against the reference.
    torch.manual_seed(0)
    demo_len = 512
    x_demo = torch.randn(demo_len, 2, 8, device="cuda") * 2.0
    tau_demo = torch.full((8,), 2.0, device="cuda")
    ref = sj_reference(x_demo, tau_demo, 1.0)
    for name, method in METHODS.items():
        spikes_demo, v_demo = method.fn(x_demo, tau_demo, 1.0)[:2]
        agree = (spikes_demo == ref["spikes"]).float().mean().item()
        depth = method.seq_depth(demo_len)
        print(f"{name:24s} spikes {tuple(spikes_demo.shape)} agree {agree:.4f} depth {depth}")
        assert math.isfinite(v_demo.sum().item())
