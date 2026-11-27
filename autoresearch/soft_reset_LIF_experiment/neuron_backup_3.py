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
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from spikingjelly._ops.plif import triton as sj_triton
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import surrogate as sj_surrogate
from torch import Tensor

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
) -> dict[str, Tensor]:
    """Exact serial result from SpikingJelly's fused Triton kernels, launched as it launches them.

    Returns `spikes`, `v_seq` (membrane after reset, [L, B, D]), `v_last` and `h`. With `r`
    (the gradient of the loss w.r.t. the spikes) it also returns `gx` and `gtau`, the
    gradients w.r.t. `x` and w.r.t. a per-channel `tau`. SpikingJelly sums its per-neuron
    leak gradient over all neurons because its leak is one scalar; here the same kernel
    output is summed over the batch only, which gives the per-channel gradient.
    """
    w = _uniform_scalar(leak_w(tau.detach()))
    v_init = torch.zeros_like(x[0]) if v0 is None else v0
    spikes, v_seq, h = sj_triton._forward_impl(
        x.detach(), v_init, w, decay_input, theta, None, True, ALPHA, True, ATAN_ID
    )
    out = {"spikes": spikes, "v_seq": v_seq, "v_last": v_seq[-1], "h": h}
    if r is None:
        return out
    q = w.float().sigmoid()
    gx = torch.empty_like(h)
    gv_init = torch.empty_like(v_init)
    gq = torch.empty_like(v_init)
    neurons = v_init.numel()
    sj_triton._backward_kernel[(triton.cdiv(neurons, SJ_BLOCK),)](
        r, torch.zeros_like(v_init), x.detach(), v_init, h, gx, gv_init, gq,
        x.shape[0], neurons, q, theta, 0.0, ALPHA,
        decay_input, True, True, ATAN_ID, False, SJ_BLOCK, None,
        enable_fp_fusion=False,
    )  # fmt: skip
    q_ch = leak_q(tau.detach())
    out["gx"] = gx
    out["gtau"] = -gq.sum(dim=0) * q_ch * q_ch  # dq/dtau = -q^2
    return out


# --------------------------------------------------------------------------------------
# A. Serial methods
# --------------------------------------------------------------------------------------

_SJ_NODES: dict[tuple[float, bool, bool], sj_neuron.ParametricLIFNode] = {}


def _sj_node(theta: float, decay_input: bool, store_v_seq: bool) -> sj_neuron.ParametricLIFNode:
    """The repo's PLIF module with a soft reset, cached per setting."""
    key = (theta, decay_input, store_v_seq)
    if key not in _SJ_NODES:
        _SJ_NODES[key] = sj_neuron.ParametricLIFNode(
            init_tau=2.0,
            decay_input=decay_input,
            v_threshold=theta,
            v_reset=None,
            surrogate_function=sj_surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
            store_v_seq=store_v_seq,
        ).cuda()
    return _SJ_NODES[key]


def _sj_module(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor | None, decay_input: bool, return_v_seq: bool
) -> tuple[Tensor, ...]:
    """Call SpikingJelly's ParametricLIFNode in multi-step mode, as the repo does."""
    node = _sj_node(float(theta), decay_input, return_v_seq)
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
) -> tuple[Tensor, ...]:
    """The reference: SpikingJelly's multi-step PLIF, one fused kernel that loops over L.

    SpikingJelly 2.0 has no cupy backend; its fused kernel is the Triton provider, which
    it selects by itself on this machine. The leak is one scalar per layer.
    """
    return _sj_module(x, tau, theta, v0, decay_input, return_v_seq)


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
    v = torch.zeros_like(x[0]) if v0 is None else v0
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

TAPE_NONE, TAPE_H32, TAPE_H16 = 0, 1, 2
_TAPE_DTYPE = {TAPE_H32: torch.float32, TAPE_H16: torch.float16}
_KERNELS: dict[tuple[str, int], object] = {}


def _jit(name: str, source: str) -> object:
    """Compile generated Triton source. `linecache` lets Triton read the source back."""
    filename = f"<generated {name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"triton": triton, "tl": tl}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - our own generated kernel text
    return namespace[name]


def _fwd_source(unroll: int) -> str:
    """Source of the forward kernel that handles `unroll` time steps per loop iteration."""
    steps = range(unroll)

    def body(indent: str, store: bool) -> list[str]:
        lines = [f"{indent}x{j} = tl.load(x_ptr + base + {j} * stride, mask=mask, other=0.0)" for j in steps]
        for j in steps:
            lines += [
                f"{indent}if DECAY:",
                f"{indent}    h{j} = v + (x{j} - v) * q",
                f"{indent}else:",
                f"{indent}    h{j} = v - v * q + x{j}",
                f"{indent}s{j} = (h{j} >= theta).to(tl.float32)",
                f"{indent}v = h{j} - s{j} * theta",
                f"{indent}v{j} = v",
            ]
        if not store:
            return lines
        lines.append(f"{indent}if SPIKES:")
        lines += [f"{indent}    tl.store(s_ptr + base + {j} * stride, s{j}, mask=mask)" for j in steps]
        lines.append(f"{indent}if TAPE == 1:")
        lines += [f"{indent}    tl.store(tape_ptr + base + {j} * stride, h{j}, mask=mask)" for j in steps]
        lines.append(f"{indent}if TAPE == 2:")
        lines += [
            f"{indent}    tl.store(tape_ptr + base + {j} * stride, h{j}.to(tl.float16), mask=mask)" for j in steps
        ]
        lines.append(f"{indent}if STORE_V:")
        lines += [f"{indent}    tl.store(vseq_ptr + base + {j} * stride, v{j}, mask=mask)" for j in steps]
        return lines

    lines = [
        "@triton.jit",
        "def lif_fwd(x_ptr, s_ptr, tape_ptr, vseq_ptr, q_ptr, vin_ptr, vstart_ptr, vout_ptr, C, W, N, D, theta,",
        "            DECAY: tl.constexpr, TAPE: tl.constexpr, SPIKES: tl.constexpr, STORE_V: tl.constexpr,",
        "            BLOCK: tl.constexpr):",
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
        *body("        ", store=False),
        "    tl.store(vstart_ptr + chunk * stride + n, v, mask=mask)",
        f"    for g in range(C // {unroll}):",
        f"        base = n + (chunk * C + g * {unroll}) * stride",
        *body("        ", store=True),
        "    tl.store(vout_ptr + chunk * stride + n, v, mask=mask)",
    ]
    return "\n".join(lines) + "\n"


def _bwd_source(unroll: int) -> str:
    """Source of the backward kernel (reverse loop over one chunk, `unroll` steps at a time)."""
    steps = range(unroll)
    ind = "        "
    lines = [
        "@triton.jit",
        "def lif_bwd(gs_ptr, tape_ptr, x_ptr, gx_ptr, q_ptr, vin_ptr, cin_ptr, cout_ptr, gacc_ptr,",
        "            C, W, NCHUNK, N, D, theta, alpha,",
        "            DECAY: tl.constexpr, TAPE: tl.constexpr, USE_X: tl.constexpr, STORE: tl.constexpr,",
        "            BLOCK: tl.constexpr):",
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
        f"    warm = tl.minimum((NCHUNK - 1 - chunk) * C, W) // {unroll}",
        "    for g in range(warm):",
        f"        base = n + (last + (warm - g) * {unroll}) * stride",
        *[f"        gw{j} = tl.load(gs_ptr + base - {j} * stride, mask=mask, other=0.0)" for j in steps],
        *[
            f"        hw{j} = tl.load(tape_ptr + base - {j} * stride, mask=mask, other=0.0).to(tl.float32)"
            for j in steps
        ],
        *[
            line
            for j in steps
            for line in (
                f"        z = (hw{j} - theta) * (1.5707963267948966 * alpha)",
                f"        gh = gw{j} * ((0.5 * alpha) / (1.0 + z * z)) + carry",
                "        carry = gh - gh * q",
            )
        ],
        "    h_cur = tl.load(tape_ptr + n + last * stride, mask=mask, other=0.0).to(tl.float32)",
        f"    for g in range(C // {unroll}):",
        f"        t_hi = last - g * {unroll}",
        "        base = n + t_hi * stride",
    ]
    lines += [f"{ind}gs{j} = tl.load(gs_ptr + base - {j} * stride, mask=mask, other=0.0)" for j in steps]
    lines += [
        f"{ind}hp{j} = tl.load(tape_ptr + base - {j + 1} * stride, mask=mask & (t_hi - {j} > first), other=0.0)"
        ".to(tl.float32)"
        for j in steps
    ]
    lines.append(f"{ind}if USE_X:")
    lines += [f"{ind}    x{j} = tl.load(x_ptr + base - {j} * stride, mask=mask, other=0.0)" for j in steps]
    for j in steps:
        lines += [
            f"{ind}z = (h_cur - theta) * (1.5707963267948966 * alpha)",
            f"{ind}gh = gs{j} * ((0.5 * alpha) / (1.0 + z * z)) + carry",
            f"{ind}if DECAY:",
            f"{ind}    gx{j} = gh * q",
            f"{ind}else:",
            f"{ind}    gx{j} = gh",
            f"{ind}carry = gh - gh * q",
            f"{ind}vp = hp{j} - (hp{j} >= theta).to(tl.float32) * theta",
            f"{ind}vp = tl.where(t_hi - {j} > first, vp, vin)",
            f"{ind}if DECAY:",
            f"{ind}    if USE_X:",
            f"{ind}        gacc += gh * (x{j} - vp)",
            f"{ind}    else:",
            f"{ind}        gacc += gh * (h_cur - vp)",
            f"{ind}else:",
            f"{ind}    gacc -= gh * vp",
            f"{ind}h_cur = hp{j}",
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
    x_t - v_{t-1} = (h_t - v_{t-1}) / q from the tape.
    """

    unroll: int = 8
    block: int = 256
    warps: int = 4
    tape: int = TAPE_H32
    use_x: bool = False
    store_v: bool = False


def launch_fwd(
    x: Tensor, q: Tensor, vin: Tensor, theta: float, decay_input: bool, opts: KernelOpts,
    chunk: int, warmup: int = 0, spikes: bool = True, tape: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """Run the forward kernel over all chunks of `chunk` steps.

    `vin` is [n_chunks, B*D], the membrane each chunk starts from. With `warmup` W, every
    chunk after the first starts W steps early and stores nothing for those steps.
    Returns (spikes, tape, v_seq, vstart, vout): `vstart` is the membrane entering the
    stored part of each chunk, `vout` the membrane leaving it. Tensors that were not
    asked for are empty.
    """
    seq_len, batch, width = x.shape
    neurons = batch * width
    n_chunks = seq_len // chunk
    if seq_len % chunk or chunk % opts.unroll or warmup % opts.unroll:
        raise ValueError(f"L={seq_len}, C={chunk}, W={warmup} must be multiples of C and of U={opts.unroll}")
    tape_mode = opts.tape if tape is None else tape
    empty = x.new_empty(0)
    s_out = torch.empty_like(x) if spikes else empty
    tape_out = torch.empty(x.shape, dtype=_TAPE_DTYPE[tape_mode], device=x.device) if tape_mode else empty
    v_seq = torch.empty_like(x) if opts.store_v else empty
    vstart = torch.empty_like(vin)
    vout = torch.empty_like(vin)
    _kernel("fwd", opts.unroll)[(triton.cdiv(neurons, opts.block), n_chunks)](
        x, s_out, tape_out, v_seq, q, vin, vstart, vout, chunk, warmup, neurons, width, theta,
        decay_input, tape_mode, spikes, opts.store_v, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
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
    tape_mode = TAPE_H16 if tape.dtype == torch.float16 else TAPE_H32
    gx = torch.empty_like(gs) if store else gs.new_empty(0)
    cout = torch.empty_like(cin)
    gacc = torch.empty_like(cin)
    _kernel("bwd", opts.unroll)[(triton.cdiv(neurons, opts.block), n_chunks)](
        gs, tape, x, gx, q, vin, cin, cout, gacc, chunk, warmup, n_chunks, neurons, width, theta, ALPHA,
        decay_input, tape_mode, opts.use_x, store, opts.block,
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
    """"h32" or "h16" -> the kernel's tape code."""
    return {"h32": TAPE_H32, "h16": TAPE_H16}[name]


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
) -> tuple[Tensor, ...]:
    """Exact serial neuron in an own fused kernel that loads U time steps ahead.

    `tape="h16"` saves the charged membrane for the backward pass in float16 (the forward
    pass stays float32 and exact). `usex=1` makes the backward pass read x again, as
    SpikingJelly does, instead of taking the leak gradient from the tape.
    """
    opts = KernelOpts(U, BLOCK, nw, _tape_mode(tape), bool(usex), return_v_seq)
    start = torch.zeros_like(x[0]) if v0 is None else v0
    spikes, v_last, v_seq = _SerialKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts)
    if return_v_seq:
        return spikes, v_last, v_seq
    return spikes, v_last


# --------------------------------------------------------------------------------------
# A. Chunked methods: every chunk of C tokens is its own program in one kernel launch
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPlan:
    """How a chunked method finds the state that enters each chunk.

    mode "cold": start `warmup` tokens early from zero. mode "picard": `passes` passes,
    each chunk starting from the end state its neighbour had in the pass before.
    The backward pass uses the same rule for the gradient carry.
    """

    chunk: int
    mode: str
    warmup: int = 0
    passes: int = 1


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
        if plan.mode == "picard":
            for _ in range(plan.passes - 1):
                ends = launch_fwd(x, q, vin, theta, decay_input, opts, plan.chunk, spikes=False, tape=TAPE_NONE)[4]
                vin = torch.cat((vin[:1], ends[:-1]))
        spikes, tape, v_seq, vstart, vout = launch_fwd(x, q, vin, theta, decay_input, opts, plan.chunk, plan.warmup)
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
        args = (gs, ctx.tape, x, q, ctx.vstart)
        cin = _last_row(gv_last, n_chunks)
        if plan.mode == "picard":
            for _ in range(plan.passes - 1):
                cout = launch_bwd(*args, cin, ctx.theta, ctx.decay_input, opts, plan.chunk, store=False)[1]
                cin = torch.cat((cout[1:], cin[-1:]))
        gx, cout, gq = launch_bwd(*args, cin, ctx.theta, ctx.decay_input, opts, plan.chunk, plan.warmup)
        return gx, gq.sum(0).view(batch, width).sum(0), cout[0].view(batch, width), None, None, None, None


def _chunked(
    x: Tensor, tau: Tensor, theta: float, v0: Tensor | None, decay_input: bool, return_v_seq: bool,
    opts: KernelOpts, plan: ChunkPlan,
) -> tuple[Tensor, ...]:  # fmt: skip
    """Shared wrapper of the chunked methods."""
    start = torch.zeros_like(x[0]) if v0 is None else v0
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
    start = torch.zeros_like(x[0]) if v0 is None else v0
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
    start = torch.zeros_like(x[0]) if v0 is None else v0
    spikes, v_last, v_seq = _PrfKernel.apply(x, leak_q(tau), start, float(theta), decay_input, opts)
    if return_v_seq:
        return spikes, v_last, v_seq
    return spikes, v_last


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
