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
        "def lif_fwd(x_ptr, s_ptr, tape_ptr, vseq_ptr, q_ptr, vin_ptr, vout_ptr, C, W, N, D, theta,",
        "            DECAY: tl.constexpr, TAPE: tl.constexpr, SPIKES: tl.constexpr, STORE_V: tl.constexpr,",
        "            BLOCK: tl.constexpr):",
        "    theta = tl.cast(theta, tl.float32)",
        "    chunk = tl.program_id(1)",
        "    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)",
        "    mask = n < N",
        "    stride = tl.cast(N, tl.int64)",
        "    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)",
        "    v = tl.load(vin_ptr + chunk * stride + n, mask=mask, other=0.0)",
        "    # Warm-up: chunks after the first start W steps early and store nothing there.",
        f"    warm = tl.minimum(chunk, 1) * (W // {unroll})",
        f"    start = chunk * C - warm * {unroll}",
        "    for g in range(warm):",
        f"        base = n + (start + g * {unroll}) * stride",
        *body("        ", store=False),
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
        "            C, N, D, theta, alpha,",
        "            DECAY: tl.constexpr, TAPE: tl.constexpr, USE_X: tl.constexpr, BLOCK: tl.constexpr):",
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
    lines += [f"{ind}tl.store(gx_ptr + base - {j} * stride, gx{j}, mask=mask)" for j in steps]
    lines += [
        "    tl.store(cout_ptr + chunk * stride + n, carry, mask=mask)",
        "    tl.store(gacc_ptr + chunk * stride + n, gacc, mask=mask)",
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """Run the forward kernel over all chunks of `chunk` steps.

    `vin` is [n_chunks, B*D], the membrane entering each chunk. Returns
    (spikes, tape, v_seq, vout); tensors that were not asked for are empty.
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
    vout = torch.empty_like(vin)
    _kernel("fwd", opts.unroll)[(triton.cdiv(neurons, opts.block), n_chunks)](
        x, s_out, tape_out, v_seq, q, vin, vout, chunk, warmup, neurons, width, theta,
        decay_input, tape_mode, spikes, opts.store_v, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
    return s_out, tape_out, v_seq, vout


def launch_bwd(
    gs: Tensor, tape: Tensor, x: Tensor, q: Tensor, vin: Tensor, cin: Tensor, theta: float,
    decay_input: bool, opts: KernelOpts, chunk: int,
) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    """Run the backward kernel over all chunks. `cin` is the carry entering each chunk from the future.

    Returns (gx, cout, gq): the gradient w.r.t. x, the carry leaving each chunk towards
    the past ([n_chunks, B*D]) and the gradient w.r.t. q per chunk and neuron.
    """
    seq_len, batch, width = gs.shape
    neurons = batch * width
    tape_mode = TAPE_H16 if tape.dtype == torch.float16 else TAPE_H32
    gx = torch.empty_like(gs)
    cout = torch.empty_like(cin)
    gacc = torch.empty_like(cin)
    _kernel("bwd", opts.unroll)[(triton.cdiv(neurons, opts.block), seq_len // chunk)](
        gs, tape, x, gx, q, vin, cin, cout, gacc, chunk, neurons, width, theta, ALPHA,
        decay_input, tape_mode, opts.use_x, opts.block,
        num_warps=opts.warps, enable_fp_fusion=False,
    )  # fmt: skip
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
        spikes, tape, v_seq, vout = launch_fwd(x, q, vin, theta, decay_input, opts, chunk=seq_len)
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


if __name__ == "__main__":
    # Smoke check: every registered method on a small random input, against the reference.
    torch.manual_seed(0)
    x_demo = torch.randn(64, 2, 8, device="cuda") * 2.0
    tau_demo = torch.full((8,), 2.0, device="cuda")
    ref = sj_reference(x_demo, tau_demo, 1.0)
    for name, method in METHODS.items():
        spikes_demo, v_demo = method.fn(x_demo, tau_demo, 1.0)[:2]
        agree = (spikes_demo == ref["spikes"]).float().mean().item()
        print(f"{name:24s} spikes {tuple(spikes_demo.shape)} agree {agree:.4f} depth {method.seq_depth(64)}")
    assert math.isfinite(v_demo.sum().item())
