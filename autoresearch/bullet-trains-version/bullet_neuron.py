"""Bullet Trains (Morrill, Pehle, Zador, ICML 2026, arXiv:2603.13283) for the discrete hard-reset LIF.

The paper's forward pass, translated from continuous-time events to the discrete neuron of
this repo: every neuron takes the next K time steps, computes all their membranes at once
with an associative scan over affine maps, finds the first threshold crossing, commits the
steps up to it, resets the membrane to 0 and starts the next round right after the spike.
Without a spike in the K steps it commits all K. No spike cap: every step is processed, so
the forward pass follows the serial neuron (up to the rounding order of the scan).

Two backward passes:
  bwd=serial  the serial backward kernel of `serial_triton` (the soft-reset study), so only
              the forward pass differs from the baseline;
  bwd=scan    a chunked reverse associative scan of the linear backward recurrence: chunks
              of K steps one after the other, a scan inside each chunk (depth L/K * log2 K).

The methods are registered into the registry of `../soft_reset_LIF_experiment/neuron.py`, so
that study's harness (`bench_neuron.py`) runs them with its gate. Nothing in that folder is
changed; `bench_bullet.py` is the entry point.
"""

import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ in the other study's folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soft_reset_LIF_experiment"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.autograd.function import once_differentiable  # noqa: E402

import neuron  # noqa: E402  (the soft-reset study's module: registry, reference, serial kernels)

TAPES = {"h32": torch.float32, "h16": torch.float16, "u8": torch.uint8}
TAPE_CODE = {"h32": 1, "h16": 2, "u8": 3}
U8_EXP_SHIFT = 118  # the same 8-bit code as `serial_triton tape=u8` (sign, 4 exponent, 3 mantissa bits)


@triton.jit
def _then(a1, b1, a2, b2):
    """Compose two affine maps v -> a v + b: first (a1, b1), then (a2, b2)."""
    return a1 * a2, a2 * b1 + b2


@triton.jit
def _tape_offset(t, n, N, TAPE: tl.constexpr):
    """Element offset of step t, neuron n in the tape. The 8-bit tape is the packed int32 layout
    of `serial_triton` seen as bytes: byte t % 4 of word (t // 4, n)."""
    if TAPE == 3:
        return ((t // 4) * N + n) * 4 + t % 4
    else:
        return t * N + n


@triton.jit
def _encode_u8(h, theta):
    """8-bit code of h - theta, bit-identical to the encoder of `serial_triton`."""
    bits = (h - theta).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 118
    m = (bits >> 20) & 7
    m = tl.where(e < 0, 0, tl.where(e > 15, 7, m))
    e = tl.minimum(tl.maximum(e, 0), 15)
    return ((((bits >> 31) & 1) << 7) | (e << 3) | m).to(tl.uint8)


@triton.jit
def _decode_u8(code, theta):
    """h from an 8-bit code, bit-identical to the decoder of `serial_triton`."""
    c = code.to(tl.int32)
    bits = ((((c >> 3) & 15) + 118) << 23) | ((c & 7) << 20) | 524288 | ((c >> 7) << 31)
    return bits.to(tl.float32, bitcast=True) + theta


@triton.jit
def bullet_fwd(
    x_ptr, s_ptr, tape_ptr, vseq_ptr, v0_ptr, vout_ptr, q_ptr, rounds_ptr, visited_ptr,
    L, N, D, theta,
    DECAY: tl.constexpr, TAPE: tl.constexpr, STORE_V: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """One program = BLOCK neurons, each with its own position in time. Round: scan the next K
    steps from the current membrane, commit up to the first spike (or all K), reset, repeat."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)
    keep = 1.0 - q
    v = tl.load(v0_ptr + n, mask=mask, other=0.0)
    pos = tl.where(mask, 0, L)
    j = tl.arange(0, K)
    rounds = tl.zeros((BLOCK,), tl.int32)
    visited = tl.zeros((BLOCK,), tl.int32)
    while tl.max(L - pos, axis=0) > 0:
        t = pos[None, :] + j[:, None]
        valid = (t < L) & mask[None, :]
        off = t.to(tl.int64) * N + n[None, :]
        xk = tl.load(x_ptr + off, mask=valid, other=0.0).to(tl.float32)
        a = tl.broadcast_to(keep[None, :], (K, BLOCK))
        if DECAY:
            b = q[None, :] * xk
        else:
            b = xk
        A, Bc = tl.associative_scan((a, b), axis=0, combine_fn=_then)
        h = A * v[None, :] + Bc
        cross = (h >= theta) & valid
        first = tl.min(tl.where(cross, j[:, None], K), axis=0)
        left = tl.minimum(L - pos, K)
        n_commit = tl.where(first < K, first + 1, left)
        commit = (j[:, None] < n_commit[None, :]) & valid
        spike = (j[:, None] == first[None, :]) & commit
        tl.store(s_ptr + off, spike.to(tl.float32), mask=commit)
        toff = _tape_offset(t.to(tl.int64), n[None, :], N, TAPE)
        if TAPE == 1:
            tl.store(tape_ptr + toff, h, mask=commit)
        elif TAPE == 2:
            tl.store(tape_ptr + toff, h.to(tl.float16), mask=commit)
        elif TAPE == 3:
            tl.store(tape_ptr + toff, _encode_u8(h, theta), mask=commit)
        if STORE_V:
            tl.store(vseq_ptr + off, tl.where(spike, 0.0, h), mask=commit)
        h_end = tl.sum(tl.where(j[:, None] == (n_commit - 1)[None, :], h, 0.0), axis=0)
        active = pos < L
        v = tl.where(active, tl.where(first < K, 0.0, h_end), v)
        rounds += active.to(tl.int32)
        visited += left
        pos += n_commit
    tl.store(vout_ptr + n, v, mask=mask)
    tl.store(rounds_ptr + n, rounds, mask=mask)
    tl.store(visited_ptr + n, visited, mask=mask)


@triton.jit
def bullet_bwd_scan(
    gs_ptr, tape_ptr, v0_ptr, cin_ptr, q_ptr, gx_ptr, cout_ptr, gacc_ptr,
    L, N, D, theta, alpha,
    DECAY: tl.constexpr, TAPE: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """Backward pass of the detached hard reset as a chunked reverse scan.

    gh_t = gs_t sg_t + (1 - s_t) k_{t+1},  k_t = (1 - q) gh_t  (k = gradient of the membrane v_t).
    Chunks of K steps run from the last to the first; inside a chunk a reverse associative scan
    of the maps gh_t = a_t gh_{t+1} + c_t gives every gh_t from the carry k entering the chunk.
    """
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = n < N
    q = tl.load(q_ptr + n % D, mask=mask, other=0.0)
    keep = 1.0 - q
    v0 = tl.load(v0_ptr + n, mask=mask, other=0.0)
    k = tl.load(cin_ptr + n, mask=mask, other=0.0)
    gacc = tl.zeros((BLOCK,), tl.float32)
    j = tl.arange(0, K)
    for c in range(L // K - 1, -1, -1):
        t = (c * K + j[:, None]).to(tl.int64) + tl.zeros((1, BLOCK), tl.int64)
        valid = mask[None, :] & (t < L)
        off = t * N + n[None, :]
        gs = tl.load(gs_ptr + off, mask=valid, other=0.0).to(tl.float32)
        prev_ok = valid & (t > 0)
        toff = _tape_offset(t, n[None, :], N, TAPE)
        poff = _tape_offset(t - 1, n[None, :], N, TAPE)
        if TAPE == 3:
            h = _decode_u8(tl.load(tape_ptr + toff, mask=valid, other=0), theta)
            hp = _decode_u8(tl.load(tape_ptr + poff, mask=prev_ok, other=0), theta)
        else:
            h = tl.load(tape_ptr + toff, mask=valid, other=0.0).to(tl.float32)
            hp = tl.load(tape_ptr + poff, mask=prev_ok, other=0.0).to(tl.float32)
        z = (h - theta) * (1.5707963267948966 * alpha)
        sg = (0.5 * alpha) / (1.0 + z * z)
        m = 1.0 - (h >= theta).to(tl.float32)
        cterm = gs * sg
        # the last row of the chunk takes the carry k itself, the other rows (1 - q) gh of the next row
        a = tl.where(j[:, None] == K - 1, m, m * keep[None, :])
        A, Bc = tl.associative_scan((a, cterm), axis=0, combine_fn=_then, reverse=True)
        gh = A * k[None, :] + Bc
        gh = tl.where(valid, gh, 0.0)
        vp = tl.where(t > 0, (1.0 - (hp >= theta).to(tl.float32)) * hp, v0[None, :])
        if DECAY:
            tl.store(gx_ptr + off, gh * q[None, :], mask=valid)
            gacc += tl.sum(gh * (h - vp), axis=0)
        else:
            tl.store(gx_ptr + off, gh, mask=valid)
            gacc -= tl.sum(gh * vp, axis=0)
        k = keep * tl.sum(tl.where(j[:, None] == 0, gh, 0.0), axis=0)
    tl.store(cout_ptr + n, k, mask=mask)
    tl.store(gacc_ptr + n, gacc, mask=mask)


class _BulletTrain(torch.autograd.Function):
    """Bullet Trains forward kernel; backward by the serial kernel or by the chunked scan."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: Tensor, q: Tensor, v0: Tensor, theta: float, decay_input: bool, cfg: dict
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len, batch, width = x.shape
        x, q = neuron._kernel_inputs(x, q, width)
        neurons = batch * width
        vin = v0.reshape(neurons).contiguous().float()
        spikes = torch.empty_like(x)
        tape_shape = (seq_len * neurons,) if cfg["tape"] == "u8" else (seq_len, batch, width)
        tape = torch.empty(tape_shape, dtype=TAPES[cfg["tape"]], device=x.device)
        v_seq = torch.empty((seq_len, batch, width), device=x.device) if cfg["store_v"] else x.new_empty(0)
        vout = torch.empty(neurons, device=x.device)
        rounds = torch.empty(neurons, dtype=torch.int32, device=x.device)
        visited = torch.empty(neurons, dtype=torch.int32, device=x.device)
        bullet_fwd[(triton.cdiv(neurons, cfg["BLOCK"]),)](
            x, spikes, tape, v_seq, vin, vout, q, rounds, visited, seq_len, neurons, width, theta,
            decay_input, TAPE_CODE[cfg["tape"]], cfg["store_v"], cfg["K"], cfg["BLOCK"],
            num_warps=cfg["nw"], enable_fp_fusion=False,
        )  # fmt: skip
        ctx.save_for_backward(q, vin, tape)
        ctx.theta, ctx.decay_input, ctx.cfg, ctx.shape = theta, decay_input, cfg, (seq_len, batch, width)
        ctx.mark_non_differentiable(v_seq, rounds, visited)
        return spikes, vout.view(batch, width), v_seq, rounds, visited

    @staticmethod
    @once_differentiable
    def backward(ctx, gs: Tensor, gv_last: Tensor, *_unused: Tensor) -> tuple:  # type: ignore[override]
        """Gradients for x, q and v0 (None for the non-tensor inputs)."""
        q, vin, tape = ctx.saved_tensors
        seq_len, batch, width = ctx.shape
        cfg, theta = ctx.cfg, ctx.theta
        gs = gs.contiguous()
        cin = gv_last.reshape(batch * width).contiguous()
        if cfg["bwd"] == "serial":
            if cfg["tape"] == "u8":
                tape = tape.view(torch.int32).view(seq_len // 4, batch, width)
            opts = neuron.KernelOpts(cfg["U"], cfg["sBLOCK"], cfg["snw"], TAPE_CODE[cfg["tape"]], False, False, True, True)
            gx, cout, gq = neuron.launch_bwd(
                gs, tape, gs, q, vin.view(1, -1), cin.view(1, -1), theta, ctx.decay_input, opts, seq_len
            )
            return gx, gq.view(batch, width).sum(0), cout.view(batch, width), None, None, None
        neurons = batch * width
        gx = torch.empty_like(gs)
        cout = torch.empty_like(cin)
        gacc = torch.empty_like(cin)
        bullet_bwd_scan[(triton.cdiv(neurons, cfg["bBLOCK"]),)](
            gs, tape, vin, cin, q, gx, cout, gacc, seq_len, neurons, width, theta, neuron.ALPHA,
            ctx.decay_input, TAPE_CODE[cfg["tape"]], cfg["Kb"], cfg["bBLOCK"],
            num_warps=cfg["bnw"], enable_fp_fusion=False,
        )  # fmt: skip
        if ctx.decay_input:
            q_row = q.repeat(batch)
            gacc = torch.where(q_row > 0, gacc / q_row, torch.zeros_like(gacc))
        return gx, gacc.view(batch, width).sum(0), cout.view(batch, width), None, None, None


@neuron.register("bullet_train", lambda L, **_: L)
def bullet_train(
    x: Tensor,
    tau: Tensor,
    theta: float,
    v0: Tensor | None = None,
    decay_input: bool = True,
    return_v_seq: bool = False,
    K: int = 128,
    BLOCK: int = 32,
    nw: int = 4,
    tape: str = "u8",
    bwd: str = "serial",
    U: int = 32,
    sBLOCK: int = 64,
    snw: int = 2,
    Kb: int = 64,
    bBLOCK: int = 32,
    bnw: int = 4,
    hard: int = 0,
    detach: int = 1,
) -> tuple[Tensor, ...]:
    """Bullet Trains for the discrete LIF with a detached hard reset. x: [L, B, D] float.

    K: steps per round of the forward scan. BLOCK, nw: neurons and warps per forward program.
    tape: "u8" | "h16" | "h32", what the forward pass saves of h for the backward pass.
    bwd: "serial" (the serial kernel of `serial_triton`, launched with U, sBLOCK, snw) or
    "scan" (chunked reverse scan with chunks of Kb steps, bBLOCK neurons, bnw warps).
    Only the hard reset (`hard=1`) with a detached reset is supported; the paper has no other.
    When `return_v_seq` is set (the gate calls), the measured rounds and work retained are
    left in `neuron.LAST_NOTE`.
    """
    neuron._one_of("tape", tape, tuple(TAPES))
    neuron._one_of("bwd", bwd, ("serial", "scan"))
    if not hard or not detach:
        raise ValueError("bullet_train implements the paper's neuron: hard reset (hard=1), detached reset")
    seq_len = x.shape[0]
    if K & (K - 1) or Kb & (Kb - 1):
        raise ValueError(f"K={K} and Kb={Kb} must be powers of two (Triton tile sizes)")
    if bwd == "serial" and seq_len % U:
        raise ValueError(f"bwd=serial needs L a multiple of U={U}")
    if bwd == "scan" and seq_len % Kb:
        raise ValueError(f"bwd=scan needs L a multiple of Kb={Kb}")
    if tape == "u8" and seq_len % 4:
        raise ValueError("the 8-bit tape needs L a multiple of 4")
    cfg = {"K": K, "BLOCK": BLOCK, "nw": nw, "tape": tape, "bwd": bwd, "U": U, "sBLOCK": sBLOCK, "snw": snw,
           "Kb": Kb, "bBLOCK": bBLOCK, "bnw": bnw, "store_v": bool(return_v_seq)}  # fmt: skip
    start = neuron.zero_state(x) if v0 is None else v0
    spikes, v_last, v_seq, rounds, visited = _BulletTrain.apply(x, neuron.leak_q(tau), start, float(theta), decay_input, cfg)
    if return_v_seq:
        neuron.LAST_NOTE.append(round_stats(rounds, visited, seq_len, BLOCK))
        return spikes, v_last, v_seq
    return spikes, v_last


def round_stats(rounds: Tensor, visited: Tensor, seq_len: int, block: int) -> str:
    """Rounds per neuron, rounds per program (its busiest neuron) and the paper's work retained."""
    per_program = torch.nn.functional.pad(rounds, (0, (-rounds.numel()) % block)).view(-1, block).max(1).values
    retained = seq_len * rounds.numel() / visited.double().sum().item()
    return (
        f"rounds/neuron mean={rounds.float().mean():.0f} max={rounds.max()}; "
        f"rounds/program mean={per_program.float().mean():.0f} max={per_program.max()}; "
        f"work_retained={retained:.3f}"
    )


if __name__ == "__main__":
    # Smoke check: a small random case against the exact serial reference, both backward passes.
    torch.manual_seed(0)
    L, B, D = 256, 2, 48
    x = (torch.randn(L, B, D, device="cuda") * 1.5 + 0.6).requires_grad_(True)
    tau = torch.full((D,), 2.5, device="cuda").requires_grad_(True)
    r = torch.randn(L, B, D, device="cuda")
    ref = neuron.sj_reference(x.detach(), tau.detach(), 1.0, r, None, True, True, True)
    for backward in ("serial", "scan"):
        for tape_name in ("h32", "u8"):
            s, v_last, v_seq = bullet_train(x, tau, 1.0, return_v_seq=True, K=32, bwd=backward, tape=tape_name, hard=1)
            gx, gtau = torch.autograd.grad(s, (x, tau), grad_outputs=r)
            wrong = int((s.detach() != ref["spikes"]).sum())
            cos = torch.nn.functional.cosine_similarity(gx.flatten(), ref["gx"].flatten(), dim=0).item()
            rel_tau = ((gtau - ref["gtau"]).norm() / ref["gtau"].norm()).item()
            print(f"bwd={backward} tape={tape_name}: wrong spikes {wrong}, cos gx {cos:.6f}, "
                  f"rel err gtau {rel_tau:.1e}; {neuron.LAST_NOTE.pop()}")  # fmt: skip
