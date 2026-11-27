"""Benchmark one (method, config, site) of `neuron.py` against the exact serial reference.

Entry point of the study in program.md. It loads the fixed inputs from `bench_inputs.pt`,
runs the method on the three test inputs (real activations, counting stream, slow leak),
compares it with `neuron.sj_reference`, times it on the real input, and prints one
`RESULT` line per config in the column order of `bench_results.tsv`.

    python bench_neuron.py --make-inputs                       # once: writes bench_inputs.pt
    python bench_neuron.py --method serial_fused --config "-" --site in > bench.log 2>&1
    grep "^RESULT" bench.log | cut -f2- >> bench_results.tsv

`--config` is "-" or "key=value,key=value"; several configs separated by ";" give one
RESULT line each. The methods themselves live in `neuron.py`.

    python bench_neuron.py --stress 240     # random cases of the recommended kernels against a plain loop

The stress test needs no input file. It prints one STRESS line and exits with status 1 if a case failed.
"""

import sys

sys.dont_write_bytecode = True  # the study folder must stay free of subfolders (__pycache__)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

logger = logging.getLogger("bench")

# Neuron variant of this run, set from the command line: keyword arguments for the method
# ("hard", "detach") and for the reference. Empty = the study's neuron (soft, detached).
VARIANT: dict[str, int] = {}

# > 1: the method is tick-aware. It gets one input per token plus the keyword `T`, and
# returns spike counts per token. The reference then runs on the tick axis (every token
# repeated T times) and is reduced to tokens: counts and gradients summed over the ticks.
TICK_FUSED = 1

# True: the method gets x and r batch-first, [B, L, D], with the keyword `bf=1`, and returns
# batch-first spikes. The comparison with the (time-first) reference transposes them back.
BATCH_FIRST = False


def layout(t: Tensor) -> Tensor:
    """A tensor [L, B, D] in the layout the method wants (a contiguous [B, L, D] copy if batch-first)."""
    return t.transpose(0, 1).contiguous() if BATCH_FIRST else t


def time_first(t: Tensor | None) -> Tensor | None:
    """A tensor that the method returned in its own layout, as a [L, B, D] view."""
    return t.transpose(0, 1) if (BATCH_FIRST and t is not None) else t

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_DIR = REPO / "runs" / "mamba3-siso-187m-wikitext103-spiking"
CHECKPOINT = RUN_DIR / "best.pt"
INPUTS = HERE / "bench_inputs.pt"

SEED = 0
SEQ_LEN = 2048            # L, tokens per sequence
BATCH = 16                # B, sequences
CAPTURE_LAYER = 6         # the middle block of the 12; its two PLIF inputs are the real input
THETA = 1.0               # the repo's v_threshold (not learnable)
COUNT_THETA = 2.0         # counting input: theta = 2, no leak
SLOW_TAU = 20.0           # slow-leak input: beta = 1 - 1/tau = 0.95
SLOW_RATE = 0.075         # target firing rate of the reference on the slow-leak input
WARMUPS, REPEATS = 3, 10
GATE_SPIKE, GATE_GRAD = 0.999, 0.99
SITE_WIDTH = {"in": 768, "out": 1536}


@dataclass
class Case:
    """One test input: the tensor [L, B, D] and the neuron settings that go with it."""

    name: str
    x: Tensor
    tau: Tensor
    theta: float
    decay_input: bool


# --------------------------------------------------------------------------------------
# Building the fixed inputs (run once)
# --------------------------------------------------------------------------------------


def capture_activations() -> dict[str, Any]:
    """One forward pass of the checkpoint on 16 validation sequences of 2048 tokens.

    Returns the inputs of the two PLIF neurons of block `CAPTURE_LAYER` as [L, B, D]
    float32 CPU tensors, their learned leak parameter w, and the firing rate that the
    trained neuron (hard reset, 4 ticks per token) had on these inputs.
    """
    sys.path.insert(0, str(REPO / "src"))
    import spiking_mamba3  # registers the "SpikingMamba3" layer
    from dataset.dataset import LMDataset, load_tokens
    from models.model import MambaConfig, MambaLMHeadModel
    from utils.checkpoint import load_model_state

    cfg = json.loads((RUN_DIR / "config.json").read_text())["config"]
    model = MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")
    model.load_state_dict(load_model_state(CHECKPOINT))
    model.eval()
    mixer = model.backbone.layers[CAPTURE_LAYER].mixer
    sites = {"in": mixer.in_proj.plif, "out": mixer.out_proj.plif}
    captured: dict[str, list[Tensor]] = {"in": [], "out": []}
    hooks = [
        plif.register_forward_pre_hook(
            lambda _module, args, site=site: captured[site].append(args[0].detach().float().cpu())
        )
        for site, plif in sites.items()
    ]
    val = LMDataset(load_tokens(REPO / cfg["data"]["val_bin"]), SEQ_LEN)
    tokens = torch.stack([val[i][0] for i in range(BATCH)]).cuda()
    spiking_mamba3.start_spike_count(model)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, BATCH, 2):
            model(tokens[start : start + 2])
    trained_rates = spiking_mamba3.spike_rates(model)
    for hook in hooks:
        hook.remove()
    out: dict[str, Any] = {"trained_rates_all_layers": trained_rates}
    for site, plif in sites.items():
        x = torch.cat(captured[site]).transpose(0, 1).contiguous()  # [B, L, D] -> [L, B, D]
        out[site] = {"x_real": x, "w_real": float(plif.neuron.w.detach())}
    return out


def calibrate_sigma(z: Tensor, tau: Tensor, target: float) -> float:
    """Bisect the scale of the Gaussian `z` until the reference fires at `target`."""
    import neuron

    lo, hi = 0.01, 1000.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        rate = neuron.sj_reference(z * mid, tau, THETA)["spikes"].mean().item()
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    return math.sqrt(lo * hi)


def make_inputs() -> None:
    """Create `bench_inputs.pt`: the three test inputs and the fixed `r`, for both sites."""
    import neuron

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    meta: dict[str, Any] = {"seed": SEED, "L": SEQ_LEN, "B": BATCH, "layer": CAPTURE_LAYER}
    started = time.time()
    try:
        real = capture_activations()
        meta["real_input"] = f"block {CAPTURE_LAYER} of {CHECKPOINT.name}, first {BATCH} validation windows"
        meta["trained_rates_all_layers"] = real["trained_rates_all_layers"]
    except Exception as error:  # the 30-minute fallback of program.md: Gaussian input
        logger.exception("activation capture failed, using the Gaussian fallback")
        meta["real_input"] = f"FALLBACK Gaussian ({type(error).__name__}: {error})"
        real = {site: {"x_real": None, "w_real": 0.0} for site in SITE_WIDTH}
    meta["capture_seconds"] = round(time.time() - started, 1)

    data: dict[str, Any] = {"meta": meta}
    for site, width in SITE_WIDTH.items():
        shape = (SEQ_LEN, BATCH, width)
        tau_real = 1.0 / torch.sigmoid(torch.tensor(real[site]["w_real"], dtype=torch.float64)).item()
        tau_vec = torch.full((width,), tau_real, device="cuda")
        x_real = real[site]["x_real"]
        if x_real is None:
            z = torch.randn(shape, generator=gen)
            x_real = z * calibrate_sigma(z.cuda(), tau_vec, SLOW_RATE)
        z_slow = torch.randn(shape, generator=gen)
        slow_tau = torch.full((width,), SLOW_TAU, device="cuda")
        sigma = calibrate_sigma(z_slow.cuda(), slow_tau, SLOW_RATE)
        x_slow = z_slow * sigma
        x_count = (torch.rand(shape, generator=gen) < 0.5).to(torch.uint8)
        r = torch.randn(shape, generator=gen)
        rate_real = neuron.sj_reference(x_real.cuda(), tau_vec, THETA)["spikes"].mean().item()
        rate_slow = neuron.sj_reference(x_slow.cuda(), slow_tau, THETA)["spikes"].mean().item()
        data[site] = {
            "x_real": x_real,
            "x_slow": x_slow,
            "x_count": x_count,
            "r": r,
            "tau_real": tau_real,
            "w_real": real[site]["w_real"],
            "slow_sigma": sigma,
            "rate_real": rate_real,
            "rate_slow": rate_slow,
        }
        logger.info(
            "site %s D=%d tau_real=%.4f rate_real=%.4f slow_sigma=%.3f rate_slow=%.4f x_real mean=%.3f std=%.3f",
            site, width, tau_real, rate_real, sigma, rate_slow, x_real.mean().item(), x_real.std().item(),
        )  # fmt: skip
    torch.save(data, INPUTS)
    logger.info("wrote %s, meta=%s", INPUTS.name, meta)


def load_cases(
    site: str, seq_len: int, batch: int, slow_tau: float = SLOW_TAU, ticks: int = 1, only_real: bool = False
) -> tuple[list[Case], Tensor, dict[str, Any]]:
    """The three test cases and `r` of one site, on the GPU.

    `seq_len` and `batch` other than 2048 and 16 repeat or crop the stored tensors along
    the time and batch axes (length-scaling and occupancy studies). `ticks` > 1 feeds
    every token `ticks` times in a row (the tick axis of the repo; `seq_len` then counts
    ticks). `slow_tau` changes the leak of the slow-leak case (leak sweep).
    """
    data = torch.load(INPUTS, mmap=True, weights_only=True)
    site_data = data[site]
    width = SITE_WIDTH[site]

    def fit(t: Tensor) -> Tensor:
        """Cut or repeat a stored tensor to [seq_len, batch, D] on the GPU."""
        t = t[: math.ceil(seq_len / ticks), :batch].cuda().float()
        if ticks > 1:
            t = t.repeat_interleave(ticks, dim=0)
        reps_l, reps_b = math.ceil(seq_len / t.shape[0]), math.ceil(batch / t.shape[1])
        if reps_l > 1 or reps_b > 1:
            t = t.repeat(reps_l, reps_b, 1)
        return t[:seq_len, :batch].contiguous()

    tau_real = torch.full((width,), site_data["tau_real"], device="cuda")
    cases = [Case("real", fit(site_data["x_real"]), tau_real, THETA, True)]
    if not only_real:
        no_leak = torch.full((width,), math.inf, device="cuda")
        cases.append(Case("count", fit(site_data["x_count"]), no_leak, COUNT_THETA, False))
        cases.append(Case("slow", fit(site_data["x_slow"]), torch.full((width,), slow_tau, device="cuda"), THETA, True))
    return cases, fit(site_data["r"]), dict(data["meta"])


# --------------------------------------------------------------------------------------
# Fidelity gate and timing
# --------------------------------------------------------------------------------------


def parse_config(text: str) -> dict[str, Any]:
    """"C=256,W=128,relax=hard" -> {"C": 256, "W": 128, "relax": "hard"}; "-" -> {}."""
    config: dict[str, Any] = {}
    if text.strip() in ("", "-"):
        return config
    for item in text.split(","):
        key, _, value = item.strip().partition("=")
        if not value:
            config[key] = True
            continue
        if value.lower() in ("true", "false"):  # as a string, "False" would count as true in bool()
            config[key] = int(value.lower() == "true")
            continue
        for cast in (int, float):
            try:
                config[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            config[key] = value
    return config


def cosine(a: Tensor | None, b: Tensor) -> float:
    """Cosine similarity of two gradients in float64; 0 for a missing or zero gradient."""
    if a is None:
        return 0.0
    a64, b64 = a.double().flatten(), b.double().flatten()
    norm = a64.norm() * b64.norm()
    if not bool(torch.isfinite(norm)) or norm.item() == 0.0:
        return 0.0
    return (a64 @ b64 / norm).item()


def floor4(value: float, tolerance: float = 0.0) -> str:
    """Four decimals, rounded down, so that 1.0000 means 1 (within `tolerance`).

    Spike agreement is an exact count, so its tolerance is 0. A cosine of two identical
    float32 gradients can come out as 1 - 1e-12 in float64, so cosines use 1e-7.
    """
    if math.isnan(value):
        return "nan"
    return f"{math.floor((value + tolerance) * 1e4) / 1e4:.4f}"


def cos4(value: float) -> str:
    """A cosine with four decimals, rounded down; 1.0000 means 1 within 1e-7."""
    return floor4(value, 1e-7)


def sync() -> None:
    """Wait until every queued GPU kernel has finished (needed around every timing)."""
    torch.cuda.synchronize()


def median_ms(fn: Any) -> float:
    """Median wall-clock milliseconds of `REPEATS` calls after `WARMUPS` warm-up calls."""
    for _ in range(WARMUPS):
        fn()
        sync()
    times = []
    for _ in range(REPEATS):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return 0.5 * (times[len(times) // 2 - 1] + times[len(times) // 2])


def reference(case: Case, r: Tensor | None) -> dict[str, Tensor]:
    """The exact serial result for one case, at the resolution the method works at."""
    import neuron

    hard, detach = bool(VARIANT.get("hard", 0)), bool(VARIANT.get("detach", 1))
    if TICK_FUSED == 1:
        return neuron.sj_reference(case.x, case.tau, case.theta, r, None, case.decay_input, hard, detach)
    ticks = TICK_FUSED
    x = case.x.repeat_interleave(ticks, dim=0)
    r_ticks = None if r is None else r.repeat_interleave(ticks, dim=0)
    full = neuron.sj_reference(x, case.tau, case.theta, r_ticks, None, case.decay_input, hard, detach)
    per_token = (case.x.shape[0], ticks, *case.x.shape[1:])
    out = {"spikes": full["spikes"].view(per_token).sum(1), "v_seq": full["v_seq"][ticks - 1 :: ticks]}
    out["v_last"] = full["v_last"]
    if r is not None:
        out["gx"], out["gtau"] = full["gx"].view(per_token).sum(1), full["gtau"]
    return out


def fidelity(method: Any, config: dict[str, Any], cases: list[Case], r: Tensor) -> dict[str, Any]:
    """Run the method on the three inputs and compare it with the reference."""
    import neuron

    scalar_leak = method.scalar_leak(**config)
    stats: dict[str, Any] = {}
    for case in cases:
        with_grad = case.name != "count"
        ref = reference(case, r if with_grad else None)
        x = layout(case.x).clone().requires_grad_(with_grad)
        tau = case.tau.clone().requires_grad_(with_grad)
        with torch.set_grad_enabled(with_grad):
            out = method.fn(x, tau, case.theta, decay_input=case.decay_input, return_v_seq=True, **config, **VARIANT)
        if neuron.LAST_NOTE:  # a remark the method left about this call
            stats.setdefault("remarks", []).append(f"{case.name}: {neuron.LAST_NOTE[-1]}")
            neuron.LAST_NOTE.clear()
        spikes, v_last = out[0], out[1]
        v_seq = out[2] if len(out) > 2 else None
        mismatch = int((time_first(spikes).detach() != ref["spikes"]).sum())
        stats[f"agree_{case.name}"] = 1.0 - mismatch / spikes.numel()
        stats[f"mismatch_{case.name}"] = mismatch
        stats[f"rate_{case.name}"] = spikes.detach().float().mean().item()
        if case.name == "count":
            # float32 counts: a 16-bit float cannot hold integers above 256 exactly
            diff = time_first(spikes).float().cumsum(0) - ref["spikes"].float().cumsum(0)
            stats["parity_ok"] = int(bool((diff.remainder(2.0) == 0).all()))
            continue
        if v_seq is None:
            stats[f"verr_{case.name}"] = math.nan
        else:
            verr = (v_seq.detach() - ref["v_seq"]).abs().max().item()
            stats[f"verr_{case.name}"] = max(verr, (v_last.detach() - ref["v_last"]).abs().max().item())
        if spikes.requires_grad:
            gx, gtau = torch.autograd.grad(spikes, (x, tau), grad_outputs=layout(r), allow_unused=True)
        else:
            gx, gtau = None, None
        stats[f"cos_x_{case.name}"] = cosine(time_first(gx), ref["gx"])
        if scalar_leak and gtau is not None:
            # A scalar leak has one gradient: compare its value with the reference summed over channels.
            total, ref_total = gtau.double().sum().item(), ref["gtau"].double().sum().item()
            rel = abs(total - ref_total) / max(abs(ref_total), 1e-30)
            stats[f"cos_tau_{case.name}"] = 1.0 if rel < 1e-3 else max(0.0, 1.0 - rel)
        else:
            stats[f"cos_tau_{case.name}"] = cosine(gtau, ref["gtau"])
        del ref, out, spikes, v_seq, gx, gtau
    return stats


def timing(method: Any, config: dict[str, Any], case: Case, r: Tensor) -> dict[str, float]:
    """Median forward and forward+backward milliseconds and peak memory on the real input."""
    x = layout(case.x).clone().requires_grad_(True)
    tau = case.tau.clone().requires_grad_(True)
    r = layout(r)

    def forward() -> Tensor:
        return method.fn(x, tau, case.theta, decay_input=case.decay_input, **config, **VARIANT)[0]

    def forward_backward() -> None:
        spikes = forward()
        if spikes.requires_grad:
            torch.autograd.grad(spikes, (x, tau), grad_outputs=r, allow_unused=True)

    def forward_no_grad() -> None:
        with torch.no_grad():
            forward()

    result = {"ms_fwd": median_ms(forward), "ms_fwd_bwd": median_ms(forward_backward)}
    result["ms_nograd"] = median_ms(forward_no_grad)
    sync()
    torch.cuda.reset_peak_memory_stats()
    forward_backward()
    sync()
    result["mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return result


def result_line(method_name: str, config_text: str, site: str, width: int, fields: dict[str, Any]) -> str:
    """One RESULT line in the column order of bench_results.tsv."""
    columns = [
        "RESULT", method_name, config_text, site, str(width),
        f"{fields['ms_fwd']:.3f}", f"{fields['ms_fwd_bwd']:.3f}", f"{fields['mem_gb']:.1f}",
        floor4(fields["spike_agree"]), f"{fields['v_maxerr']:.1e}", cos4(fields["grad_cos"]),
        str(fields["parity_ok"]), str(fields["seq_depth"]), fields["status"], fields["notes"],
    ]  # fmt: skip
    return "\t".join(columns)


def crash_fields(reason: str, seq_depth: int = 0) -> dict[str, Any]:
    """Row content for a crashed or skipped config: zeros in the numeric columns."""
    return {
        "ms_fwd": 0.0, "ms_fwd_bwd": 0.0, "mem_gb": 0.0, "spike_agree": 0.0, "v_maxerr": 0.0,
        "grad_cos": 0.0, "parity_ok": 0, "seq_depth": seq_depth, "status": "crash", "notes": reason,
    }  # fmt: skip


def run_config(method_name: str, config_text: str, site: str, args: argparse.Namespace) -> dict[str, Any]:
    """Fidelity gate plus timing of one config. Returns the fields of its RESULT line."""
    import neuron

    method = neuron.METHODS[method_name]
    config = parse_config(config_text)
    neuron.LAST_NOTE.clear()  # a remark of an earlier (crashed or timing-only) config must not land in this row
    depth_config = {**config, **({"T": VARIANT["T"]} if "T" in VARIANT else {})}
    cases, r, _meta = load_cases(site, args.L, args.B, args.slow_tau, args.ticks, args.timing_only)
    if args.io != "fp32":  # feed the neuron activations and gradients in a 16-bit float type
        io_dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[args.io])
        cases = [Case(c.name, c.x.to(io_dtype), c.tau, c.theta, c.decay_input) for c in cases]
        r = r.to(io_dtype)
    size = ""
    if (args.L, args.B, args.ticks) != (SEQ_LEN, BATCH, 1):
        size = f"L={args.L},B={args.B}" + (f",ticks={args.ticks}" if args.ticks > 1 else "")
    if args.slow_tau != SLOW_TAU:
        size += ("," if size else "") + f"slow_tau={args.slow_tau:g}"
    variant = [f"io={args.io}"] if args.io != "fp32" else []
    variant += [f"{key}={value}" for key, value in VARIANT.items()]
    if variant:
        size += ("," if size else "") + "VARIANT " + ",".join(variant)
    if args.timing_only:
        times = timing(method, config, cases[0], r)
        notes = f"TIMING ONLY (gate not re-run at this size); {size}; nograd_ms={times['ms_nograd']:.3f}"
        return {
            "ms_fwd": times["ms_fwd"], "ms_fwd_bwd": times["ms_fwd_bwd"], "mem_gb": times["mem_gb"],
            "spike_agree": math.nan, "v_maxerr": math.nan, "grad_cos": math.nan, "parity_ok": "nan",
            "seq_depth": method.seq_depth(args.L, **depth_config), "status": "ok",
            "notes": notes + ("; " + args.note if args.note else ""),
        }  # fmt: skip
    stats = fidelity(method, config, cases, r)
    real = cases[0]
    del cases
    torch.cuda.empty_cache()
    times = timing(method, config, real, r)

    cos_values = [stats[f"cos_{kind}_{name}"] for kind in ("x", "tau") for name in ("real", "slow")]
    grad_cos = min(cos_values)
    v_maxerr = max(stats["verr_real"], stats["verr_slow"])
    failed = []
    if stats["agree_real"] < GATE_SPIKE:
        failed.append("spikes")
    if grad_cos < GATE_GRAD:
        failed.append("grad")
    if stats["parity_ok"] != 1:
        failed.append("parity")
    gate = "PASS" if not failed else "FAIL(" + "+".join(failed) + ")"
    notes = (
        f"{gate}; mis real/count/slow={stats['mismatch_real']}/{stats['mismatch_count']}/{stats['mismatch_slow']}; "
        f"agree_slow={floor4(stats['agree_slow'])}; verr real/slow={stats['verr_real']:.1e}/{stats['verr_slow']:.1e}; "
        f"cos x,tau real={cos4(stats['cos_x_real'])},{cos4(stats['cos_tau_real'])} "
        f"slow={cos4(stats['cos_x_slow'])},{cos4(stats['cos_tau_slow'])}; "
        f"rate={stats['rate_real']:.4f}; nograd_ms={times['ms_nograd']:.3f}"
    )
    neuron.LAST_NOTE.clear()
    if stats.get("remarks"):
        notes += "; " + " | ".join(stats["remarks"])
    if size:
        notes += "; " + size
    if method.scalar_leak(**config):
        notes += "; scalar leak: tau-grad compared as channel sum"
    if args.note:
        notes += "; " + args.note
    return {
        "ms_fwd": times["ms_fwd"], "ms_fwd_bwd": times["ms_fwd_bwd"], "mem_gb": times["mem_gb"],
        "spike_agree": stats["agree_real"], "v_maxerr": v_maxerr, "grad_cos": grad_cos,
        "parity_ok": stats["parity_ok"], "seq_depth": method.seq_depth(args.L, **depth_config),
        "status": "ok", "notes": notes,
    }  # fmt: skip


# --------------------------------------------------------------------------------------
# Random stress test of the recommended kernels:  python bench_neuron.py --stress 240
# --------------------------------------------------------------------------------------

STRESS_EXACT = 0.9999       # smallest accepted gradient cosine with the float32 tape
STRESS_ROUNDED = 0.98       # the same with a rounded tape (h16, u8, tick tape) or the interval certificate
STRESS_TAU_ROUNDED = 0.95   # leak gradient with a rounded tape; checked from 32 channels on (see below)
STRESS_NORM = {STRESS_EXACT: 0.01, STRESS_ROUNDED: 0.1, STRESS_TAU_ROUNDED: 0.2}  # allowed |norm ratio - 1|


@dataclass
class StressCase:
    """One random test: a method of `neuron.py`, its config, the neuron variant and the tensors."""

    method: str
    config: dict[str, Any]
    theta: float
    decay_input: bool
    ticks: int      # > 1: the method is tick-aware and gets one input per token
    x: Tensor       # [L, B, D], time-first
    tau: Tensor     # [D], a different leak per channel
    v0: Tensor      # [B, D]
    r: Tensor       # [L, B, D], the incoming gradient for the spikes
    r_last: Tensor  # [B, D], the incoming gradient for the last membrane
    x_view: bool    # give the method x as a view that is not contiguous
    r_view: bool    # give the backward pass a gradient that is not contiguous


def loop_reference(case: StressCase) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The neuron of `case` as a plain PyTorch loop with autograd.

    Returns (spikes, v_last, grad x, grad tau, grad v0) for the loss
    sum(r * spikes) + sum(r_last * v_last). For a tick-aware case every token is fed
    `ticks` times, and the spikes and the gradient for x are summed over the ticks.
    """
    import neuron
    from spikingjelly.activation_based import surrogate

    hard, detach = bool(case.config["hard"]), bool(case.config["detach"])
    x = case.x.repeat_interleave(case.ticks, 0).requires_grad_(True)
    tau = case.tau.clone().requires_grad_(True)
    v0 = case.v0.clone().requires_grad_(True)
    q = neuron.leak_q(tau)
    fire = surrogate.ATan(alpha=neuron.ALPHA)
    v, spikes = v0, []
    for step in range(x.shape[0]):
        h = neuron.charge(v, x[step], q, case.decay_input)
        s = fire(h - case.theta)
        s_reset = s.detach() if detach else s
        v = (1.0 - s_reset) * h if hard else h - case.theta * s_reset
        spikes.append(s)
    out = torch.stack(spikes)
    r = case.r.repeat_interleave(case.ticks, 0)
    gx, gtau, gv = torch.autograd.grad((out, v), (x, tau, v0), grad_outputs=(r, case.r_last))
    per_token = (case.x.shape[0], case.ticks, *case.x.shape[1:])
    return out.detach().view(per_token).sum(1), v.detach(), gx.view(per_token).sum(1), gtau, gv


def random_stress_case(rng: random.Random, layout_rng: random.Random, extra: torch.Generator) -> StressCase:
    """Draw one case: shape, leaks, neuron variant, one of the three kernels and a valid config for it.

    `layout_rng` decides the memory layout of the tensors (views, an expanded gradient) and
    `extra` draws the gradient for the last membrane. They are separate generators, so the
    shapes, configs and inputs of a seed stay the same when such draws are added.
    """
    unroll = rng.choice([4, 8, 16, 32])
    seq_len = unroll * rng.choice([2, 3, 4, 6, 8, 12, 16])
    batch, width = rng.choice([1, 2, 3, 5]), rng.choice([1, 7, 33, 64, 130, 257])
    theta, decay_input = rng.choice([0.5, 1.0, 2.0]), rng.random() < 0.6
    hard, detach = rng.random() < 0.35, rng.random() < 0.6
    block, warps = rng.choice([(32, 1), (64, 2), (64, 1), (128, 4), (256, 4), (512, 8)])
    method = rng.choice(["serial_triton", "serial_triton", "spec_verify", "spec_verify", "serial_ticks"])
    ticks = 1
    if method == "serial_triton":
        config = {"U": unroll, "BLOCK": block, "nw": warps, "tape": rng.choice(["h32", "h16", "u8"]),
                  "usex": int(rng.random() < 0.3)}  # fmt: skip
    elif method == "spec_verify":
        chunk = rng.choice([c for c in range(unroll, seq_len + 1, unroll) if seq_len % c == 0])
        # the exact chunked backward pass needs the detached soft reset; the 8-bit tape needs an untruncated one
        backward = rng.choice(["serial", "cold", "exact"] if (detach and not hard) else ["serial", "cold"])
        tape = rng.choice(["h32", "h16"] if backward == "cold" else ["h32", "h16", "u8"])
        config = {"C": chunk, "W": unroll * rng.choice([0, 1, 2, 4]), "bwd": backward, "tape": tape,
                  "cert": "bit" if hard else rng.choice(["bit", "interval"]),
                  "Wb": unroll * rng.choice([1, 2, 4]), "U": unroll, "BLOCK": block, "nw": warps}  # fmt: skip
    else:
        ticks = rng.choice([1, 2, 4])
        config = {"T": ticks, "U": min(unroll, 16), "BLOCK": min(block, 64), "nw": warps if block <= 64 else 1}
    config.update(hard=int(hard), detach=int(detach), bf=int(rng.random() < 0.4))
    scale = rng.choice([0.5, 1.5, 4.0]) * (3.0 if decay_input else 1.0)
    x = torch.randn(seq_len, batch, width, device="cuda") * scale + rng.choice([-0.5, 0.0, 0.5])
    tau = 1.05 + 40.0 * torch.rand(width, device="cuda") ** 3
    v0 = torch.randn(batch, width, device="cuda") * rng.choice([0.0, 0.0, 1.0, 3.0])
    r = torch.randn(seq_len, batch, width, device="cuda")
    layout = layout_rng.random()
    if layout < 0.15:  # the gradient of a loss like spikes.sum(): one row, expanded over time (stride 0)
        r = r[:1].expand(seq_len, batch, width)
    r_last = torch.randn(batch, width, device="cuda", generator=extra) * layout_rng.choice([0.0, 1.0])
    return StressCase(method, config, theta, decay_input, ticks, x, tau, v0, r, r_last,
                      x_view=layout_rng.random() < 0.3, r_view=0.15 <= layout < 0.45)  # fmt: skip


def _as_view(t: Tensor) -> Tensor:
    """The same values as `t` in a tensor that is not contiguous (its two last axes are swapped in memory)."""
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def stress_problems(case: StressCase) -> list[str]:
    """Run one case and compare it with the loop. Returns what is wrong; an empty list is a pass.

    Spikes must be equal. The last membrane must be bit-equal, except with the interval
    certificate, which proves equal spikes only. Gradients of sum(r * spikes) +
    sum(r_last * v_last) are compared by cosine and by norm. A truncated backward pass
    (`bwd=cold`) is approximate by design and only has to be finite. With a rounded tape
    the leak gradient of ONE channel can have the wrong sign when its sum nearly cancels,
    so that gradient is checked from 32 channels on. Some cases hand over x or the incoming
    gradient as a view that is not contiguous; the result must not depend on that.
    """
    import neuron

    s_ref, v_ref, gx_ref, gtau_ref, gv_ref = loop_reference(case)
    batch_first = bool(case.config["bf"])
    x = (case.x.transpose(0, 1).contiguous() if batch_first else case.x.clone()).requires_grad_(True)
    r = case.r.transpose(0, 1).contiguous() if batch_first else case.r
    r = _as_view(r) if case.r_view else r
    tau = case.tau.clone().requires_grad_(True)
    v0 = case.v0.clone().requires_grad_(True)
    method = neuron.METHODS[case.method].fn
    x_in = _as_view(x) if case.x_view else x
    spikes, v_last = method(x_in, tau, case.theta, v0=v0, decay_input=case.decay_input, **case.config)[:2]
    gx, gtau, gv = torch.autograd.grad(
        (spikes, v_last), (x, tau, v0), grad_outputs=(r, case.r_last), allow_unused=True
    )
    if batch_first:
        spikes, gx = spikes.transpose(0, 1), None if gx is None else gx.transpose(0, 1)

    problems = []
    wrong = int((spikes.detach() != s_ref).sum())
    if wrong:
        problems.append(f"{wrong} wrong spikes")
    interval = case.config.get("cert") == "interval"
    v_err = float((v_last.detach() - v_ref).abs().max())
    if v_err > (1e-3 * max(1.0, float(v_ref.abs().max())) if interval else 0.0):
        problems.append(f"v_last off by {v_err:.1e}")
    if case.config.get("bwd") == "cold":
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in (gx, gtau, gv))
        return problems if finite else [*problems, "missing or non-finite gradient"]
    rounded = case.config.get("tape", "tick") != "h32" or interval
    floor = STRESS_ROUNDED if rounded else STRESS_EXACT
    checks = [("x", gx, gx_ref, floor), ("v0", gv, gv_ref, floor)]
    if not rounded or case.tau.numel() >= 32:
        checks.append(("tau", gtau, gtau_ref, STRESS_TAU_ROUNDED if rounded else floor))
    for name, grad, ref, limit in checks:
        if float(ref.abs().max()) <= 1e-12:  # a zero gradient has no direction
            continue
        ratio = float(grad.double().norm() / ref.double().norm()) if grad is not None else 0.0
        if cosine(grad, ref) < limit:
            problems.append(f"grad {name} cosine {cosine(grad, ref):.4f} < {limit}")
        elif (limit == STRESS_EXACT or ref.numel() >= 32) and abs(ratio - 1.0) > STRESS_NORM[limit]:
            # with a rounded tape a gradient of a few elements can be a sum that nearly cancels: no norm check
            problems.append(f"grad {name} norm ratio {ratio:.4f}")
    return problems


def stress(n_cases: int, seed: int = SEED) -> int:
    """Run `n_cases` random cases of `serial_triton`, `spec_verify` and `serial_ticks`.

    Prints one STRESS-FAIL line per failed case and one STRESS summary line. Returns the
    number of failed cases. The same seed gives the same cases on the same GPU.
    """
    rng, layout_rng = random.Random(seed), random.Random(seed + 1000)
    torch.manual_seed(seed)
    extra = torch.Generator(device="cuda").manual_seed(seed)
    failed = 0
    counts: dict[str, int] = {}
    for index in range(n_cases):
        case = random_stress_case(rng, layout_rng, extra)
        counts[case.method] = counts.get(case.method, 0) + 1
        label = f"case {index}\t{case.method}\t{case.config}\tshape={tuple(case.x.shape)} theta={case.theta} " \
                f"decay_input={case.decay_input}"  # fmt: skip
        try:
            problems = stress_problems(case)
        except Exception:
            logger.error("stress test crashed in %s", label.replace("\t", " "))
            raise
        if problems:
            failed += 1
            print(f"STRESS-FAIL\t{label}\t{'; '.join(problems)}", flush=True)
    per_method = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    print(f"STRESS\tcases={n_cases}\tfailed={failed}\tseed={seed}\t{per_method}", flush=True)
    return failed


def main() -> None:
    """Parse the arguments, then build the inputs, run the stress test or benchmark the configs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-inputs", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--config", default="-")
    parser.add_argument("--site", choices=list(SITE_WIDTH), default="in")
    parser.add_argument("--L", type=int, default=SEQ_LEN)
    parser.add_argument("--B", type=int, default=BATCH)
    parser.add_argument("--note", default="", help="free text appended to the notes column")
    parser.add_argument("--timing-only", action="store_true", help="skip the fidelity gate (large sizes)")
    parser.add_argument("--slow-tau", type=float, default=SLOW_TAU, help="tau of the slow-leak input")
    parser.add_argument("--ticks", type=int, default=1, help="feed every token this many times in a row")
    parser.add_argument("--reset", choices=["soft", "hard"], default="soft", help="neuron variant: reset type")
    parser.add_argument("--detach", type=int, choices=[0, 1], default=1, help="neuron variant: detach_reset")
    parser.add_argument("--io", choices=["fp32", "bf16", "fp16"], default="fp32", help="dtype of x and spikes")
    parser.add_argument("--tick-fused", type=int, default=1, help="T for a tick-aware method (input per token)")
    parser.add_argument("--batch-first", action="store_true", help="give the method [B, L, D] tensors (bf=1)")
    parser.add_argument("--stress", type=int, default=0, help="run this many random stress cases, then exit")
    parser.add_argument("--stress-seed", type=int, default=SEED, help="seed of the random stress cases")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)

    if args.method == "serial_python":
        own = ["impl=own" in config_text for config_text in args.config.split(";")]
        if any(own) and not all(own):
            parser.error("serial_python: run impl=own and impl=sj in separate processes (provider is fixed at import)")
        if not any(own):
            # SpikingJelly reads this once, when it is imported: its Python-loop provider.
            os.environ["SJ_PLIF_CUDA_IMPLEMENTATION"] = "torch"
    global TICK_FUSED, BATCH_FIRST
    BATCH_FIRST = args.batch_first
    if BATCH_FIRST:
        VARIANT["bf"] = 1
    TICK_FUSED = args.tick_fused
    if TICK_FUSED > 1:
        VARIANT["T"] = TICK_FUSED
    if args.reset == "hard":
        VARIANT["hard"] = 1
    if args.detach == 0:
        VARIANT["detach"] = 0
    torch.manual_seed(SEED)
    if args.make_inputs:
        make_inputs()
        return
    if args.stress:
        sys.exit(1 if stress(args.stress, args.stress_seed) else 0)
    width = SITE_WIDTH[args.site]
    for config_text in args.config.split(";"):
        config_text = config_text.strip() or "-"
        try:
            fields = run_config(args.method, config_text, args.site, args)
        except Exception as error:
            logger.exception("config %s crashed", config_text)
            reason = f"{type(error).__name__}: {str(error).splitlines()[0][:160] if str(error) else ''}"
            fields = crash_fields(reason + ("; " + args.note if args.note else ""))
        print(result_line(args.method, config_text, args.site, width, fields), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
