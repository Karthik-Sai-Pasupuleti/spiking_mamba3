# REPORT — fastest soft-reset LIF over L = 2048 tokens (RTX 5090)

Last rewrite: 2026-11-22 05:51 CET. Study started 2026-11-21 23:58. The method menu (sections A–F of `program.md`) and
the extended study are complete: 23 methods, 1260+ result rows in `bench_results.tsv`. An independent code
review and a random stress test followed; their findings are fixed (section 6). Milestones: M1 reached, M2
at `in` only, M3 not reached (an earlier claim is withdrawn, section 2).

All times are `ms_fwd_bwd`: the median of 10 forward+backward passes of one neuron site at `[2048, 16, D]`.
`in` = D 768 (input of `in_proj`), `out` = D 1536 (input of `out_proj`). Where a config was run in several
separate processes, the table shows the median of those runs. Gate = `spike_agree ≥ 0.999` AND
`grad_cos ≥ 0.99` AND `parity_ok = 1`.

## 1. Ranked table (best passing config per method, sorted by the `in` time)

| rank | method | best config | in ms | out ms | spike_agree | grad_cos in / out | parity | seq_depth | speed-up in / out |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `serial_triton` | in `U=32,BLOCK=64,nw=2,tape=u8`; out `U=32,BLOCK=512,nw=8,tape=u8` | **0.415** | **0.677** | 1.0000 | 0.9996 / 0.9997 | 1 | 2048 | **6.9× / 4.6×** |
| 2 | `spec_verify` (bit certificate, 8-bit tape) | in `C=1024,W=128,bwd=serial,tape=u8,U=32`; out `C=1024,W=256,bwd=serial,tape=u8,U=32,BLOCK=512,nw=8` | 0.425 | 0.750 | 1.0000 | 0.9996 / 0.9997 | 1 | 1154 / 1282 forward, 2048 backward | 6.7× / 4.1× |
| 2b | `spec_verify` (interval certificate, chunked backward) | `C=32,W=32,cert=interval,bwd=cold,Wb=32,U=32,BLOCK=64,nw=2` | 0.517 | 1.00 | 1.0000 | 0.9949 / 0.9949 | 1 | 128 (32 + 32, then a sweep over 64 chunks) | 5.5× / 3.1× |
| 3 | `chunk_warm` | `C=1024,p=2,tape=u8,U=32` (= L/C passes) | 0.514 | 0.974 | 1.0000 | 0.9996 / 0.9997 | 1 | 2048 | 5.5× / 3.2× |
| 4 | `decoupled_threshold` | in `U=32,BLOCK=64,nw=2`; out `U=16,BLOCK=256,nw=4` | 0.587 | 1.007 | 1.0000 | 0.9998 / 0.9999 | 1 | 2048 | 4.9× / 3.1× |
| 5 | `parareal` (= `multiple_shooting`) | `C=512,iters=4` (= L/C passes) | 0.845 | 1.479 | 1.0000 | 0.9998 / 0.9999 | 1 | 2048 | 3.4× / 2.1× |
| 6 | `carry_select` | `C=256,K=3` | 0.884 | 1.338 | 1.0000 | 0.9998 / 0.9999 | 1 | 552 | 3.2× / 2.3× |
| 7 | `map_scan` | `C=256,K=1,W=32` | 1.189 | 1.553 | 1.0000 | 0.9998 / 0.9999 | 1 | 555 | 2.4× / 2.0× |
| 8 | `serial_fused` (reference) | `-` | 2.847 | 3.089 | 1.0000 | 1.0000 | 1 | 2048 | 1× / 1× |
| 9 | `speculative_chunk` | in `C=128`; out `C=32` | 96.2 | 148.7 | 1.0000 | 0.9999 | 1 | ~L (1226–2111 rounds) | 0.03× / 0.02× |
| 10 | `serial_python` | `impl=own` | 194.7 | 192.0 | 1.0000 | 1.0000 | 1 | 2048 | 0.015× / 0.016× |

Run-to-run spread (separate processes): `serial_fused` 2.834–2.859 / 3.080–3.121 (9 and 8 runs);
`serial_triton` with the 8-bit tape 0.390–0.419 / 0.673–0.702 (7 runs each). Rank 1 and rank 2 are within that
spread at `in`. `serial_triton` is the simpler one (one kernel launch each way, no certificate), so it is ranked
first. After the code review and its fixes (section 6) every row was run again: same verdicts, times within
4 % of this table; `serial_triton` 0.416 / 0.676 ms against `serial_fused` 2.837 / 3.084 ms.

Every passing method had 0 wrong spikes on the real input in the rows of this table (the gate counts spikes
on the real input). Exact by construction, with 0 wrong spikes and a bit-identical membrane on all three
inputs: `serial_triton`, `spec_verify` with the bit certificate, `chunk_warm` and `parareal`. Not exact by
construction: `decoupled_threshold` (the same neuron in another order of float operations: 5 wrong spikes
of 50 M on the slow-leak input at `out`), `carry_select` (42 / 93 wrong spikes on the slow-leak input),
`map_scan` and row 2b (interval certificate: 0 wrong spikes in these rows, see section 6). No method that
changes the neuron passes the gate.

## 2. Recommendation

```
RECOMMENDED      = serial_triton, U=32, BLOCK=64, nw=2, tape=u8   (out: U=32, BLOCK=512, nw=8, tape=u8)
SERIAL_FUSED_MS  = 2.847 / 3.089
RECOMMENDED_MS   = 0.415 / 0.677        (best single run: 0.390 / 0.673)
SPEEDUP          = 6.9x / 4.6x
```

Other tapes of the same kernel: `tape=h16` 0.45–0.48 / 0.76 ms (6.0–6.3× / 4.1×, grad_cos 0.9998 / 0.9999);
`tape=h32` 0.47–0.52 / 0.88–0.91 ms (5.5–6.0× / 3.4–3.5×, gradient bit-identical to SpikingJelly's for x).
The float tapes at `in` were 5 % slower in the last hour than in the first, for old and new code alike.

Milestones:
- **M1 (≥ 2× at both sites): reached** 2026-11-22, about 00:30, by `serial_triton` (first config: 3.1× / 2.6×).
- M2 (≥ 5× at both sites): reached at `in` (6.9×). **Not reached at `out`** (4.6×). It is not reachable with
  float32 input and output: see "the floor".
- M3 (≥ 5× with seq_depth ≤ 64): **not reached.** An earlier version of this report claimed it at `in` for
  `spec_verify C=32,W=32,cert=interval,bwd=cold,Wb=32` (0.49–0.52 ms, 5.5–5.8×, 0 wrong spikes). That claim
  is withdrawn. The config has 32 + 32 sequential steps in its speculative pass, but its repair kernel then
  walks over the 64 chunks one after the other (one cheap check per chunk). Counted like the sweep of
  `carry_select`, the depth is 128, not 64. With a serial sweep no chunk plan gets below C + L/C ≥ 91 at
  L = 2048. And 128 holds on the real input only: on the counting input the repair kernel runs all 12096
  chunk pairs again, on the slow-leak input 11904 of 12096, one after the other. So on two of the three
  gate inputs the depth is about L. The gate itself forces this. The counting input (β = 1) needs the exact
  state through all 2048 steps, and the slow-leak input (β = 0.95) needs a warm-up of 64 steps or more for
  the gradient (`chunk_cold C=32,W=32`: grad_cos 0.90). A method with depth ≤ 64 on every gate input must
  be an exact scan of logarithmic depth like `map_scan`, and that is 2.4× slower than the serial kernel
  here. At `out` no method reaches 5× at any depth.

When to use something else than `serial_triton`:
- **Few neurons and long sequences** (B·D below ~12k and L above ~4k): `spec_verify` with the bit certificate
  (`cert=bit`, the default). With 768 neurons it is 2.4× faster at L = 8192 and 6× faster at L = 131072
  (section 4).
- **T ticks per token with a carried membrane**: `serial_ticks`. It reads x once per token and is 3× faster than
  `serial_triton` on the tick axis (section 4).
- **bf16 activations**: feed them to `serial_triton` directly, do not cast to float32 first (section 4).
- **Inside the model** the activations are batch-first `[B, L, D]`. Call the kernels with `bf=1`. A transpose
  before and after the neuron costs more than the neuron (0.135 / 0.267 ms per copy, four copies per step).

### What `serial_triton` is

The same serial recurrence as SpikingJelly, with bit-identical spikes and membrane on all three test inputs,
in an own fused Triton kernel. Three differences:
1. **Loads ahead.** Each program issues the loads of U = 32 future time steps first and computes them
   afterwards, so the memory round trips overlap. SpikingJelly's kernel waits for one load per time step
   (about 0.4 µs × 2048 in the forward pass, 0.95 µs × 2048 in the backward pass). This alone gives 3×.
2. **No second read of x.** The backward pass takes `x_t − v_{t−1}` from the saved membrane
   (it equals `(h_t − v_{t−1}) / q`).
3. **Small tape.** The forward pass saves the charged membrane `h` for the backward pass in 8 bits per step
   (sign, 4 exponent bits, 3 mantissa bits of `h − θ`). The forward pass itself is float32 and exact. Only the
   surrogate gradient sees the rounded value.

### The floor: why 5× at `out` is out of reach with float32 tensors

After the latency fix the kernel is limited by memory traffic. Measured at `out`, forward kernel only: reading
`x` 0.138 ms; plus writing the spikes 0.287 ms; plus the 8-bit tape 0.328 ms; with a float16 tape 0.37 ms. A
plain `copy_` of the tensor takes 0.267 ms. So time ≈ bytes moved / 1.4 TB/s (the pin bandwidth of the card is
1.79 TB/s). The kernels move bytes as fast as the plain copy. Triton's cache hints on the loads and stores
change nothing (11 variants, all within 1 %).

Any method must read `x` and the incoming gradient and write the spikes and the outgoing gradient. The floor
was measured with the real kernels: every tape store and load was sent to one cached row, so the tape costs
no memory traffic. The two kernels, each timed alone with its launch call: 0.329 ms at `in`, **0.585 ms at
`out`**. One whole pass needs 0.02–0.03 ms more than that sum (with the real tape: 0.677–0.684 against 0.654).
So a tape with **zero** bits would give 0.605–0.615 ms at `out`, 5.0–5.1×.

Every tape bit per step adds 0.0086 ms at `out`. 5× needs 0.618 ms, so the tape may have at most 1 bit per
step. The smallest tape that keeps `grad_cos ≥ 0.99` has 5 bits per step (simulated: 0.995; 4 bits gives 0.98
and fails). A 5-bit tape would give about 0.65 ms, 4.75×. So M2 at `out` is closed for this interface.

The same arithmetic explains the other rows: each extra pass over `x` (a warm-up, a Picard sweep, a candidate)
costs 0.07 ms at `in` and 0.14 ms at `out`. That is a third of the whole serial neuron.

### Where the time of one pass goes

Time stamps inside one forward+backward pass of `serial_triton` (`tape=u8`, B = 16):

| part of the pass | in | out |
|---|---|---|
| Python before the forward kernel starts (leak conversion τ → w → q, allocations, Triton launcher) | 0.07 ms | 0.07 ms |
| the two kernels on the GPU, plus a short tail of small operations | 0.33 ms | 0.61 ms |
| whole pass | 0.40 ms | 0.68 ms |

All Python work of one pass (two launch calls, allocations, leak conversion, autograd) is 0.23 ms. At B = 16
only the first 0.07 ms count: the GPU waits for the forward kernel in that time. The rest of the Python work
runs while the kernels run. So 10 % (`out`) to 17 % (`in`) of the pass is not kernel time. With few neurons
this turns around: see "micro-batch 2" in section 4.

## 3. Methods that failed the gate, and which input broke them

| method | best spike_agree in / out (real input) | what failed | why |
|---|---|---|---|
| `chunk_cold` | 1.0000 / 0.9999 (W ≥ 64) | `parity` in every config; `grad` on the slow leak for W ≤ 64 | With β = 1 a chunk that starts from v = 0 has the wrong parity half of the time. |
| `chunk_warm` (passes < L/C) | 1.0000 / 1.0000 | `parity` | Needs L/C passes when β = 1. |
| `parareal` (iters < L/C) | 1.0000 / 1.0000 | `parity` | The affine correction is right only modulo θ when β = 1. |
| `carry_lookahead` | 1.0000 / 0.9999 (K = 3) | `grad` on the slow leak (0.989) | Tables hand over candidate values, not the true carry. Parity is kept with K = 3. |
| `carry_select`, K = 1 | 1.0000 / 1.0000 | `parity` | One candidate cannot contain both parities. K = 3 passes. |
| `fpt` (Picard, K ≤ 32) | 1.0000 / 0.9942 (K = 32) | `grad` (slow leak, 0.13–0.75), `parity`; `spikes` for K ≤ 5 | Needs as many iterates as the neuron has memory. Sigmoid feedback is worse than hard feedback. |
| `pmbc` | 0.9998 / 0.9824 (8 rounds) | `grad`, `parity`, `spikes` (out) | Same iteration as `fpt` with hard feedback. |
| `newton_soft` (= DEER, quasi-DEER) | 0.9897 / 0.9795 | `spikes`, `grad`, `parity` | Converges to the sigmoid-relaxed neuron, a different neuron. |
| `cumsum_closed_form` (5 modes) | 0.8994 / 0.9489 (`cummax`) | `spikes`, `grad` on the real and slow-leak inputs; **exact on the counting input** | The identities need β = 1 and bounded input. |
| `sdn_predictor` | 0.9796 / 0.9782 | `spikes`, `grad`, `parity` | 10-token receptive field; fitted to one τ. |
| `psn` / `sliding_psn` / `spsn` | 0.94 / 0.91; 0.94 / 0.92; 0.87 / 0.88 | everything | No reset: 6–13 % wrong spikes, 28 % on the slow leak. |
| `psu` | 0.9259 / 0.9511 | everything | One reset estimate is not enough. |
| `mpe_psn` | 0.9513 / 0.9608 | everything | Uses only the previous input. |
| 16-bit membranes (SpikingJelly fed bf16 / fp16) | 0.9989 / 0.9991; 0.9998 / 0.9998 | `spikes` + `grad`; `grad` | Rounding the membrane changes spikes. SpikingJelly then also runs a Python loop (237 ms). |

`waveform_relaxation`, `multiple_shooting`, `deer` / `quasi_deer` / `elk` and `sigma_delta_lookahead` are the same
computations as other rows; `METHODS.md` says which. None of the failing methods is faster than
`serial_triton`, except `chunk_cold` at `in` in its first run (0.417 ms).

Three findings behind this table:
- **The counting input separates the methods.** With β = 1 nothing forgets the start state. Only methods that
  carry the true state through all 2048 tokens keep the parity: the serial loop, or a chunk method with a proof
  for every chunk boundary.
- **Sweep-based methods collapse into one serial pass.** Iterate k at time t needs iterate k−1 only at times t
  and t−1. So K Picard or Newton sweeps run as K stages inside one serial pass (`fpt impl=cascade`: 0.49 ms
  for K = 3 instead of 1.49 ms as three passes). That pass costs the same memory traffic as the exact neuron
  and gives a worse answer.
- **A certificate is cheaper than more candidates.** `spec_verify` runs every chunk once from a warm-up guess
  and proves it: the start state must be bit-equal to the true one. Chunks without proof are run again. This
  is exact for every input and costs 2–20 % over the serial kernel; `carry_select` and `map_scan` cost 2–3×.
  A weaker proof (the true state lies in the interval of start states that give the same spikes) needs a
  much shorter warm-up, but it is not exact by construction (section 6).

## 4. Occupancy crossover, length scaling, leak sweep, variants

### Occupancy at L = 2048

Site `in` (D 768), timing only. `ms_fwd_bwd` by number of neurons B·D:

| B·D | 768 | 2.3k | 7.7k | 25k | 100k | 400k |
|---|---|---|---|---|---|---|
| `serial_fused` | 1.362 | 1.443 | 2.692 | 3.093 | 4.648 | 16.151 |
| `serial_triton` (best of 3 configs, h16 tape) | 0.223 | 0.261 | 0.350 | 0.746 | 2.988 | 11.892 |
| `spec_verify` (best of 4 configs) | 0.324 | 0.343 | 0.364 | 0.843 | 3.107 | 12.203 |
| `chunk_cold` (best of 3, fails the gate) | 0.272 | 0.213 | 0.329 | 0.834 | 3.223 | 12.807 |
| speed-up `serial_triton` / `serial_fused` | 6.1× | 5.5× | 7.7× | 4.1× | 1.6× | 1.4× |

SpikingJelly's kernel is latency-bound up to about 25k neurons and bandwidth-bound from about 100k.
`serial_triton` is bandwidth-bound from about 8k–12k neurons. At L = 2048 no time-parallel method that passes
the gate beats `serial_triton` at any size.

### Where parallel-in-time does pay: few neurons and long sequences

Site `in`, timing only. Best `ms_fwd_bwd` of `serial_triton` and of `spec_verify` with each certificate:

| B·D | 768 | 1536 | 3072 | 6144 | 12288 |
|---|---|---|---|---|---|
| L = 8192, `serial_triton` | 0.773 | 0.890 | 0.966 | 1.076 | 1.389 |
| L = 8192, `spec_verify`, bit certificate | **0.317** | **0.525** | **0.559** | **0.969** | 1.423 |
| L = 8192, `spec_verify`, interval certificate | 0.265 | 0.336 | 0.534 | 0.929 | 1.648 |
| L = 32768, `serial_triton` | 3.365 | 3.493 | 3.732 | 4.107 | 5.504 |
| L = 32768, `spec_verify`, bit certificate | **0.645** | **1.048** | **1.900** | **3.521** | **5.298** |
| L = 32768, `spec_verify`, interval certificate | 0.610 | 1.143 | 1.875 | 3.325 | 6.285 |

At L = 131072 and 768 neurons: `serial_fused` 182 ms, `serial_triton` 13.4 ms, `spec_verify` 2.2 ms (bit
certificate; 2.1 ms with the interval certificate). The bit certificate is exact by construction; the
interval certificate is a little faster with few neurons but is not (section 6).

With few neurons the serial kernel has few programs, and each program runs its chain at about 35 ns per step.
Chunks turn that chain into many short ones. The crossover is at about 12k neurons for every length measured.
At L = 2048 the fixed cost of the kernel launches (about 0.2 ms) hides the effect. The interval values
were measured again after the certificate fixes of section 6: all within 4 % of this table.

### The repo's training shape (micro-batch 2)

`program.md` fixes B = 16. The configs of the repo train with a micro-batch of 2. Gate passed in every row:

| L = 2048, B = 2 | in ms | out ms |
|---|---|---|
| `serial_fused` | 1.413 | 2.119 |
| `serial_triton` (`U=64,BLOCK=32,nw=1,tape=u8` / `U=32,BLOCK=64,nw=2,tape=u8`) | 0.208 | 0.256 |
| `spec_verify C=64,W=64,cert=interval,bwd=cold,Wb=64,U=32,BLOCK=32,nw=1` | 0.229 | 0.235 |
| speed-up `serial_triton` | 6.8× | 8.3× |
| with 4 ticks per token: `serial_fused` on the tick axis | 9.913 | 10.873 |
| with 4 ticks per token: `serial_ticks` | 0.326 | 0.427 |

At this size the pass is limited by Python, not by the kernel. A probe measured the whole pass at
0.242 / 0.287 ms and the Python part of it at 0.229 / 0.235 ms (two launch calls, allocations, leak conversion,
autograd). The kernels alone need 0.08–0.11 ms each, launch call included. More kernel work cannot help here.
Times at this size also vary from process to process: the same `spec_verify` code gave 0.24 ms in one process
and 0.34 ms in the next (thread wake-ups). The table shows single runs at the fast level.

Levers on the Python part, measured and not built: a cached zero start state plus a direct call of the compiled
kernel (without Triton's argument binder) give 4–5 % at B = 2 and 4.1 % / 1.6 % at B = 16. That is inside the
run-to-run spread, and the direct call depends on Triton internals. The leak conversion inside the kernel
would save about 20 µs more, but q could then differ from PyTorch's by one rounding step. In the model the
leak parameter is `w`, so only the sigmoid remains of the conversion.

### Length scaling

Site `in`, B = 16; the gate passed at every length:

| L | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| `serial_fused` | 0.721 | 1.397 | 2.847 | 5.703 | 11.413 |
| `serial_triton` (h16 tape) | 0.155 | 0.284 | 0.452 | 0.859 | 1.624 |
| `spec_verify C=512,W=128` | 0.302 | 0.300 | 0.492 | 0.978 | 1.816 |
| `spec_verify C=32,W=32,cert=interval`, chunked backward | 0.323 | 0.282 | 0.500 | 0.957 | 1.788 |
| `chunk_warm C=L/2,p=2` | 0.240 | 0.290 | 0.565 | 1.076 | 2.124 |
| speed-up `serial_triton` / `serial_fused` | 4.7× | 4.9× | 6.3× | 6.6× | 7.0× |

At B = 16 everything is linear in L and there is no crossover. The two shortest columns of the chunked rows
vary by ±15 % between processes (0.29–0.37 ms): such short passes are limited by Python, like micro-batch 2.

### Leak sweep

Site `in`, Gaussian input, `chunk_cold C=256`. Wrong spikes of 25.2 M at W = 16 / 32 / 64 / 128:

| β | W = 16 | W = 32 | W = 64 | W = 128 |
|---|---|---|---|---|
| 0.5 | 0 | 0 | 0 | 0 |
| 0.9 | 33 746 | 12 484 | 1 605 | 40 |
| 0.95 | 137 368 | 91 923 | 40 727 | 7 446 |
| 0.99 | 174 006 | 163 029 | 144 062 | 112 959 |

The count falls more slowly than β^W. One spike that differs during the warm-up puts the membrane error back
to θ. `spec_verify` with the bit certificate is exact at every β.

### How much carry the gradient needs (online training: OTTT, SLTT)

Online training methods drop the temporal path of the backward pass. A probe kept the exact forward pass and a
carry over K future steps only. Cosine to the exact gradient for x:

| K (steps of carry) | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `in`, real input (β 0.50) | 0.864 | 0.967 | 0.992 | 0.9995 | 1.0000 | 1.0000 | 1.0000 |
| `out`, real input (β 0.80) | 0.605 | 0.773 | 0.863 | 0.947 | 0.992 | 0.9998 | 1.0000 |
| slow-leak input (β 0.95) | 0.313 | 0.432 | 0.516 | 0.635 | 0.778 | 0.909 | 0.983 |

The values match `sqrt(1 − β^(2K+2))` to three digits. `grad_cos ≥ 0.99` needs K ≥ 1.96 / |ln β| − 1: 2 steps at
β = 0.5, 8 at 0.8, 38 at 0.95. Truncation does not save time: a short carry moves the same bytes as the exact
backward kernel. It only makes the backward pass parallel in time (`spec_verify bwd=cold`).

A second probe rebuilt the membrane backwards in time instead of a tape (reversible-network idea). The rebuilt
value is off by more than θ after 22 steps (`in`) and 52 steps (`out`), because every reverse step divides by β.

### All 12 blocks of the model

`bench_inputs.pt` holds block 6 and validation windows 0–15. A probe ran the recommended kernels on the real
activations of all 12 blocks (24 neuron sites, learned τ 1.98–2.14 at `in`, 4.29–5.98 at `out`), captured on
windows 16–31:

| kernel | wrong spikes (24 sites) | smallest grad cosine in / out | in ms (min–max) | out ms (min–max) | chunks run again |
|---|---|---|---|---|---|
| `serial_triton`, `tape=u8` | 0 | 0.9995 / 0.9973 | 0.389–0.416 | 0.673–0.678 | — |
| `spec_verify`, bit certificate, `C=1024`, `W=128` / `256` | 0 | 0.9995 / 0.9973 | 0.424–0.455 | 0.737–0.747 | 0 at all sites |
| `spec_verify`, interval certificate, `C=32,W=32`, chunked backward | 0 | 0.9999 / 0.9989 | 0.494–0.528 | 0.93–1.16 | `in`: 10–46 of 12096; `out`: 1061–17629 of 24192 |

The firing rate of the study neuron is 2.6–20 % at `in` and 0.13–15.6 % at `out`. The membrane reaches 947 θ at
`out` (block 5).

### The linear scan (section D)

Eight ways to compute `k_t = β k_{t−1} + u_t` over `[2048, 16, D]`, forward only (first row: a plain copy):

| implementation | in ms | out ms |
|---|---|---|
| plain copy of the tensor (for scale) | 0.135 | 0.267 |
| fused serial kernel with loads ahead | **0.150** | **0.289** |
| chunked C×C lower-triangular matmul, shared β, C = 128, plus carry | 0.550 | 1.012 |
| one `torch.cumsum` over the time axis (no leak) | 0.730 | 0.759 |
| weighted cumsum in chunks of 64 plus carry | 0.853 | 1.722 |
| chunked matmul with one matrix per channel, C = 64 | 1.086 | 2.152 |
| FFT, zero-padded to 2560 | 1.830 | 3.891 |
| doubling scan, 11 rounds | 4.773 | 9.642 |
| PyTorch loop | 13.65 | 13.87 |

The fused serial pass is the fastest linear scan by 3.5×. All scan-based methods above were run with it.

### Variants of the neuron

Each against `serial_fused` with the same variant, B = 16:

| variant | `serial_fused` in / out | own kernel in / out | note |
|---|---|---|---|
| hard reset (the reset of the checkpoint), carried over tokens | 2.854 / 3.078 | `serial_triton` 0.416 / 0.675 | exact; 6.9× / 4.6× |
| `detach_reset=False` | 2.861 / 3.090 | `serial_triton` 0.421 / 0.675 | 6.8× / 4.6×; `tape=h32` gives grad_cos 1.0000 |
| 4 ticks per token, membrane carried over 8192 ticks | 11.446 / 12.082 | `serial_triton` 1.508 / 2.537 | 7.6× / 4.8× |
| the same, with a tick-aware kernel (x and spike count per token) | 11.446 / 12.082 | `serial_ticks` **0.520 / 0.920** | 22× / 13×; a different interface |
| bf16 input and spikes, float32 membrane | (2.847 / 3.089 with float32 tensors) | `serial_triton` 0.355 / 0.410 | 8.0× / 7.5×; a different interface |
| batch-first tensors `[B, L, D]` (`bf=1`), float32 | (2.847 / 3.089, time-first) | `serial_triton` 0.430 / 0.679; `serial_ticks` 0.531 / 0.945 | no transpose needed; at most 5 % slower than time-first |

## 5. In progress and how to resume

Nothing is running. Open ideas, none started:
- a verification of `spec_verify` without the serial sweep over the chunks, and an interval certificate that
  is exact by construction (redo from the last bit-exact state). Both would help on inputs like the real
  one only; on the counting and slow-leak inputs the depth stays about L (section 2);
- less Python around the two launches (measured: 4–5 % at B = 2, 2–4 % at B = 16; section 4);
- a 5-bit tape (about 4 % at `out`; it cannot reach 5×);
- the tick kernel with a reset per token (that is the neuron the repo trains today; it would replace
  `PLIFRate` with about a third of the memory traffic). This is outside the question of `program.md`.

Resume command (one config, from this folder; `python` = `/home/karthik/Desktop/spiking_mamba3/.venv/bin/python`):

```
python -B bench_neuron.py --method serial_triton --config "U=32,BLOCK=64,nw=2,tape=u8" --site in > bench.log 2>&1
grep "^RESULT" bench.log | cut -f2- >> bench_results.tsv
```

Other switches of `bench_neuron.py`: `--L`, `--B`, `--timing-only`, `--slow-tau`, `--ticks`, `--tick-fused`,
`--reset hard`, `--detach 0`, `--io bf16`, `--batch-first`. Several configs in one process: `--config "a;b;c"`.
Smoke check of all methods: `python -B neuron.py`.
Random stress test of `serial_triton`, `spec_verify` and `serial_ticks` against a plain PyTorch loop:
`python -B bench_neuron.py --stress 240` (add `--stress-seed 1` for other cases; exit status 1 if a case fails;
240 cases take 1–6 minutes, most of it kernel compilation on the first run).

## 6. Things to know before using these numbers

- The trained checkpoint uses a **hard** reset (`v_reset=0.0`), 4 ticks per token and a membrane reset per token.
  This study measures the **soft**-reset neuron carried over tokens, as `program.md` asks. Only the input
  activations and the learned `tau` come from the checkpoint. The hard-reset and tick variants are in section 4.
- The checkpoint has one scalar `tau` per layer, not a per-channel `tau`. All own methods take a per-channel `tau`
  and give a per-channel gradient.
- `cupy` is not installed, and SpikingJelly 2.0.0 does not use it any more. Its fused kernel is a Triton kernel,
  and that is the reference here.
- On the real input the soft-reset membrane is not bounded by θ. It reaches 4.7 θ at `in` and 88 θ at `out`,
  because one spike removes only θ per token. Methods that assume a membrane below θ fail for that reason.
- `grad_cos` is the smallest of four cosines (x and tau, real input and slow-leak input). The slow-leak input
  (β = 0.95) is the stricter one for every truncated method.
- The captured activations are exactly representable in bf16 (the model runs under bf16 autocast). The repo
  casts them to float32 before the neuron. A kernel that reads bf16 directly computes the same spikes with half
  the memory traffic. Do not let SpikingJelly's module see bf16 input: it then keeps the membrane in bf16,
  leaves its fused kernel, and the spikes change.
- The interval certificate of `spec_verify` (`cert=interval`) is built to prove equal spikes, not an equal
  membrane. A kept chunk keeps the membrane and the tape of the guessed start state. Three defects were found;
  two are fixed:
  1. Its first version gave 2 wrong spikes of 201 M on the 8192-tick run at `out`. This was explained as a
     true state at an interval edge that rounds to the other side of θ, and the interval got a rounding
     margin (now a tracked bound of the rounding error). The two cases went away, but see defect 3.
  2. The guess could be far from the true state: 0.8 θ on the real input at `out`, and several θ in a random
     test case (slow leak, start membrane far from 0). There the spikes were exact but the gradient was not
     (cosine 0.86). The certificate now also needs |true − guessed| ≤ `dmax`·θ at the chunk start (`dmax` =
     0.25 by default, 0 = bit certificate). The tape error is then at most `dmax`·θ·β^t. This costs nothing at
     `in` and 3 % at `out`.
  3. Not fixed: the spikes of the interval certificate are **not exact by construction**. The state after
     a kept chunk is known only up to float rounding. If a later membrane lands exactly on the threshold,
     its spike can come one step early or late. Found on the 8192-tick run at `out` (`C=32,W=32`): one
     spike one tick late, 2 mismatching positions of 201 M. The mismatches of defect 1 were the same event;
     the margin did not remove the cause. At `[2048, 16, D]` every run of the interval certificate had 0
     wrong spikes (all gate rows, all 24 sites of the model).
  The bit certificate has none of these cases: it is exact by construction (0 wrong spikes on the same
  tick run). Use `cert=bit` (the default) whenever spikes or gradients must be exact.
- Checked outside the gate: every exact method gives 0 wrong spikes when a sequence is run as two halves with
  `v0` = the last membrane of the first half, and its gradient for `v0` matches PyTorch autograd. Neuron counts
  that do not fill the last block are exact too.
- Stress test outside the gate (`bench_neuron.py --stress`): random cases (shapes, a different τ per channel, θ,
  start membrane, hard and soft reset, detached or not, batch-first, all tapes, block shapes, chunk plans)
  against a plain PyTorch loop, for `serial_triton`, `spec_verify` and `serial_ticks`. It compares spikes, the
  last membrane and the gradients (cosine and norm) for x, τ and the start membrane. Some cases pass x or the
  incoming gradient as a view that is not contiguous. 0 wrong spikes in every run. The second seed found
  defect 2 of the interval certificate (above). After all fixes: 6 seeds, 2240 cases, 0 failed.
- How exact the gradient is depends on the tape. For the leak gradient of one channel: `tape=u8` about 3 %
  relative error (more when the sum of that channel nearly cancels), `tape=h16` 0.1–0.3 %, `tape=h32` 1e-7.
  The gradient for x has a cosine of 0.9998 (`u8`) to 1.0000 (`h32`). If the leak is trained and matters, use
  `tape=h16`: it costs 0.04 ms at `in` and 0.08 ms at `out`.
- The own kernels need L to be a multiple of U (and of 4 for the 8-bit tape). They raise an error otherwise.
- An independent code review (read-only, after the benchmark was complete) found no error in the index
  arithmetic of the kernels, but found inputs for which the kernels gave wrong numbers without an error.
  All were reproduced and fixed, and the stress test now covers them:
  1. `x` or the incoming gradient as a view that is not contiguous. Example: `spikes.sum().backward()` gave
     a gradient with cosine 0.08. The kernels now make such tensors contiguous first.
  2. A `tau` with fewer than D values, for example the scalar of the checkpoint: wrong spikes. Now an error.
     Expand a scalar `tau` to `[D]`.
  3. Methods with the tape-gradient backward and L not a multiple of 32; unknown option strings. Now errors.
  No logged row is affected: the harness always passed contiguous `[L, B, D]` tensors and a `[D]` tau.
  `mem_gb` of the `serial_fused` rows before the fix is 0.3–0.6 GB too high (a cached module kept one tensor).
- The reference is timed fairly. `serial_fused` turns the per-channel `tau` into SpikingJelly's scalar on every
  call; that adds 87 µs (3 %) to its time. The own kernel pays its own conversion (22 µs before the launch).
  With each side owning its leak parameter, as in a model, the speed-up is the same: 2.738 against 0.390 ms
  at `in` (7.0×), 2.963 against 0.655 ms at `out` (4.5×).
- The 8-bit tape holds |h − θ| below 128 in absolute units and clamps larger values. On the real activations
  this is active in 5 of 12 blocks at `out` (membranes up to 947 θ) and has no measurable effect on the
  gradient (cosine 0.9996–0.9998 there): a clamped step has a surrogate near zero. The code is built for a
  threshold near 1.
- The clock times in the comment lines of `bench_results.tsv` between "00:50" and "10:50" are wrong: they were
  typed as estimates. A correction line in the file gives the true timeline. All work up to that line happened
  between 23:58 and 01:42.
- No linter was run: `ruff`, `mypy`, `pylint` and `codespell` are not installed in the venv, and this study
  installs nothing. The code was checked for syntax, unused imports and missing docstrings only.
