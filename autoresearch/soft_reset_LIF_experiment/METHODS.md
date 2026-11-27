# METHODS — the method ledger

One entry per method, written before the first line of code and updated after every attempt.
Numbers are `ms_fwd_bwd` at `[2048, 16, D]`, `in` = D 768, `out` = D 1536. Gate = `spike_agree >= 0.999`
AND `grad_cos >= 0.99` AND `parity_ok = 1`.

## Reference neuron

Source: `src/spiking_mamba3.py`, class `PLIFRate`, used in front of `in_proj` and `out_proj` of every block.

What the repo trains:

```
neuron.ParametricLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="m")
```

| setting | repo value | this study |
|---|---|---|
| class | SpikingJelly 2.0.0 `ParametricLIFNode` | same |
| `decay_input` | `True` (the class default) | same |
| `v_reset` | `0.0` = **hard** reset | `None` = **soft** reset (the study question) |
| surrogate | `ATan`, alpha = 2 | same |
| `detach_reset` | `True` | same |
| leak | one scalar `w` per layer, `1/tau = sigmoid(w)`, init `tau = 2` | same parameterisation; the interface takes `tau` per channel `[D]` |
| theta | `1.0`, not learnable | same |
| T-axis | T = 4 ticks per token, same input 4 times, membrane reset before every token, output = spike count / 4 | one step per token, membrane carried over all L = 2048 tokens, zero at t = 0, output = the spike |
| dtype | input cast to fp32, membrane fp32 | same |

Equations of the reference (SpikingJelly's order of operations, fp32, no FMA), with `q = sigmoid(w) = 1/tau`:

```
h_t = v_{t-1} + (x_t - v_{t-1}) * q            charge   (= (1-q) v_{t-1} + q x_t, so beta = 1 - q)
s_t = 1 if h_t >= theta else 0                  fire
v_t = h_t - theta * s_t                         soft reset; s_t is detached here
backward:  sg_t = (alpha/2) / (1 + (pi/2 * alpha * (h_t - theta))^2)
           gh_t = gs_t * sg_t + carry_{t+1};   gx_t = gh_t * q;   carry_t = gh_t - gh_t * q
           dL/dq = sum_t gh_t * (x_t - v_{t-1})
```

With `decay_input=False` (only used by the counting input, where the leak is removed):
`h_t = v_{t-1} - v_{t-1} * q + x_t`, `gx_t = gh_t`, `dL/dq = sum_t gh_t * (-v_{t-1})`.

Consequences used all over this study:
- For a fixed spike pattern, `v_0 -> v_t` is affine with slope `beta^t`. Two runs with the same input and
  different `v_0` come together at rate `beta` per step, until one spikes and the other does not; then their
  difference jumps by `theta`.
- With `detach_reset=True` the backward pass is a **linear** reverse recurrence. It needs only `h_t` from the
  forward pass. So only the forward pass is hard to parallelise.
- With `beta = 1` the neuron is a first-order sigma-delta modulator: the membrane is the input sum modulo theta.

Learned values in `best.pt` (block 6, used for the real input): `tau_in = 2.0156` (beta 0.504),
`tau_out = 4.9187` (beta 0.797). Over the 12 blocks: tau 1.98–2.14 at `in_proj`, 4.29–5.98 at `out_proj`.

## serial_fused
origin:        SpikingJelly 2.0.0, `ParametricLIFNode`, `step_mode='m'` (program.md A)
idea:          One fused GPU kernel per layer; the loop over L runs inside the kernel, one thread per neuron
               pair. In SpikingJelly 2.0 this kernel is Triton (no cupy backend exists any more).
exactness:     exact — it is the reference
seq_depth:     L
learnable tau: yes, but one scalar for the whole layer (the kernel sums the per-neuron gradient)
status:        ok
attempts:      1 (plus one re-run after a rounding fix in the result formatter; same numbers)
best config:   `-`: in 2.847 ms (fwd 0.886), out 3.089 ms (fwd 0.972); spike_agree 1.0000, grad_cos 1.0000, parity 1
verdict:       The reference. Its time barely changes from D 768 to D 1536: it waits for one memory load per
               time step (~0.4 us x 2048 forward, ~0.95 us x 2048 backward). It is latency-bound, not data-bound.

## serial_python
origin:        SpikingJelly Torch provider (`SJ_PLIF_CUDA_IMPLEMENTATION=torch`) / plain PyTorch loop (program.md A)
idea:          The same neuron as a Python loop over L. `impl=sj`: SpikingJelly's loop, with its hand-written
               backward loop. `impl=own`: ten lines of PyTorch with autograd and a per-channel leak.
exactness:     exact (bit-identical spikes and membrane to serial_fused in the sanity check)
seq_depth:     L
learnable tau: sj: scalar; own: per channel
status:        ok
attempts:      1
best config:   `impl=own`: in 194.7 ms, out 192.0 ms; `impl=sj`: in 203.5 ms, out 203.0 ms; all gate columns exact
verdict:       ~70x slower than the fused kernel. For the record only. `impl=own` confirms the per-channel tau gradient.

## serial_triton
origin:        own kernel (program.md A "serial_triton", F "GPU engineering"). The same idea was found later in the
               survey: Volkov's latency-hiding thesis (2016) and the tuned serial kernel of Hard Resets Compose.
idea:          The exact serial recurrence in an own fused Triton kernel, with a per-channel leak. Each program
               issues the loads of U future time steps first and only then computes them, so U memory
               round trips overlap instead of waiting one after the other (software pipelining).
               The backward pass takes x_t - v_{t-1} from the saved membrane, so it does not read x again.
               `tape` = how the forward pass saves the charged membrane h for the backward pass: float32 (`h32`),
               float16 (`h16`), or 8 bits (`u8`: sign, 4 exponent bits, 3 mantissa bits of h - theta, four time
               steps packed into one int32). The forward pass is float32 and exact in all three.
exactness:     exact (same operations in the same order as SpikingJelly, fp fusion off): spikes and membrane are
               bit-identical on all three inputs. Only the gradient sees the rounded tape.
seq_depth:     L
learnable tau: yes, per channel (per-neuron leak gradient summed over the batch axis)
status:        ok
attempts:      4. (1) U=8, fp32 tape: passed at once, 0.910 / 1.210 ms. (2) Grid of 56 configs per site (U 1-64,
               four BLOCK/warp pairs, tape h32/h16): all 112 rows pass with 0 mismatches. (3) After the kernel
               revision (`neuron_backup_6.py` is the version before it; 8-bit tape, bf16 I/O, hard reset, non-detached reset): regression check of the
               best h16 configs (0.447 / 0.743, unchanged) and a grid of 8 u8 configs per site, all pass.
               (4) Variants, see "Extended study" below.
best config:   in: `U=32,BLOCK=64,nw=2,tape=u8` **0.415 ms** median of 4 separate runs (0.390-0.419), 6.9x. out:
               `U=32,BLOCK=512,nw=8,tape=u8` **0.677 ms** median of 4 runs (0.673-0.698), 4.6x. spike_agree 1.0000, v_maxerr 0, parity 1, grad_cos 0.9996 / 0.9997.
               With the float16 tape: 0.447-0.452 / 0.743-0.747, grad_cos 0.9998 / 0.9999.
               With the float32 tape: 0.470 / 0.871, grad_cos 1.0000.
               All 12 blocks (probe on validation windows 16-31, tau 1.98-5.98): 0 wrong spikes at all 24 neuron
               sites; smallest gradient cosine 0.9995 at `in`, 0.9973 at `out`; 0.388-0.415 / 0.672-0.676 ms.
               A different tau in every channel (1.2 to 31): 0 wrong spikes, per-channel tau-gradient cosine 0.9995.
               Stress test (240 random cases against a plain PyTorch loop: shapes, per-channel tau, theta, start
               membrane, hard/soft, detached or not, batch-first, all tapes and block shapes; 103 of them this
               method): 0 wrong spikes, 0 errors.
               Accuracy of the leak gradient per channel: `tape=u8` ~3 % relative error (up to ~20 % for a channel
               whose gradient sum nearly cancels), `tape=h16` 0.1-0.3 %, `tape=h32` 1e-7.
verdict:       The fastest passing method at both sites, and the simplest fast one: one forward and one backward
               kernel launch. It is bandwidth-bound: kernel-only time at `out` is 0.653 ms (0.328 + 0.325) for
               0.9 GB of traffic; the Python and autograd overhead is 0.02-0.04 ms. The forward pass stops
               improving at U >= 8, the backward pass needs U >= 16. Each halving of the tape buys ~10 %.
               Nothing is left to gain at this interface except a smaller tape; 5x at `out` would need a tape of
               about 2 bits per step and copy-speed bandwidth.
               2026-11-22 05:18, last tuning and checks: (a) block-shape sweep 2, 24 shapes per site with the 8-bit
               tape: the best configs are unchanged (in `U=32,BLOCK=64,nw=2`, tied with 32/1 and 128/4; out
               `U=32,BLOCK=512,nw=8`); U=64 and BLOCK=1024 run out of registers (up to 6x slower). (b) Measured
               floor with no tape traffic: 0.329 / 0.585 ms, so a tape of zero bits would give 5.0-5.1x at `out`
               (see "where the time of one pass goes"). (c) After the code review the launchers make x and the
               incoming gradient contiguous and check the size of the leak; cost at most 1 % (A/B).
               (d) Stress test with 6 seeds, 2240 cases, 0 failed.

## chunk_cold
origin:        receptive-field truncation of chunked RNN / WaveNet inference (program.md A)
idea:          Cut the L tokens into chunks of C. Every chunk is its own program in the same kernel launch. A
               chunk starts W tokens early from v = 0, computes those W warm-up tokens without storing them, and
               then stores its C tokens. The backward pass does the same in reverse (W tokens of carry warm-up).
exactness:     approximate: membrane error ~ beta^W at each chunk start; no memory beyond W tokens
seq_depth:     C + W
learnable tau: yes, per channel
status:        ok (runs), fails the gate
attempts:      2. Attempt 1 crashed at C=32,W=64 with an illegal memory access: a warm-up longer than the history
               before the chunk read before t = 0. Fixed by clamping the warm-up. Attempt 2: grid C {32..512} x
               W {16..128}, 20 configs per site, all run.
best config:   none passes. Fastest: in `C=512,W=16` 0.417 ms; out `C=256,W=16` 0.791 ms.
               Real input: 0 spike mismatches at `in` for W >= 32 (beta 0.50); at `out` (beta 0.80) 26351
               mismatches at W=32, 21 at W=128. Slow leak (beta 0.95): agree 0.973 (W=16) to 0.9998 (W=128).
verdict:       Fails `parity` in every config: with beta = 1 nothing ever forgets the start state, so a chunk
               that starts from v = 0 has the wrong parity about half of the time. Small W also fails `grad` on
               the slow-leak input (0.83 at W=16, 0.99 at W=128). As fast as serial_triton at `in`, slower at
               `out`. Not usable for state tracking.

## chunk_warm
origin:        chunk-level Picard iteration (program.md A); parareal with a trivial coarse step
idea:          Pass 1 without outputs runs every chunk from v = 0 and keeps only the chunk end states. Pass p
               reruns every chunk from the end state its predecessor had in pass p-1. The last pass stores
               spikes and tape. The backward pass iterates the gradient carry between chunks the same way.
exactness:     approximate: error ~ beta^((passes-1) C); exact only after L/C passes when beta = 1
seq_depth:     passes * C
learnable tau: yes, per channel
status:        ok
attempts:      1. Grid C {32..1024} x passes {2,3}, 12 configs per site.
best config:   passing: `C=1024,p=2` in 0.565 ms, out 1.092 ms (exact: 2 chunks, 2 passes; seq_depth 2048).
               Non-passing: `C=256,p=2` in 0.558, out 1.089 with 0 mismatches on the real input.
verdict:       Passes only when passes = L/C, where it is exact by construction and has no depth advantage.
               Every other config fails `parity`. Each extra pass reads x again (+0.13 ms at `in`), so it is
               slower than serial_triton in every config. The same kernel with exact carries is the base of
               the exact chunk methods below.

## fpt
origin:        Fixed-point Parallel Training, Feng et al., ICML 2025 (arXiv:2506.12087)
idea:          Picard iteration on the spike train. Iterate k computes the membrane with the resets of iterate
               k-1: `v_k[t] = h_k[t] - theta * s_{k-1}[t]`, `s_k = H(h_k - theta)` (or a sigmoid with slope
               alpha_k), `s_0 = 0`. K iterations, final spikes = H of the last membrane.
               Implementation note: iterate k at time t needs only iterate k-1 at time t, so the K sweeps run as
               K stages inside ONE serial pass (`impl=cascade`). `impl=passes` runs one kernel pass per iterate.
exactness:     approximate (exact only when K reaches the longest chain of dependent spikes, up to L)
seq_depth:     K (K linear scans; each scan runs here as a fused serial pass because that is the fastest scan)
learnable tau: yes, per channel (backward = the standard linear reverse recurrence on the last iterate)
status:        ok (runs), fails the gate
attempts:      1. Grid K {1,2,3,5,8,16,32} hard, K {1,2,3,5} sigmoid, and K=3 as separate passes; 13 configs per site.
best config:   none passes. Closest: `K=32,relax=hard` in 1.021 ms, 0 mismatches on the real input (beta 0.50) but
               agree 0.969 on the slow leak and grad_cos 0.74; at `out` (beta 0.80) spike_agree 0.9942.
               `K=3,relax=hard`: in 0.492 ms, spike_agree 0.9903; out 0.771 ms, 0.9416.
               `K=3` as 3 separate passes: in 1.486 ms, out 1.948 ms (same spikes as the cascade).
verdict:       Fails `spikes`, `grad` and `parity`. Each iterate fixes only spikes whose causes are already fixed, so
               the number of iterates needed grows with the memory of the neuron: about 32 at beta 0.5, far
               more at beta 0.8-0.95, and L at beta 1. Odd iterates are upper bounds (too many spikes), even
               iterates lower bounds, so odd K is much worse on the slow-leak and counting inputs. The sigmoid
               feedback of the paper is worse than the hard one at every K. In one serial pass the cascade
               costs the same memory traffic as the exact neuron and is less accurate: no reason to use it here.

## pmbc
origin:        SPikE-SSM, parallel max-min boundary compression (arXiv:2410.17268)
idea:          Keep a lower and an upper bound on the spike train (start: all 0 and all 1) and tighten both with
               the reset trace of the other bound. Mathematically these are two Picard chains: the bounds after
               M rounds are iterate M started from all-zeros and from all-ones. The returned lower bound is
               iterate M of the all-zeros chain for even M. So `pmbc(iters=M)` = `fpt(K=M, relax=hard)` for even
               M; it is run once under this name with `init` chosen by the parity of M.
exactness:     exact when the two bounds meet (up to L rounds); approximate after a fixed number of rounds
seq_depth:     iters
learnable tau: yes, per channel
status:        ok (runs), fails the gate
attempts:      1. iters {2,3,4,8}, both sites.
best config:   none passes. `iters=8`: in 0.637 ms, spike_agree 0.9998, grad_cos 0.41, parity 0; out 0.799 ms, 0.9824.
verdict:       Same numbers as `fpt` hard with K = iters for even iters (it is the same iteration). Exact only when the
               two bounds meet, which needs as many rounds as the longest chain of dependent spikes.

## newton_soft
origin:        FidLIF (arXiv:2610.16924) = DEER (Lim et al. 2024) specialised to this neuron; soft-reset form
               derived here
idea:          Newton on the whole membrane trajectory of the sigmoid-relaxed neuron
               `F_t(v) = g(beta v + u_t)`, `g(h) = h - theta * sigmoid(alpha (h - theta))`:
               `v_k[t] = F_t(v_{k-1}[t-1]) + a * (v_k[t-1] - v_{k-1}[t-1])`, `a = beta g'(h)`.
               Iterate k at time t needs only iterates k and k-1 at time t-1, so the K Newton steps also run as K
               stages in one serial pass. Final spikes = H of the charged membrane of the last iterate.
exactness:     approximate: it converges to the relaxed neuron, not to the Heaviside neuron
seq_depth:     K
learnable tau: yes, per channel (standard reverse recurrence on the last iterate; memory independent of K)
status:        ok (runs), fails the gate
attempts:      1. K {2,3,5} x alpha {4,12}, both sites. No NaN.
best config:   none passes. `K=5,alpha=4`: in 0.867 ms, spike_agree 0.9769, grad_cos 0.87; out 1.035 ms, 0.9795.
               `K=5,alpha=12`: in 0.870 ms, 0.9897, grad_cos 0.35.
verdict:       Fails `spikes`, `grad` and `parity`. More Newton steps do not help: spike_agree stays at 0.977 for
               alpha = 4 from K = 2 to 5, because the iteration converges to the sigmoid-relaxed neuron, which
               is a different neuron. A sharper sigmoid (alpha 12) is closer on the real input but converges
               worse (v_maxerr 180-750). Not memory-bound here, so quasi-DEER was not needed.

## decoupled_threshold
origin:        PRF, Huang et al. 2024 (arXiv:2410.03530), decoupled reset
idea:          Pass 1: the membrane without reset, `k_t = beta k_{t-1} + u_t` (a linear scan). Pass 2: a loop that
               carries only the reset trace `r_t = beta (r_{t-1} + theta s_{t-1})` and fires when
               `k_t - r_t >= theta`. An exact rewrite of the neuron.
exactness:     exact in exact arithmetic (rounding differs from the direct recurrence)
seq_depth:     L (the reset loop stays serial)
learnable tau: yes, per channel
status:        ok
attempts:      1.
best config:   attempt 2 (pass 1 no longer writes spikes nobody reads): in `U=32,BLOCK=64,nw=2` 0.587 ms,
               out `U=16,BLOCK=256,nw=4` 1.007 ms. Attempt 1: 0.706 / 1.186 ms.
               spike_agree 1.0000, grad_cos 0.9998, parity 1. Real and counting inputs:
               0 mismatches. Slow leak at `out`: 5 mismatches of 50 M (ties at the threshold; `k - r` rounds
               differently from the direct recurrence).
verdict:       Passes, but it is 1.4-1.5x slower than serial_triton: the reset-free membrane is one more 4-byte stream to
               write and read. A scalar-only reset loop is not cheaper, because the loop was never the cost;
               the memory traffic is.

## psn / sliding_psn / spsn / psu / mpe_psn  (the no-reset family)
origin:        Fang et al. NeurIPS 2023 (PSN, sliding PSN); Yarga & Wood 2023 (SPSN); Li et al. 2024 (PSU);
               Chen et al. 2024 (MPE-PSN)
idea:          Make the neuron parallel by changing it. `psn`: no reset, spike = H(leaky sum - theta).
               `sliding_psn`: the same with only the last k inputs. `spsn`: no reset, spike ~ Bernoulli(sigmoid).
               `psu`: one reset estimate from the sigmoid of the reset-free membrane. `mpe_psn`: the previous
               membrane is estimated from the previous input alone. The learned weights of the papers are
               replaced by the leak weights beta^j, so these rows show what dropping the reset costs in fidelity.
exactness:     changes the neuron
seq_depth:     1 (psn, spsn, mpe_psn), k (sliding_psn), 2 (psu)
learnable tau: yes (psn, spsn, psu); sliding_psn through the taps; mpe_psn yes
status:        ok (all five run), all fail the gate
attempts:      1 each.
best config:   none passes. psn: in 0.495 ms, spike_agree 0.9410; out 0.772 ms, 0.9117. spsn: 0.8712 / 0.8842.
               psu: 0.9259 / 0.9511. Slow leak: agree 0.72 (psn, spsn), 0.93 (psu). parity 0 for all.
               sliding_psn (PyTorch conv1d): k=8 in 2.429 ms, 0.9413; k=32 in 3.235 ms, 0.9410; out 4.8-6.4 ms.
               mpe_psn (PyTorch): in 3.966 ms, 0.9513; out 8.008 ms, 0.9608.
verdict:       This is the floor of what dropping the reset costs: 6-9 % of the spikes are wrong on the real input and
               28 % on the slow leak; the membrane error is ~12 theta. They are not faster than the exact kernel.

## cumsum_closed_form
origin:        PMSN (arXiv:2408.14917) and the ANN-to-SNN conversion identity (QCFS, ST-BIF); the running-maximum
               form for signed input was derived in this study (the survey found it nowhere in the SNN literature)
idea:          Without leak the membrane is the input sum modulo theta, so the spike count up to t is a function of
               the prefix sum P_t. `mode=floor`: count = floor(P/theta) (exact for beta = 1, 0 <= u <= theta).
               `mode=cummax`: count = max(0, cummax(floor(P/theta))) (exact for beta = 1, u <= theta, signed input).
               `mode=preleak`: the same with P replaced by the leaky membrane without reset (an approximation for
               beta < 1). `mode=ternary`: signed spikes, count = floor(P/theta) with steps in {-1, 0, 1}.
               `mode=minplus`: count_t = t + min(0, min_{s<=t}(floor(P_s/theta) - s)) (exact for beta = 1 and
               non-negative input of any size; the min-plus dual of `cummax`, from the survey).
exactness:     exact for beta = 1 under the stated input condition; otherwise approximate / a different neuron
seq_depth:     1 (prefix sum, prefix max, difference)
learnable tau: only through the standard backward on the implied membrane; the count itself ignores the leak
status:        ok (runs), fails the gate
attempts:      1. Four modes, both sites.
best config:   none passes. All four modes are exact on the counting input (parity 1, 0 mismatches).
               Real input: `cummax` spike_agree 0.8994 (in) / 0.9489 (out); `preleak` 0.8818 / 0.9396;
               `floor` 0.8204 / 0.9035; `ternary` 0.6557 / 0.7269; `minplus` 0.8028 / 0.8993 (added later; exact
               on the counting input like the others). Time 3.4-7.3 ms (plain PyTorch operations).
verdict:       Exact only where the theory says so (no leak, input below theta). On the real input 5-18 % of the spikes
               are wrong and grad_cos is ~0.1, because the leak and inputs larger than theta break the identity
               (the membrane error reaches 10^3-10^4 theta: the prefix sum never forgets). The running-maximum
               form is the best of the four and the right tool for a leak-free layer with bounded input.

## sdn_predictor
origin:        SpikingSSMs, surrogate dynamic network (AAAI 2025, arXiv:2408.14909)
idea:          A tiny causal 1-D CNN (175 parameters: conv 1->6 k8, conv 6->6 k3, conv 6->1 k1) predicts the reset
               term r_t from the membrane without reset. Then `h = k - r_hat`, spike = H(h - theta). The CNN is
               trained on the real input of the site (first call), 600 Adam steps, fixed seed.
exactness:     approximate by design (receptive field 10 tokens; trained for one tau)
seq_depth:     1 scan + 3 conv layers
learnable tau: yes for the scan; the CNN is frozen and fits one tau
status:        ok (runs), fails the gate
attempts:      1. 175 parameters, 600 Adam steps on the real input of the site, final mse 0.0073 (in) / 0.0168 (out).
best config:   none passes. in 7.673 ms, spike_agree 0.9796, grad_cos 0.19; out 15.498 ms, 0.9782, grad_cos 0.35.
               Slow leak: agree 0.79 / 0.88. parity 0.
verdict:       Approximate by design and fitted to one tau. 2 % wrong spikes even in-sample; a receptive field of 10
               tokens cannot hold a reset trace with a slow leak. 17-20x slower than the exact kernel.

## speculative_chunk
origin:        Bullet Trains (ICML 2026, arXiv:2603.13283) in discrete time; also Tachyon's draft-verify-commit
idea:          Per chunk of C tokens: compute the membrane without further resets for the whole chunk in parallel,
               find the first threshold crossing of every neuron, apply the reset there, and repeat from that
               point. The reset only shifts one number per neuron (`k_t = S_t + beta^t A`, `A -= theta beta^-t*`).
exactness:     exact in exact arithmetic (prefix sums round differently from the serial loop)
seq_depth:     sum over chunks of (largest spike count of any neuron in the chunk + 1); up to L
learnable tau: yes, per channel (standard backward on the implied membrane)
status:        ok
attempts:      1. C {32, 64, 128}, both sites.
best config:   `C=128`: in 96.2 ms; `C=32`: out 148.7 ms. spike_agree 1.0000, grad_cos 0.9999, parity 1.
               Real and counting inputs: 0 mismatches. Slow leak: 7 (in) / 14 (out) threshold ties.
               Rounds on the real input: 1226-1587 (in), 2063-2111 (out), for L = 2048.
verdict:       Passes the gate (exact up to rounding ties) and is 200x slower than serial_triton. The number of rounds
               is set by the busiest neuron of each chunk; the real input has bursts (the membrane reaches
               88 theta at `out`, so some neuron fires at every step), so the rounds are ~L and nothing is
               gained. Speculating on "no spike" only pays for sparse firing.

## spec_verify
origin:        own combination; the same idea appears as speculative adders with error detection (Verma et al.
               2008), speculative FSM matching with validation (Luchaup et al. 2011), time-parallel simulation
               with fix-up (Lin & Lazowska 1991) and Tachyon's draft-verify-commit
idea:          `chunk_cold` plus a certificate. Every chunk starts W tokens early from v = 0, with the serial float
               operations. A chunk is proven correct if the membrane it starts with is bit-equal to the membrane
               its predecessor ends with (induction from chunk 0: equal state + equal input = equal future).
               A repair kernel then walks over the chunks of each neuron block and re-runs only the chunks whose
               start state was wrong. The result is always bit-identical to the serial loop.
exactness:     `cert=bit`: exact (bit-identical) for every input, by construction; only the amount of repair
               depends on the input. `cert=interval`: exact in every gate run, but NOT by construction
               (attempt 9: one moved spike in 2e8 neuron-steps on the tick input)
seq_depth:     C + W + L/C when no repair is needed: W + C steps of the speculative pass, then the repair kernel
               walks over the L/C chunks one after the other (one cheap check per chunk); + C per repaired chunk;
               L in the worst case (beta = 1). The column shows C + W + L/C for a chunked backward and L for
               `bwd=serial`. CORRECTED on 2026-11-22: rows before the correction line in the TSV show C + W only.
               `carry_select` was always counted with its sweep; the review found the inconsistency.
               In the notes below, "depth 64" and "seq_depth 64" mean C + W = 64 WITHOUT the sweep (true: 128).
learnable tau: yes, per channel
status:        ok
attempts:      3. Attempt 1: grid C {32..256} x W {32,64,128} x bwd {serial, exact}: all 48 rows pass with 0 mismatches on
               all three inputs. Attempt 2: repair counts per input, truncated chunked backward (`bwd=cold`),
               smaller repair blocks, larger C.
best config:   fastest (bit certificate): in `C=512,W=128,bwd=serial` 0.492 ms; out `C=512,W=256,bwd=serial` 0.812 ms (0 re-runs on the
               real input in both). Chunked backward: in `C=64,W=64,bwd=cold,Wb=64` 0.497 ms, seq_depth 128,
               0 re-runs. Depth 64: in `C=32,W=32,bwd=cold,Wb=32,BLOCK=64,nw=2` 0.604 ms (4.7x), out 1.198 ms,
               but 73-98 % of the (block, chunk) pairs are re-run on the real input, so the real depth is ~L.
verdict:       Exact for every input, bit-identical (v_maxerr 0). About 9 % slower than serial_triton, because the
               warm-up reads part of x twice. Re-runs on the real input: none for W >= 64 at `in` (beta 0.50);
               at `out` (beta 0.80) 520 of 2976 pairs at W=128 and none at W=256. On the counting and slow-leak
               inputs almost every chunk is re-run (no contraction within W), so there it degrades to one
               speculative pass plus one serial pass; still exact. Useful when sequential depth matters
               (few neurons, long sequences), not at B*D = 12k-25k. With the bit certificate a depth of 64
               is not usable: W <= 32 is too short for bit-equal states even at beta 0.5.
               `cert=interval` (attempt 3, 10 configs per site): a chunk is kept if the true entry state lies in
               the interval that gives the same spikes. This needs a much shorter warm-up. At `in`:
               `C=32,W=32,cert=interval,bwd=cold,Wb=32` 0.500 ms (5.69x), seq_depth 64, 0 mismatches on all three
               inputs, 0 re-runs on the real input, grad_cos 0.9949; `C=32,W=16,...` 0.507 ms, depth 48, 38 of
               3024 pairs re-run. At `out` (beta 0.80, membrane up to 88 theta) W=32 is too short: 3811 of 6048
               pairs re-run, 1.139 ms (2.7x); `C=512,W=64,cert=interval` 0.904 ms with 26 of 288 re-run.
               The interval certificate proves equal spikes, not an equal membrane: v_maxerr is up to 0.5-0.8
               at chunk starts (it decays by beta per token) and the tape carries the same error.
               M3 was claimed here at `in` (5.5-5.8x at "depth 64"). That claim is WITHDRAWN: with the
               verification sweep the depth of that config is 32 + 32 + 64 = 128 (see seq_depth above).
               Attempt 4 (8-bit tape): in `C=1024,W=128,bwd=serial,tape=u8,U=32` 0.425 ms; out
               `C=1024,W=256,bwd=serial,tape=u8,U=32,BLOCK=512,nw=8` 0.750 ms (medians of 4 runs).
               Attempt 5 (rounding margin): the interval certificate gave 2 wrong spikes of 201 M on the 8192-tick
               run at `out`: a true state at an interval edge can round to the other side of theta. The interval
               now stays 5e-7 (|h| + theta) / beta^t away from every edge. After that: 0 wrong spikes in all
               re-runs; the depth-64 config costs 0.515 ms at `in` (was 0.493).
               Where it wins (timing only, D 768): with fewer than ~12k neurons and more than ~4k steps the serial
               kernel waits again (too few programs), and certified chunks are faster: B*D = 768: L = 8192
               0.265 vs 0.773 ms, L = 32768 0.660 vs 3.365 ms, L = 131072 2.212 vs 13.403 ms (6x).
               Attempt 6 (carry kernel): the exact chunked backward (`bwd=exact`) first built a [D, n, n] matrix
               over the n = L/C chunks (28 ms and 4.7 GB at n = 512). A small kernel now walks over the chunks
               (`cin[c] = local[c+1] + beta^C cin[c+1]`). Depth 64 with exact gradients:
               `C=32,W=32,cert=interval,bwd=exact,tape=u8,U=32,BLOCK=64,nw=2` in 0.585 ms (was 1.012), out 1.077 ms
               (was 1.984), grad_cos 0.9996. Long and narrow, B*D = 768, `bwd=exact`: L = 32768 0.610 ms,
               L = 131072 2.062 ms. `bwd=exact` refuses the hard and the non-detached reset (there the carry factor
               depends on the spikes); use `bwd=serial` or `bwd=cold` for those.
               All 12 blocks (probe on validation windows 16-31): bit certificate 0 wrong spikes and 0 re-runs at
               all 24 sites; interval certificate at depth 64: 0 wrong spikes at all 24 sites, 6-26 of 12096 pairs
               re-run at `in`, 924-14593 of 24192 at `out`; 0.491-0.524 ms at `in`.
               Attempt 7 (offset bound, 2026-11-22 04:22): a random stress test (seed 1, 400 cases) found 2
               cases where `cert=interval` gives exact spikes but a wrong gradient: cosine for x 0.86 (L=24, D=1,
               tau 27, |v0| up to 5.8, W=0) and for v0 0.977. With `cert=bit` both give 1.0000. Cause: a kept
               chunk keeps the tape of the GUESSED start state. The interval only proves equal spikes, so the
               guess can be several theta away from the true state (slow leak, start membrane far from 0).
               Fix: new argument `dmax` (default 0.25). The interval certificate keeps a chunk only if also
               |true start - guessed start| <= dmax * theta. Then the tape of a kept chunk is off by at most
               dmax * theta * beta^t. `dmax=0` is the bit certificate; a huge `dmax` is the old behaviour.
               Result (2026-11-22 04:27): both stress cases now give cosine 1.0000 / 0.9998; all 48 interval cases of two
               seeds have cosine >= 0.9992; the stress test passes (seed 0: 240 cases, seed 1: 400 cases, 0 failed).
               Gate re-run, all PASS, 0 wrong spikes: depth 64 `C=32,W=32,cert=interval,bwd=cold,Wb=32,U=32,
               BLOCK=64,nw=2` in 0.517 / 0.523 / 0.493 ms (3 runs), out 0.996 / 0.999 / 1.022 ms; v_maxerr
               2.1e-3 (in), 0.20 (out, was 0.80); grad_cos 0.9949 (unchanged, set by the truncated backward).
               `bwd=exact,tape=u8`: 0.578 / 1.088 ms. `C=64,W=64`: exact 0.557 / 0.971, cold 0.496 / 0.878.
               `C=512,W=64,cert=interval`: 0.532 / 0.889. Cost of the bound on the real input, depth 64: `in`
               none (10 of 12096 pairs re-run for every dmax from 1e9 down to 0.02; offsets there are ~2e-3);
               `out` 5495 pairs re-run without bound, 7342 with 0.25, 12645 with 0.05 (0.967 / 0.996 / 1.084 ms).
               Bit certificate unchanged: 0.426 / 0.739 ms. All 12 blocks again (windows 16-31): 0 wrong spikes,
               8-30 of 12096 pairs re-run at `in` (0.490-0.499 ms), 1059-17622 of 24192 at `out` (0.93-1.14 ms),
               smallest cosine 0.9999 / 0.9988. The time of the depth config stands: 0.49-0.52 ms.
               Attempt 8 (after the code review, 2026-11-22 04:55): (a) seq_depth corrected, M3 withdrawn (above). A
               verification without the serial sweep is possible in principle: a local test per chunk pair
               (`vout[c-1] - vstart[c]` plus the largest possible offset must lie inside the interval of chunk c)
               proves all chunks by induction in one parallel step, and failed chunks could be re-run in
               bounded parallel rounds. Depth W + C + 1 + rounds * (C + 1). Not built: with L = 2048 it needs
               C = W = 16 and at most one round, the chunked backward needs C + Wb <= 64 with Wb >= 38 for the
               slow-leak gate, and one block that needs a second round is back at the serial sweep.
               (2026-11-22 05:51) A stronger reason not to build it: the depth would be small on the real input only.
               On the gate's counting input the repair kernel re-runs 12096 of 12096 chunk pairs, on the
               slow-leak input 11904 of 12096 (`C=32,W=32`, site in). A certificate cannot pass where the
               neuron does not forget within W steps. Without a certificate the same plan fails the gate
               (`chunk_cold C=32,W=32`: grad_cos 0.90 on the slow leak, parity lost). So no chunk method of
               this family has depth <= 64 on all three gate inputs.
               (b) A kept chunk now also keeps its GUESSED entry state for the backward pass. Before, the true
               entry state was stored next to the tape of the guess; the leak term of the first step of the
               chunk was then off by offset * beta / q instead of offset. (c) The rounding margin of the
               interval is now a tracked bound: err_t = beta * err_{t-1} + 2^-24 (2 |h - v_prev| + 2 |h| +
               theta), margin = max(4 err_t, old fixed margin) / beta^t. The fixed margin was tuned on
               one failure and does not grow with large inputs.
               Attempt 9 (2026-11-22 05:24): the 8192-tick run at `out` with `C=32,W=32,cert=interval` has 2 mismatches
               again, with the tracked margin. A probe found the cause, and it is not the interval: one spike
               comes one tick late. The membrane entering its chunk is one float32 rounding step below the true
               one, and the true h of that step is exactly theta. The state after a KEPT chunk is `stored end +
               beta^C * offset`; that equals the true float state only up to rounding. A chunk that is run
               again from it is exact except at a tie with the threshold. So `cert=interval` is not exact by
               construction: 0 wrong spikes in every run at [2048,16,D] and at all 24 sites of the model, one
               moved spike per 2e8 neuron-steps on the tick input (which repeats every token 4 times). The
               mismatches of attempt 5 were the same event; the margin did not remove the cause. `dmax=0`
               (the bit certificate) gives 0 wrong spikes on the same run. Not repaired: a sound version must
               redo all chunks since the last bit-exact state and carry an uncertainty of the state.
               RECOMMENDATION CHANGED: use `cert=bit` (exact by construction). For few neurons and long
               sequences it is within 5-20 % of the interval certificate (768 neurons: L = 8192 0.317 vs
               0.265 ms, L = 32768 0.645 vs 0.610 ms, L = 131072 2.171 vs 2.062 ms).

## parareal  (also: multiple_shooting)
origin:        Lions, Maday, Turinici 2001; Gander & Vandewalle 2007. Newton multiple shooting on the chunk boundary
               states is the same iteration here, because the Jacobian of a chunk is beta^C almost everywhere.
idea:          Fine propagator F = the exact serial chunk (all chunks in parallel). Coarse propagator G = the chunk
               without reset, which is affine: G(v) = beta^C v + c. Update of the chunk entry states:
               `U[c+1] <- F(U_old[c]) + beta^C (U[c] - U_old[c])`. The sweep over the chunks is a linear recurrence
               and is done in closed form with a small triangular matrix. `iters` = number of fine passes.
               With beta^C ~ 0 this is `chunk_warm`.
exactness:     approximate after a fixed number of iterations; the fixed point is the exact serial result
seq_depth:     iters * C
learnable tau: yes, per channel (the backward pass is linear, so one coarse correction makes it exact)
status:        ok
attempts:      1. C {64,128,256,512} x iters {2,3,4}, 9 configs per site.
best config:   passing: `C=512,iters=4` (= L/C fine passes, exact by construction): in 0.845 ms, out 1.479 ms.
               Non-passing, 0 mismatches on the real input: `C=512,iters=2` in 0.718 ms, out 1.221 ms.
verdict:       Passes only with iters = L/C. The affine coarse correction helps the counting input (9324 wrong spikes
               instead of 3.2 M for chunk_warm at C=512, 2 passes) but cannot make it exact: with beta = 1 the
               correction is right only modulo theta. For beta < 1 it equals chunk_warm. Every fine pass reads
               x again, so it is slower than serial_triton in every config. Multiple shooting = same rows.

## waveform_relaxation
origin:        Lelarasmee, Ruehli, Sangiovanni-Vincentelli 1982
idea:          Iterate on the reset term. Jacobi (all time steps use the reset of the previous sweep) is exactly the
               Picard iteration `fpt relax=hard`. Gauss-Seidel in time (each step uses the newest reset) is the
               serial loop, `serial_triton`. Block Gauss-Seidel over chunks is `chunk_warm` with passes = L/C.
exactness:     Jacobi: approximate for a fixed number of sweeps. Gauss-Seidel: exact.
seq_depth:     Jacobi: K. Gauss-Seidel: L.
learnable tau: yes
status:        ok (same computations as the rows of `fpt`, `serial_triton` and `chunk_warm`; not run again)
attempts:      0 extra
best config:   Jacobi = `fpt K=32,relax=hard`: in 1.021 ms, fails grad+parity. Gauss-Seidel = `serial_triton`:
               in 0.452 ms, out 0.747 ms, passes.
verdict:       Gauss-Seidel needs one sweep and is exact; Jacobi needs as many sweeps as the neuron has memory.
               On this GPU a sweep costs the same in both, so Gauss-Seidel (the serial kernel) wins.

## deer / quasi_deer / elk
origin:        Lim et al. 2024 (DEER); Gonzalez et al. 2024 (quasi-DEER, ELK)
idea:          Newton over the whole trajectory with the Jacobian solve as a scan. For a scalar state per channel the
               diagonal (quasi) Jacobian is the full Jacobian, so quasi-DEER = DEER = `newton_soft`. ELK adds damping
               against Jacobians larger than 1; here the slope is at most beta <= 1.
exactness:     approximate (relaxed neuron)
seq_depth:     K
learnable tau: yes
status:        ok as `newton_soft`; quasi-DEER and ELK are skipped (same computation; `newton_soft` was not
               memory-bound: 0.6 GB at `in`, and it never diverged)
attempts:      0 extra
best config:   see `newton_soft`
verdict:       see `newton_soft`

## carry_select
origin:        carry-select adder (Bedrij 1962); enumerative FSM execution (Mytkowicz et al. 2014); unrolled
               speculative delta-sigma modulators (Johansson 2014)
idea:          Pass 1 runs every chunk from K candidate entry states (a warm-up guess plus offsets
               `spread * theta * k`) and keeps only the K end states. A sweep over the L/C chunks then picks, for the
               true entry state, the nearest candidate and corrects its end state with the affine term
               `beta^C (v - candidate)`. Pass 2 runs all chunks from the selected entry states and stores.
exactness:     approximate (error set by K and by beta^C); exact when the true carry is one of the candidates
seq_depth:     W + C (pass 1) + L/C (selection sweep) + C (pass 2)
learnable tau: yes, per channel (standard backward on the tape of pass 2)
status:        ok
attempts:      1. C {64,256} x K {1,3,5}, W=32, spread 0.5.
best config:   passing: `C=256,K=3` in 0.884 ms, out 1.338 ms (parity 1; 42 / 93 wrong spikes on the slow leak).
               `K=1` fails parity (it is chunk_cold with a second pass).
verdict:       K = 3 candidates half a threshold apart contain the true carry of the counting input, so parity is
               exact; one candidate is not enough. Each candidate is one more read of x (+0.07 / 0.14 ms), and
               the selection sweep in PyTorch costs ~10 small operations per chunk. 2x slower than serial_triton.

## carry_lookahead
origin:        carry-lookahead / parallel-prefix adders (Ladner & Fischer 1980); FSM prefix over transition tables
               (Hillis & Steele 1986)
idea:          The same K candidates. Each chunk becomes a table "candidate index in -> nearest candidate index
               out" for the next chunk. Tables compose, so a prefix scan over the chunks (log2(L/C) rounds of
               gathers) gives the candidate index of every chunk entry without a sequential sweep.
exactness:     approximate (entry states are snapped to candidates); exact when the carries lie on the candidates
seq_depth:     W + C + log2(L/C) + C
learnable tau: yes, per channel
status:        ok (runs), fails the gate
attempts:      1. C {64,256} x K {1,3}.
best config:   none passes. `C=256,K=3`: in 0.737 ms, out 1.287 ms, parity 1, 0 / 2133 wrong spikes on the real input,
               grad_cos 0.989 (60296 wrong spikes on the slow leak).
verdict:       The table scan replaces the sweep over chunks by log2(L/C) gathers and keeps parity with K = 3. But a
               table can only hand over candidate values, not the true carry, so every chunk starts from a
               snapped state. With a slow leak that error does not die inside the chunk: `grad` fails.

## map_scan
origin:        Hard Resets Compose (repository, 2026); symbolic chunk summaries (Raychev et al. 2015); the validity
               interval form is from the survey of this study (LITERATURE.md, C)
idea:          For a fixed spike pattern the chunk map is affine with slope beta^C. Pass 1 runs each chunk from K
               guesses and records, per guess, the interval of entry states that give the same spikes
               (`lo = max over spike steps of (theta - h_t)/beta^t`, `hi = min over non-spike steps`) and the end
               state. A piece is (anchor, lo, hi, offset, slope). Pieces compose associatively; a prefix scan
               over the chunks gives every entry state. Pass 2 replays all chunks from these entry states and
               tracks the intervals again; a repair sweep re-runs a chunk only if the true entry state left the
               interval of the replay (then the spikes could differ).
exactness:     exact spikes (interval certificate + repair); the membrane differs from serial by rounding only
seq_depth:     W + C + log2(L/C) + C, + C per repaired chunk
learnable tau: yes, per channel
status:        ok
attempts:      1. C {64,256} x K {1,3} x W {16,32}, with and without the repair sweep.
best config:   `C=256,K=1,W=32`: in 1.159 ms, out 1.489 ms; spike_agree 1.0000 on all three inputs, parity 1,
               grad_cos 0.9998. Real input: every entry certified at `in`, 1.4 % uncertified at `out`, no
               re-run needed in either. K = 3 certifies every entry of the counting input.
verdict:       Works and is exact in the spikes, but costs K reads of x for the maps, a scan in PyTorch, and one
               replay: 2.6x slower than serial_triton. On the slow leak 37-60 % of the composed entries are not
               certified (too many pieces for K = 1..3) and the repair sweep fixes them. The useful part is
               the interval certificate itself: put into `spec_verify` it removes the need for bit-equal
               states (see `spec_verify cert=interval`). Agrees with the Hard Resets Compose measurements:
               the scan does not pay when the serial kernel already fills the GPU.
               2026-11-22 04:27: the replay's repair sweep now uses the same offset bound as `spec_verify` (0.25 theta), so a
               kept chunk's tape is at most 0.25 theta off. Re-run, PASS: `C=256,K=1,W=32` 1.189 / 1.553 ms.
               seq_depth corrected on 2026-11-22: with the repair sweep it is W + C + log2(L/C) + C + L/C
               (555 for the best config, the rows before the correction show 547).

## sigma_delta_lookahead
origin:        look-ahead / block-parallel sigma-delta modulators (time-interleaved: Khoini-Poorfard, Lim, Johns
               1997; unrolled and speculative: Johansson 2014; trellis: Kato 2002, Harpe et al. 2003)
idea:          The leak-free neuron is a first-order sigma-delta modulator, the membrane is its error state. The
               block-parallel modulators of the literature come in three kinds: (1) time-interleaved block filters,
               which keep a delay-free quantizer chain (exact, no depth gain); (2) unrolled modulators that
               precompute both outcomes of each bit and select like a carry-select adder (exact, 2^N work);
               (3) trellis / look-ahead encoders, which change the bit stream. Kind (2) with the enumeration done
               over states instead of bit patterns is `carry_select`; with exact pieces it is `map_scan`.
exactness:     kind 1 and 2 exact, kind 3 changes the output
seq_depth:     as `carry_select`
learnable tau: as `carry_select`
status:        skip (no separate implementation: it is the same computation as `carry_select` / `map_scan`; kind 1
               gives no gain and kind 3 changes the neuron)
attempts:      0
best config:   see `carry_select` and `map_scan`
verdict:       The correspondence is exact for beta = 1 and is logged in LITERATURE.md, section D. For a leaky
               neuron the error state is a real number, so "exhaustive look-ahead" means enumerating affine
               pieces of the chunk map, which is `map_scan`.

# Section F and the extended study

These are not new neuron methods. They are measurements of the top methods under other sizes and variants.
Every number has rows in `bench_results.tsv` (notes column: `L=..,B=..`, `slow_tau=..`, `VARIANT ..`,
`TIMING ONLY`).

## F: occupancy crossover
origin:        program.md F
idea:          Same D = 768, batch B from 1 to 520, so B*D from 768 to 400k neurons, at L = 2048, 8192, 32768 and
               131072. Timing only (the gate was run at B = 16 and at B = 2).
status:        ok
best config:   L = 2048, ms_fwd_bwd at B*D = 768 / 2.3k / 7.7k / 25k / 100k / 400k:
               serial_fused 1.362 / 1.443 / 2.692 / 3.093 / 4.648 / 16.151
               serial_triton (best of 3 configs) 0.223 / 0.261 / 0.350 / 0.746 / 2.988 / 11.892
               spec_verify (best of 4) 0.324 / 0.343 / 0.364 / 0.843 / 3.107 / 12.203
               chunk_cold (best of 3, fails the gate) 0.272 / 0.213 / 0.329 / 0.834 / 3.223 / 12.807
               Longer sequences, serial_triton vs spec_verify at B*D = 768 / 1536 / 3072 / 6144 / 12288:
               L = 8192:  0.773 vs 0.265; 0.890 vs 0.336; 0.966 vs 0.554; 1.076 vs 0.929; 1.389 vs 1.423
               L = 32768: 3.365 vs 0.660; 3.493 vs 1.048; 3.732 vs 1.875; 4.107 vs 3.325; 5.504 vs 5.298
               L = 131072, B*D = 768: serial_fused 182.4; serial_triton 13.40; spec_verify 2.212
verdict:       SpikingJelly's kernel is latency-bound up to ~25k neurons (time almost flat) and bandwidth-bound
               from ~100k. serial_triton is bandwidth-bound from ~8k-12k neurons; its speed-up over serial_fused
               falls from 6-8x (below 25k neurons) to 1.4x at 400k, where both kernels wait for memory.
               Below ~12k neurons the serial chain itself is the limit (~35 ns per step per program). At
               L = 2048 this hides behind the fixed cost of a launch (~0.2 ms), so nothing beats serial_triton
               there. From L ~ 4k on, certified chunks (`spec_verify`) win: 3x at 768 neurons and L = 8192,
               6x at L = 131072. The crossover is at ~12k neurons for every length measured.

## F: fuse the two neuron sites of a layer into one launch
origin:        program.md F
idea:          One kernel launch for the `in_proj` and `out_proj` neurons of a layer.
status:        skip (measured bound instead)
best config:   -
verdict:       The two sites have different widths and different inputs; a fused launch would save only launch
               and autograd overhead. That overhead is 0.02-0.04 ms per site for forward+backward (kernel-only
               0.371 ms vs full call 0.411 ms at `in`), against 1.06 ms for both sites. Upper bound of the gain:
               4 %. The memory traffic, which is the cost, does not change.

## F: fuse pass 1 and pass 2 of chunk_warm
status:        skip (the passes need the end states of all chunks of pass 1 before pass 2 can start, and pass 1
               is a full read of x; `spec_verify` replaces pass 1 by a warm-up inside the same launch, which is
               the fused form and is measured)

## F: fp16 / bf16 membranes
origin:        program.md F
idea:          Keep the membrane in a 16-bit float.
status:        ok (measured with SpikingJelly itself: its module keeps the membrane in the dtype of the input)
best config:   serial_fused fed bf16: spike_agree 0.9989 / 0.9991, grad_cos 0.83 / 0.81, 238 ms.
               serial_fused fed fp16: spike_agree 0.9998 / 0.9998, grad_cos 0.87 / 0.73, 237 ms.
verdict:       The gate does not survive: bf16 fails `spikes` and `grad`, fp16 fails `grad`. SpikingJelly also
               leaves its fused kernel in this mode and runs a Python loop. Keep the membrane in float32.

## F: bf16 input and spikes with a float32 membrane
origin:        own extension (the repo runs under bf16 autocast and casts to float32 before the neuron)
idea:          The kernel reads x in bf16 and writes the spikes in bf16; the membrane and all arithmetic stay
               float32. The gradients in and out are bf16 too. This halves the four large streams.
status:        ok
best config:   in `U=32,BLOCK=256,nw=4,tape=u8` 0.355 ms; out `U=32,BLOCK=128,nw=2,tape=u8` 0.410 ms; 0 mismatches
               against the float32-membrane reference on the same bf16 input; grad_cos 0.9996 / 0.9997.
               With fp16 input: 0.359 / 0.439 ms.
verdict:       The captured activations are 100 % exactly representable in bf16 (they come from a bf16 autocast
               forward), so on the real input this is the same computation as the float32 study, with half the
               traffic. Against the float32 serial_fused (2.847 / 3.089 ms) it is 8.0x / 7.5x. It is a different
               interface (16-bit tensors), so it is not counted for M2. Block size matters more here: 1 element
               of 2 bytes per thread (BLOCK=64, nw=2) is slow (0.56-0.68 ms).

## Extended: length scaling
status:        ok
best config:   site `in`, B = 16, ms_fwd_bwd at L = 512 / 1024 / 2048 / 4096 / 8192 (gate passed at every length):
               serial_fused 0.721 / 1.397 / 2.847 / 5.703 / 11.413
               serial_triton (h16) 0.155 / 0.284 / 0.452 / 0.859 / 1.624
               spec_verify C=512,W=128 0.302 / 0.300 / 0.492 / 0.978 / 1.816
               spec_verify interval C=32,W=32, chunked backward 0.323 / 0.282 / 0.500 / 0.957 / 1.788
               chunk_warm C=L/2, p=2 0.240 / 0.290 / 0.565 / 1.076 / 2.124
verdict:       Everything is linear in L. serial_triton is 4.7x (L=512) to 7.0x (L=8192) faster than serial_fused.
               No crossover with length: the chunked methods are bandwidth-bound too and move more bytes.

## Extended: leak sweep
status:        ok
best config:   site `in`, Gaussian input, chunk_cold C=256. Wrong spikes of 25.2 M at W = 16 / 32 / 64 / 128:
               beta 0.5: 0 / 0 / 0 / 0. beta 0.9: 33746 / 12484 / 1605 / 40. beta 0.95: 137368 / 91923 / 40727 / 7446.
               beta 0.99: 174006 / 163029 / 144062 / 112959.
               chunk_warm C=256, p=2 / p=3: beta 0.9: 0 / 0. beta 0.95: 243 / 0. beta 0.99: 53908 / 12012.
verdict:       The predicted membrane error beta^W holds only while no spike differs. As soon as one spike differs
               during the warm-up the error is back at theta, so the count of wrong spikes falls more slowly than
               beta^W (about beta^(W/2) at beta 0.9), and v_maxerr is theta whenever any spike differs. Truncation
               is safe at beta 0.5 from W = 32 and never safe at beta >= 0.99 within L = 2048.
               spec_verify with the bit certificate is exact at every beta.

## Extended: detach_reset=False
status:        ok
best config:   serial_fused 2.861 / 3.090. serial_triton `tape=h16` 0.421 / 0.757, `tape=u8` 0.482 / 0.675,
               `tape=h32` 0.500 / 0.874 (grad_cos 1.0000). All pass.
verdict:       The backward recurrence gets the factor (1 - theta sg_t) per step. It is still one linear reverse
               pass with the same traffic, so the times do not change.

## Extended: hard reset (the reset the checkpoint was trained with), carried over tokens
status:        ok
best config:   serial_fused 2.854 / 3.078. serial_triton `tape=h16` 0.416 / 0.764, `tape=u8` 0.454 / 0.675, exact.
               spec_verify (bit certificate) `C=512,W=128,bwd=serial` 0.507 / 0.853, exact.
verdict:       Same picture as the soft reset: 6.9x / 4.6x with the serial kernel. The interval certificate is not
               defined for the hard reset (the chunk map is not affine with one slope); the bit certificate works.

## Extended: the 4*L tick axis
status:        ok
best config:   every token fed 4 times, membrane carried over 8192 ticks, B = 16; gate passed.
               serial_fused 11.446 / 12.082. serial_triton `tape=u8` 1.508 / 2.537 (7.6x / 4.8x), `tape=h16`
               1.642 / 2.985. spec_verify C=512,W=128 1.839 / 3.579.
verdict:       Four times the steps, four times the time: the kernels read the repeated input four times. A kernel
               that knows about ticks could read x once per token and write one spike count per token; that is a
               different interface and is the next item.

## Combination: spec_verify + the 8-bit tape of serial_triton
status:        ok
best config:   in `C=1024,W=128,bwd=serial,tape=u8,U=32` 0.425 ms (6.7x); out
               `C=1024,W=256,bwd=serial,tape=u8,U=32,BLOCK=512,nw=8` 0.739 ms (4.2x). "Depth 64" at `in` (C + W; 128 with the sweep):
               `C=32,W=32,cert=interval,bwd=cold,Wb=32,U=32,BLOCK=64,nw=2` 0.497 ms (5.73x). All exact in the spikes.
verdict:       The two best ideas add up (0.492 -> 0.425 ms at `in`), but the result stays ~9 % behind
               serial_triton: the warm-up of every chunk is a second read of part of x. Certified chunks are the
               right tool when the sequential depth must be small; the serial kernel is the right tool here.

## serial_ticks
origin:        own idea, for the tick axis of the repo (T ticks per token)
idea:          With T ticks per token the input of a token is the same in all T ticks, and the model uses only the
               spike count per token. A kernel that knows this reads x once per token, runs the T ticks in
               registers, and writes one count per token and one int32 of tape per token (T 8-bit codes, T <= 4).
               The backward pass reads the gradient of the count and the tape word and writes one gradient per
               token. The membrane is carried over all L*T ticks, as on the tick axis of the extended study.
exactness:     exact (the same float32 operations per tick as the serial reference on the expanded axis)
seq_depth:     L * T
learnable tau: yes, per channel
status:        ok
attempts:      3. (1) U=8,BLOCK=256: exact at once, 1.085 ms at `in`. (2) Sweep of 10 block/unroll configs per site.
               (3) 5 more configs, bf16 tensors, hard reset. All rows pass with 0 wrong counts on all three inputs.
best config:   `U=32,BLOCK=32,nw=1`: in **0.520 ms**, out **0.920 ms** for 2048 tokens x 4 ticks (grad_cos 0.9999).
               Same job on the plain tick axis: serial_fused 11.446 / 12.082 ms, serial_triton 1.508 / 2.537 ms.
               So 22x / 13x over serial_fused and 2.9x / 2.8x over serial_triton. Hard reset: 0.623 / 0.924 ms.
               With bf16 tensors: 0.801 / 0.940 ms (no gain; the backward kernel is compute-bound there).
verdict:       The right kernel if the model keeps T ticks per token and carries the membrane: it moves 3 token-sized
               tensors per direction instead of 3 tick-sized ones. It is partly compute-bound (128 unrolled steps
               per loop iteration); one element per thread (BLOCK = 32 x warps) is clearly best, two elements per
               thread can be 2-3x slower in the backward pass. Its interface differs from the study's (input and
               output per token), so it is listed as an extension and not ranked.

## D: the linear scan
origin:        program.md D ("make sure the linear scan is implemented the fastest way before judging any method
               that uses it"); Parhi & Messerschmitt block IIR; Maleki & Burtscher PLR; Mamba-2 chunked matmul
idea:          Seven ways to compute k_t = beta k_{t-1} + u_t over [2048, 16, D], forward only (a probe, no result
               rows; numbers in the TSV comments).
status:        ok
best config:   in / out ms. Fused serial kernel with loads ahead: **0.150 / 0.289** (a plain copy of the tensor:
               0.135 / 0.267). Chunked C x C lower-triangular matmul, one shared beta, plus carry: C=128
               0.550 / 1.012; C=64 0.600 / 1.086. The same with one matrix per channel: 1.086 / 2.152.
               Weighted torch.cumsum in chunks of 64 plus carry: 0.853 / 1.722. FFT, zero-padded to 2560:
               1.830 / 3.891. Doubling scan, 11 rounds: 4.773 / 9.642. PyTorch loop: 13.65 / 13.87.
               One `torch.cumsum` over the time axis alone: 0.730 / 0.759.
verdict:       The fused serial pass is the fastest linear scan here by 3.5x; it costs one copy. A matmul or FFT
               does more work per element than a memory-bound pass needs. PyTorch's own cumsum along the time
               axis is latency-bound like SpikingJelly's kernel (same time at both widths). All scan-based
               methods of this study (`fpt`, `pmbc`, `newton_soft`, `psn`, `decoupled_threshold`, the candidate
               methods) were run with the fused pass, so none of them was judged with a slow scan.

## F: how small can the tape be
origin:        own question, to settle M2 at `out`
idea:          Simulate small-float codes of h - theta in PyTorch on the reference membrane, feed the decoded tape
               to the real backward kernel, and measure the gradient cosines (probe, numbers in the TSV comments).
status:        ok
best config:   min cosine over x and tau, real and slow-leak input, in / out: 8 bits 0.9997 / 0.9998 (the `u8`
               tape); 7 bits 0.9987 / 0.9991; 6 bits 0.9949 / 0.9964; **5 bits (sign + 3 exponent bits from 2^-5 +
               1 mantissa bit) 0.9948 / 0.9964, passes**; 4 bits 0.98 / 0.99, fails; 3 bits 0.96, fails.
               Tape-free floor, kernel-only: 0.304 / 0.555 ms (forward that stores only spikes, plus one read+write
               pass for the gradient).
verdict:       5 bits per step is the smallest tape that keeps grad_cos >= 0.99. At `out` every tape bit costs about
               0.009 ms, so a 5-bit tape would give ~0.65 ms (4.75x) against 0.677 ms now. 5x at `out` needs
               0.618 ms and is not reachable. The 5-bit tape was not built: 4 % is inside the run-to-run spread, and
               it costs code and gradient accuracy.
               Measured again later with the real backward kernel and no tape traffic: floor 0.329 / 0.585 ms,
               see "where the time of one pass goes". The conclusion is the same and stronger.
               Related work: ActNN (ICML 2021) and Chakrabarti & Moseley (NeurIPS 2019) save 2-4 bit
               activations for the backward pass of conv nets; there the gradient is linear in the saved
               value. Here it goes through the surrogate, which needs the distance to theta.

## Extended: the repo's training shape (micro-batch 2)
status:        ok
best config:   L = 2048, B = 2, gate passed in every row. serial_fused 1.413 / 2.119 ms. serial_triton: in
               `U=64,BLOCK=32,nw=1,tape=u8` 0.208 ms, out `U=32,BLOCK=64,nw=2,tape=u8` 0.256 ms (6.8x / 8.3x).
               spec_verify `C=64,W=64,cert=interval,bwd=cold,Wb=64,U=32,BLOCK=32,nw=1` 0.229 / 0.235 ms.
               With 4 ticks per token (8192 ticks): serial_fused 9.913 / 10.873; serial_triton on the tick axis
               0.988 / 1.066; spec_verify on the tick axis 0.330 / 0.596; serial_ticks 0.326 / 0.427 (30x / 25x).
verdict:       `program.md` fixes B = 16, but the configs of the repo train with a micro-batch of 2. At that size
               one neuron site has 1.5k-3k neurons. All own kernels are then at 0.2-0.3 ms, close to the fixed cost
               of two kernel launches plus autograd; SpikingJelly's kernel still needs 1.4-2.1 ms. With one program
               per 32 neurons (BLOCK=32, nw=1) the small layer still gets enough programs.

## Extended: batch-first tensors (the layout of the model)
origin:        own extension, for use in the repo: the model's activations are [B, L, D], the study's are [L, B, D]
idea:          `bf=1` makes `serial_triton`, `spec_verify` and `serial_ticks` read x and write the spikes (and the
               gradients) as [B, L, D] directly. Only the address computation changes: element (t, b, d) is at
               `b*L*D + t*D + d`. A warp still reads 32 neighbouring channels of one time step.
status:        ok
attempts:      3. (1) A second address base in every kernel: correct, but an A/B test against the version before
               showed the TIME-FIRST backward kernels 13-25 % slower where a thread holds one element (register
               pressure). (2) The layout became a compile-time switch, so the time-first kernel is the old code
               again (A/B: no difference). (3) Float tapes and the tick tape got the layout of x, so the
               batch-first kernel needs one address base too (only the packed 8-bit tape still needs two).
best config:   gate passed in every row, 0 wrong spikes. serial_triton `tape=u8`: in `U=32,BLOCK=128,nw=2` 0.430 ms,
               out 0.679 ms; `tape=h16`: 0.430 / 0.757 ms. serial_ticks `U=32,BLOCK=32,nw=1`: 0.531 / 0.945 ms.
               spec_verify `C=32,W=32,cert=interval` config: 0.506 / 0.981 ms. At B = 2: serial_triton 0.182 / 0.225 ms,
               serial_ticks 0.350 / 0.445 ms.
verdict:       Batch-first costs at most 5 % (the 8-bit tape at `in`) and nothing elsewhere. A transpose before and
               after the neuron would cost one copy each way in the forward pass and again in the backward pass
               (0.135 / 0.267 ms per copy), more than the neuron itself. Use `bf=1` in the model.
               Lesson for kernel work here: these unrolled kernels are close to the register limit; one more
               64-bit value per thread in the backward pass costs 13-25 %. Always A/B against the last backup.

## Extended: where the time of one pass goes (cache hints, launch cost, floor without tape traffic)
origin:        own follow-up to "how small can the tape be" and to the micro-batch 2 result
idea:          Three probes on `serial_triton U=32, tape=u8`, no change of the kernels.
               (1) Triton cache hints (`cache_modifier`, `eviction_policy`) on every large load and store.
               (2) Time stamps around the two kernel launches inside one forward+backward pass, and the slope of
               the kernel time over L, to split a pass into Python work, launch cost and memory traffic.
               (3) The same kernels with every tape store and load redirected to one cached row: the arithmetic of
               the 8-bit tape stays, its memory traffic goes. This is the floor for ANY smaller tape.
status:        ok (probes only, comment lines in the TSV; nothing in `neuron.py` changed)
best config:   (1) Loads `.ca/.cg/.cs/.cv`, stores `.wb/.cg/.cs/.wt`, `evict_first` on stores: 0.371-0.376 / 0.652-0.661 ms
               against 0.374 / 0.654 (kernels alone). No change. `evict_first` on loads: backward at `in` 36 % slower.
               (2) Kernels alone run at 1.25-1.44 TB/s; a plain `copy_` reaches 1.4-1.5 TB/s. One pass needs
               0.23-0.24 ms of Python (two launch calls, allocations, leak conversion, autograd). 71 us of it come
               before the forward kernel is launched; the GPU waits in that time. The rest hides behind the running
               kernels at B = 16. Whole pass at `in`: 71 us + kernels 0.316 ms + tail 0.014 ms = 0.401 ms.
               (3) No tape traffic: 0.329 / 0.585 ms kernels alone (with the tape: 0.372 / 0.654). The 8-bit tape
               costs 0.043 / 0.069 ms, at `out` 0.0086 ms per bit and step.
verdict:       The kernels already move bytes as fast as a plain copy. What is left is the number of bytes, which
               the float32 interface fixes, and about 85 us of Python around the two launches.
               M2 at `out` is closed by measurement: a tape with ZERO bits would give 0.605-0.615 ms (5.0-5.1x);
               5x needs 0.618 ms, so the tape may have at most 1 bit per step; the smallest tape that passes
               grad_cos has 5 bits (0.651 ms, 4.74x). The earlier floor 0.555 ms in "how small can the tape be"
               used a plain copy for the backward pass and was 0.03 ms too low.
               At the repo's micro-batch (B = 2) the pass is CPU-bound: whole pass 0.242 / 0.287 ms, Python part
               0.229 / 0.235 ms. More kernel work cannot help there. Levers that were NOT built: the leak conversion
               inside the kernel (about 25 us; q could differ from PyTorch's by one rounding step, so the spikes
               would no longer be bit-identical by construction); calling the compiled kernel directly (about 5 us
               per launch; needs own checks of pointer alignment, depends on Triton internals); folding the five
               small operations after the backward kernel into it.

## truncated_carry_backward (online training: OTTT, SLTT)
origin:        Xiao et al., NeurIPS 2022 (OTTT); Meng et al., ICCV 2023 (SLTT); fifth literature round
idea:          Keep the exact forward pass. In the backward pass keep the carry over K future steps only.
               K = 0 is no temporal path at all: one elementwise pass.
exactness:     spikes exact; gradient approximate, cosine sqrt(1 - beta^(2K+2)) for a white incoming gradient
seq_depth:     forward L, backward K
learnable tau: yes
status:        skip (fails the gate for the K that would matter, and gives no time gain)
attempts:      1 probe in PyTorch on the float32 tape, no kernel, no result rows
best config:   cosine to the exact gradient, x / tau. K=0: in real 0.864 / 0.886, out real 0.605 / 0.692, slow
               leak 0.313 / 0.69. K=2: in real 0.992 / 0.994. K=8: out real 0.992 / 0.995. K=32: slow leak
               0.983 / 0.995.
verdict:       The x cosine matches sqrt(1 - beta^(2K+2)) to 3 digits. grad_cos >= 0.99 needs
               K >= 1.96/|ln beta| - 1: 2 steps at beta 0.5, 8 at 0.8, 38 at 0.95. So the gate's slow-leak input needs
               K >= 38. A backward pass with a short carry reads and writes the same bytes as the exact one,
               and the exact one is already at the memory limit. Truncation buys depth (`spec_verify bwd=cold`),
               not time.

## reversible_backward (rebuild the membrane backwards instead of a tape)
origin:        reversible RNNs (MacKay et al., NeurIPS 2018); fifth literature round
idea:          h_t = v_t + theta*s_t and v_{t-1} = (h_t - q*x_t)/(1-q): the backward pass rebuilds h from the last
               membrane and needs no tape.
exactness:     unstable in floating point
seq_depth:     L
learnable tau: yes
status:        skip (probe)
attempts:      1 probe in PyTorch on the real input, no result rows
best config:   error of the rebuilt h above 1e-3 theta after 12 (in) / 22 (out) steps back, above theta after
               22 / 52 steps. A float64 reverse pass: 22 / 56 steps.
verdict:       Every reverse step divides by beta, so a rounding error doubles per step at `in`. The paper's fix
               is to store the bits that the forward step lost. The reverse step also reads x and the spikes
               again, 8 bytes per step; the 8-bit tape costs 1 byte per step. Not usable here.

## Independent code review (2026-11-22 04:55)
origin:        own step: a read-only reviewer read `neuron.py` and `bench_neuron.py` without the GPU
idea:          Look for defects that the gate and the stress test cannot see: inputs they never use.
status:        ok (all confirmed findings fixed; `neuron_backup_15.py`, `bench_neuron_backup_1.py` hold the old files)
attempts:      1. Every finding was first reproduced on the GPU, then fixed, then the stress test got a case for it.
best config:   No defect in the index arithmetic of the generated kernels (bounds, masks, warm-up clamps, 8-bit
               word addresses, batch-first addresses, 64-bit offsets).
               Fixed: (1) x and the incoming gradient were read as contiguous memory: `spikes.sum().backward()`
               gave gradient cosine 0.08, a transposed consumer 0.28, a non-contiguous x 678 wrong spikes of
               7680, with no error. (2) A tau with fewer than D values (the checkpoint has a scalar) was read out
               of bounds: 1816 wrong spikes of 7680. Now an error. (3) The tape-gradient backward ignored steps
               when L was not a multiple of 32. Now an error. (4) Unknown option strings ran another branch.
               Now an error. (5) `mem_gb` of the `serial_fused` rows was 0.3-0.6 GB too high (a cached module
               kept the last membrane sequence). (6) `seq_depth` of `spec_verify` left out the verification
               sweep; M3 is withdrawn. (7) Interval certificate: guessed entry state for kept chunks, tracked
               rounding margin. (8) The backward methods are marked `once_differentiable`. These kernels give
               first-order gradients only: second-order terms through the saved tape do not exist, and a second
               differentiation raises only when the incoming gradient itself needs a gradient.
               Checked and found fair: the reference wrapper adds 87 us (3 %) to SpikingJelly's time; the own
               kernel pays its own leak conversion; with each side owning its leak parameter the speed-up is the
               same (7.0x / 4.5x).
               Open, documented: the 8-bit tape holds |h - theta| < 128 (absolute); larger values are clamped (see
               "8-bit tape range" below). Gradients are compared by cosine in the gate; the stress test now also
               compares norms and a gradient through the last membrane.
verdict:       None of the logged rows is wrong: the harness always passed contiguous [L, B, D] tensors and a
               [D] tau. But the kernels were unsafe outside the harness, in ways a model would hit (a loss on
               `spikes.sum()`, a scalar tau).
               After the fixes (2026-11-22 05:12): every method re-run at both sites (rows "after review fixes"), same
               verdicts, times within 4 % of before; serial_triton `tape=u8` 0.416 / 0.676 ms against
               serial_fused 2.837 / 3.084 ms (6.8x / 4.6x). A/B against `neuron_backup_15.py`: the fixes cost at
               most 1 %. Stress test, extended by the review cases: 2240 random cases (6 seeds), 0 failed.

## 8-bit tape range (review finding, 2026-11-22 04:57)
origin:        the code review: the 8-bit tape clamps silently
idea:          The tape code of `tape=u8` (and of `serial_ticks`) holds 2^-9 <= |h - theta| < 128 in ABSOLUTE units
               (16 binades, 3 mantissa bits). Smaller values decode to about 2^-9, larger ones to about 124.
status:        ok (no change of the code; range documented, effect measured)
best config:   Real activations of all 12 blocks at `out` (probe): the membrane reaches 205, 572, 391, 947 and
               172 theta in blocks 1, 3, 4, 5 and 10, so the clamp is active there. Gradient cosine to the reference
               in those blocks: 0.9996-0.9998 for x and for tau, the same as in blocks without clamping. The
               smallest value of the probe (0.9973, tau, block 0) is a block with a firing rate of 0.13 % and
               no clamping.
verdict:       A clamped step has a surrogate of about 6e-6, so its gradient weight is close to zero; only the
               leak term of such a step is wrong, and it is multiplied by that weight. No measurable effect.
               Two limits to know: (1) the code is in absolute units, built for theta near 1 (the repo uses 1.0);
               with a much smaller threshold scale the leak gradient loses resolution; (2) for a gradient that
               must be exact, use `tape=h16` or `tape=h32`. A wider range (other exponent offset) was not
               tried: the present code is the validated one, and nothing was gained on the real data.
