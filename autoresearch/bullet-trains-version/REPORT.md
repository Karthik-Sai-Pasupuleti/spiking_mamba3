# REPORT: Bullet Trains against `serial_triton` for the hard-reset LIF (RTX 5090)

Study run on 2026-11-23, 10:34–10:43 CET. 260 result rows in `bench_results.tsv`.

## 1. Question and answer

**Question.** Bullet Trains (Morrill, Pehle, Zador, ICML 2026, arXiv:2603.13283) runs a hard-reset
LIF with associative scans over chunks of input. Is this faster than `serial_triton` (U = 32, 8-bit tape)
for the neuron of this repo, in training and in inference?

**Answer: no.** On the real activations, the best Bullet Trains config is 2.2× slower than `serial_triton`
at site `in` and 2.3× slower at site `out` for training (forward + backward). For inference (forward only),
it is 3.6× and 3.5× slower. It is still faster than SpikingJelly at both sites. It only ties `serial_triton`
when the neurons almost never fire (0.01 % of the steps).

| site, L = 2048, B = 16, θ = 1 | forward + backward (ms) | forward, no grad (ms) | spike_agree | grad_cos |
|---|---|---|---|---|
| `in`: SpikingJelly fused kernel | 2.847 | 0.864 | 1.0000 | 1.0000 |
| `in`: `serial_triton` U=32, 8-bit tape | **0.415** | **0.195** | 1.0000 | 0.9996 |
| `in`: Bullet Trains, best (K = 2) | 0.923 | 0.705 | 0.9999 | 0.9996 |
| `out`: SpikingJelly fused kernel | 3.101 | 0.956 | 1.0000 | 1.0000 |
| `out`: `serial_triton` U=32, 8-bit tape | **0.672** | **0.349** | 1.0000 | 0.9996 |
| `out`: Bullet Trains, best (K = 2) | 1.573 | 1.212 | 1.0000 | 0.9996 |

All rows use the neuron of the trained checkpoint: a hard reset to 0 with a detached reset, carried
over all 2048 tokens. Times are medians of 10 passes of one neuron site. The gate is the gate of the
soft-reset study: `spike_agree ≥ 0.999`, `grad_cos ≥ 0.99`, `parity_ok = 1`.

## 2. What was built

**The paper's method.** The paper has a continuous-time LIF with a synaptic current and sparse input
spike events. One round takes the next K input events and computes all states with an associative scan
over affine maps. It finds the first output spike, resets the membrane to 0 and discards the work after
the spike. The next round starts right after the spike. The paper uses K = 128.

**The discrete version (`bullet_train` in `bullet_neuron.py`).** The same rounds, with one input per
time step instead of one per event:

1. Each neuron keeps its own position in time.
2. One round loads the next K inputs and scans the maps `v → (1 − q) v + q x_t` (`tl.associative_scan`).
3. The first step with `h ≥ θ` is the spike. The round commits the steps up to it, sets `v = 0`, and moves
   the position to the step after the spike. Without a spike, the round commits all K steps.
4. The kernel writes the spikes and the same 8-bit tape as `serial_triton`.

32 neurons share one loop (one program). A program runs until its busiest neuron reaches L.

**Two backward passes**, as you asked:

- `bwd=serial`: the serial backward kernel of `serial_triton`. Only the forward pass differs from the baseline.
- `bwd=scan`: a chunked reverse associative scan. The backward pass of the detached hard reset is a linear
  recurrence (`gh_t = gs_t·sg_t + (1 − s_t)(1 − q)·gh_{t+1}`). Chunks of Kb steps run one after the
  other, with a scan inside each chunk.

**Differences from the paper:**

- **No spike cap.** The paper caps each neuron at 42 output spikes and leaves the rest of the input
  unprocessed. Here every step is processed, so the result is exact. With a cap of 42, most inputs would be
  dropped here: the busiest neurons fire 720 times (`in`) and 1902 times (`out`).
- **Surrogate gradient, not spike-time gradient.** Exact spike-time gradients (implicit function theorem)
  need continuous spike times. A discrete neuron has none.
- **The neuron is discrete and has one state value.** The paper's neuron is continuous-time with two
  state values (membrane and synaptic current). The scan idea is the same for both.

**Correctness.**

- `check_bullet.py` ran 720 random cases (6 seeds) against a plain PyTorch loop. The cases cover a non-zero
  start membrane, the no-leak mode, a gradient through the last membrane, L not a multiple of K, all
  tapes, both backward passes and a non-contiguous incoming gradient. 0 cases failed.
- With the float32 tape, the gradients match the loop (cosine ≥ 0.999). With the 8-bit and the float16
  tape, they match `serial_triton` with the same tape. The tape itself changes the gradient, and that is
  the same for both methods (see section 6).
- On the gate inputs, the scan gives 1 wrong spike of 25 M on the real input at `in`, and 3–4 on the slow-leak
  input. These are threshold ties: the scan adds the same numbers in another order, so h can round to the
  other side of θ. The hard reset stops the error at the next spike. Parity holds on the counting input.

## 3. Results on the real activations

### Chunk size K (forward grid, serial backward)

Best block shape for each K. "Rounds per program" is the number of rounds of the busiest neuron of the
32. "Work retained" is the paper's metric: steps committed / steps scanned.

| K | `in` fwd+bwd (ms) | `in` rounds per program (mean) | `in` work retained | `out` fwd+bwd (ms) | `out` rounds per program (mean) | `out` work retained |
|---|---|---|---|---|---|---|
| 2 | **0.923** | 1153 | 94 % | **1.573** | 1163 | 97 % |
| 4 | 1.098 | 722 | 84 % | 2.411 | 747 | 92 % |
| 8 | 1.293 | 530 | 68 % | 5.084 | 564 | 81 % |
| 16 | 2.330 | 458 | 47 % | 8.767 | 489 | 65 % |
| 32 | 3.813 | 441 | 28 % | 12.186 | 459 | 45 % |
| 64 | 5.433 | 439 | 15 % | 13.131 | 448 | 28 % |
| 128 (the paper's K) | 8.995 | 439 | 7 % | 16.622 | 444 | 15 % |
| `serial_triton` | 0.415 | 2048 steps | 100 % | 0.672 | 2048 steps | 100 % |

All 88 configs of the grid passed the gate (K = 2…128, block 32–128, 1–4 warps).

What the table shows:

- **Rounds stop falling at about 440.** A large K cannot remove rounds below the number of spikes. The
  busiest neuron of a program fires about 440 times on average, and up to 721 (`in`) and 1902 (`out`) times.
- **Large K wastes work.** At K = 128 the method keeps only 7 % (`in`) and 15 % (`out`) of what it scans.
  The paper reports 80–90 % on SHD and SSC. There the neurons fire rarely compared with their input events.
  Here a neuron fires about once every 9 steps at `in` (rate 10.8 %).
- **The best K is the smallest.** K = 2 is the closest thing to the serial loop, and it is the fastest.

### Backward pass: scan against serial

Same forward pass (K = 4, block 32, 1 warp). Backward time = forward + backward minus forward.

| backward pass | `in` (ms) | `out` (ms) |
|---|---|---|
| serial kernel of `serial_triton` | 0.21 | 0.37 |
| chunked reverse scan, best Kb (64 at `in`, 16 at `out`) | 0.22 | 0.39 |
| `serial_triton`'s own backward, for scale | 0.22 | 0.32 |

36 scan configs per site (Kb 16–128, block 32–128, 1–4 warps) all passed the gate and gave the same
gradient cosine (0.9996). **The two backward passes cost the same.** Both read the incoming gradient and
the tape once and write the outgoing gradient once, so both are limited by memory traffic. The scan has a
smaller depth, but depth was not the limit. So the whole gap to `serial_triton` comes from the forward pass.

### Micro-batch 2 (the repo's training shape)

| L = 2048, B = 2 | `in` fwd+bwd (ms) | `in` no grad (ms) | `out` fwd+bwd (ms) | `out` no grad (ms) |
|---|---|---|---|---|
| SpikingJelly | 1.415 | 0.416 | 2.129 | 0.412 |
| `serial_triton` | **0.225** | **0.103** | **0.284** | **0.101** |
| Bullet Trains K = 2, serial backward | 0.424 | 0.306 | 0.674 | 0.416 |
| Bullet Trains K = 4, scan backward | 0.479 | 0.318 | 0.754 | 0.533 |

All rows passed the gate. At this size Bullet Trains is 1.9–2.4× slower than `serial_triton` for training
and 3.0–4.1× slower for inference.

## 4. Sparsity sweep: where would Bullet Trains pay?

The paper's method needs sparse firing. A higher threshold θ on the same real input gives fewer spikes.
θ changes the real and the slow-leak inputs; the counting input stays as it is.

Site `in` (forward + backward, ms):

| θ | firing rate | `serial_triton` | Bullet K=2 | K=4 | K=16 | K=32 | K=128 |
|---|---|---|---|---|---|---|---|
| 1 | 10.8 % | **0.415** | 0.923 | 1.098 | 2.495 | 4.025 | 9.124 |
| 1.5 | 4.4 % | **0.418** | 0.838 | 0.951 | 1.466 | 2.830 | 6.825 |
| 2 | 1.5 % | **0.414** | 0.781 | 0.806 | 1.187 | 1.404 | 4.568 |
| 3 | 0.11 % | **0.418** | 0.720 | 0.575 | 0.618 | 0.694 | 1.184 |
| 4 | 0.005 % | **0.390** | 0.696 | 0.541 | 0.437 | 0.435 | 0.610 |

Site `out` (forward + backward, ms):

| θ | firing rate | busiest neuron | `serial_triton` | Bullet K=2 | K=4 | K=16 | K=128 |
|---|---|---|---|---|---|---|---|
| 1 | 4.9 % | 1902 spikes | **0.672** | 1.573 | 2.411 | 9.088 | 16.823 |
| 2 | 1.8 % | 1804 | **0.701** | 1.271 | 1.757 | 4.386 | 12.923 |
| 4 | 0.46 % | 1574 | **0.674** | 1.097 | 1.275 | 2.393 | 8.918 |
| 8 | 0.08 % | 1042 | **0.672** | 0.951 | 0.975 | 1.578 | 5.621 |
| 16 | 0.01 % | 510 | **0.704** | 0.916 | 0.798 | 1.027 | 2.961 |
| 32 | 0.001 % | 148 | **0.673** | 0.922 | 0.754 | 0.765 | 1.349 |

Rows at `out` with θ = 16 and 32 fail the gradient gate for both methods equally (cosine 0.98 and 0.92).
That is the 8-bit tape, which is built for θ near 1. It does not change the times.

What the sweep shows:

- **`serial_triton` keeps its time at every θ** (0.39–0.42 ms and 0.67–0.70 ms). Its cost is memory
  traffic, and the traffic does not depend on the spikes.
- **Bullet Trains gets faster as firing drops, but it never wins.** At `in` with θ = 4 (1 spike per
  20 000 steps), K = 32 ties at 0.435 ms against 0.390 ms.
- **At `out`, a few neurons keep firing.** Even at a mean rate of 0.001 %, the busiest neuron fires
  148 times, and its program of 32 neurons waits for it.

### Few neurons and long sequences

The soft-reset study found that parallel-in-time methods pay off with few neurons and long sequences: the
GPU is then mostly idle. Probe: B = 1, D = 768 (site `in`), L = 32768, timing only.

| θ | `serial_triton` | Bullet K=4 | K=16 | K=64 | K=128, scan backward |
|---|---|---|---|---|---|
| 1 (10.8 %) | **3.271** | 8.735 | 14.098 | 33.434 | 61.794 |
| 4 (0.005 %) | **3.316** | 5.859 | 3.774 | 3.552 | 3.769 |

Bullet Trains does not win here either. Its rounds split the time axis of each neuron into steps of K,
but one neuron still runs its rounds one after the other, on one program. With 768 neurons the GPU stays
idle in both methods. A certified chunk method that splits the time axis over many programs
(`spec_verify` in the soft-reset study, 6× at L = 131072) uses the idle GPU. Bullet Trains does not.

## 5. Why the method does not pay off here

1. **The bottleneck is different.** Bullet Trains removes sequential depth: a scan of K steps has depth
   log₂ K instead of K. `serial_triton` already hides the memory latency by loading 32 steps ahead, so depth
   was not its limit. Its limit is the bytes it moves, and it moves each byte once.
2. **A spike ends the round.** After each spike the rest of the scanned chunk is discarded. With dense
   input (one value every step) and firing rates of 1–11 %, most scanned steps are discarded at K = 128.
   Each discarded step is one more load of x.
3. **The busiest neuron sets the time.** Neurons of one program share the loop. On the real input a few
   neurons fire in long bursts (up to 1902 of 2048 steps at `out`). This is the "bursting" case that the
   paper names as its worst case (Appendix G.1), where the method "reverts to serial execution".
4. **The paper's 44× is against a naive serial loop.** The paper compares with a serial event loop in JAX
   on an H100, which processes one event per step. The baseline here is a tuned serial kernel.
   SpikingJelly plays the role of the naive baseline, and Bullet Trains does beat it here: 3.1× at `in` and
   2.0× at `out`, measured as SpikingJelly time / Bullet Trains time with the best K.

## 6. Limits of this study

- One GPU (RTX 5090), one block of the model (block 6), validation windows 0–15.
- The paper's own setting was not reproduced: continuous time, sparse event input (SHD, SSC),
  spike-time gradients, JAX. The conclusion holds for the discrete, dense, surrogate-gradient setting of
  this repo only.
- The forward kernel is my own Triton translation of the method. Other layouts are possible, for example
  one warp per neuron with the scan across the lanes. In such a layout the busiest-neuron effect is
  smaller, but every round costs a warp per neuron. I did not try it.
- The 8-bit and float16 tapes change the gradient by themselves. The leak gradient of the 8-bit tape gets
  worse for θ far from 1, as documented in the soft-reset study. This affects both methods equally.
- Spikes are exact up to threshold ties (1–4 per 25–50 M neuron-steps). `serial_triton` is bit-exact.

## 7. Files and how to reproduce

| file | what it is |
|---|---|
| `bullet_neuron.py` | the forward kernel, both backward kernels, the `bullet_train` method; `python -B bullet_neuron.py` is a smoke check |
| `bench_bullet.py` | entry point: the soft-reset study's harness, plus `--theta` for the sweep |
| `check_bullet.py` | random edge cases against a plain PyTorch loop: `python -B check_bullet.py 120 0` |
| `bench_results.tsv` | all 260 result rows, with comment lines for each part of the study |
| `bench_*.log`, `sweep.log`, `mb.log`, `ln.log` | raw logs of the runs |

The study imports `neuron.py` and `bench_neuron.py` from `../soft_reset_LIF_experiment/` and reads its
`bench_inputs.pt`. It changes nothing in that folder (all runs use `python -B`, so no bytecode is written).

```
python -B bench_bullet.py --method bullet_train --config "K=2,BLOCK=32,nw=1,tape=u8,bwd=serial,U=32,sBLOCK=64,snw=2" --site in --reset hard
python -B bench_bullet.py --method bullet_train --config "K=4,BLOCK=32,nw=1,tape=u8,bwd=scan,Kb=64,bBLOCK=32,bnw=2" --site in --reset hard
python -B bench_bullet.py --method serial_triton --config "U=32,BLOCK=64,nw=2,tape=u8" --site in --reset hard
python -B bench_bullet.py --theta 4 --method bullet_train --config "K=32,BLOCK=32,nw=4,tape=u8,bwd=serial,U=32,sBLOCK=64,snw=2" --site in --reset hard
```

`python` = `/home/karthik/Desktop/spiking_mamba3/.venv/bin/python`. No linter was run: `ruff`, `mypy`,
`pylint` and `codespell` are not installed in the venv.
