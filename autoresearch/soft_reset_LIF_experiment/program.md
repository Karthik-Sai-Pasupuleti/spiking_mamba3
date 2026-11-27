# autoresearch — the fastest way to compute soft-reset LIF spikes across L tokens (RTX 5090)

This is an experiment to have the LLM do its own research. One GPU (RTX 5090, 32 GB), one
question, no time limit:

> **What is the fastest way to compute the spikes of a soft-reset LIF with a learnable leak over
> all L = 2048 tokens of a sequence — membrane carried from token to token, not reset per token —
> that still matches the exact serial computation?**

Nothing else: no language-model training, no architecture work, no perplexity. The output of
this study is a ranked table of methods, the fidelity numbers behind each, and a recommended
method with its measured speed-up over the exact serial kernel.

**There is no human in the loop.** Nobody will answer a question, confirm a plan, approve a
step, or fix the environment. Read this file, set up, start, and keep going until the process is
killed. Every decision you would have asked about is made by the defaults in this file; where the
file is silent, take the simplest option, log it as a `#` comment in `bench_results.tsv`, and
continue.

## Where things live

### This folder — everything you create (strictly)

Every file you create or write lives in the folder that contains this `program.md`. No
subfolders, no git, no cloning, no downloads, no `pip install`, nothing written anywhere else.

- `program.md` — this file, read-only.
- `bench_lif_over_tokens.py` — seed script for the reference method, if present.
- `neuron.py` — **you create it.** Every method lives here behind one interface.
- `bench_neuron.py` — **you create it.** Builds inputs, runs one (method, config, site), prints
  `RESULT` lines.
- `bench_inputs.pt` — the fixed test inputs (see "Fidelity gate"), generated once.
- `bench.log` — output of the most recent run; overwritten each run; never streamed into your
  context.
- `bench_results.tsv` — one row per (method, config, site). The main artefact.
- `METHODS.md` — **the method ledger**: one entry per method, written *before* you implement it
  and updated *after* (see "Logging"). Every method you even consider gets an entry.
- `LITERATURE.md` — the literature ledger (see "Literature review").
- `REPORT.md` — the running report, rewritten at every milestone.
- `neuron_backup_<n>.py` — a copy of `neuron.py` taken before any edit that could break a method
  already logged `ok`. This replaces git.

### Read-only inputs — outside this folder, never written to

- Repo: `/home/karthik/Desktop/spiking_mamba3/` — the training code (model definition, the
  spiking-neuron class and its exact settings, data loader). **Read it; import from it; change
  nothing in it.** Add it to `sys.path` from `bench_neuron.py` when you need the model class.
- Checkpoint: `/home/karthik/Desktop/spiking_mamba3/runs/mamba3-siso-187m-wikitext103-spiking/best.pt`
  — the best trained spiking Mamba-3 (187M, SISO). Used once, to capture real input activations
  at both neuron sites and the learned per-channel neuron parameters.
- Data: `/home/karthik/Desktop/spiking_mamba3/data/wikitext103/` — WikiText-103, tokenised with
  the **Llama-3.1 tokenizer**. Use the validation split for activation capture. If the folder
  holds pre-tokenised shards, use them; if it holds only text and the Llama-3.1 tokenizer is not
  available locally, do not download it — fall back as described under "Fidelity gate".

## Setup (do it yourself, then start — no confirmation step)

1. **List this folder** and the three read-only locations.
2. **Verify the GPU stack** (print, do not fix):
   `python -c "import torch, spikingjelly; print(torch.__version__, torch.cuda.get_device_name())"`
   and `python -c "import cupy, triton; print(cupy.__version__, triton.__version__)"`.
   The 5090 is Blackwell (sm_120). If `cupy` is missing or refuses the device, the reference uses
   SpikingJelly's `backend='torch'` (slow but exact). If `spikingjelly` is missing, implement the
   reference as a plain PyTorch loop (ten lines). If `triton` is missing, the Triton methods are
   `skip`. Never install, never wait. Log every fallback.
3. **Read the neuron used in training.** Open the repo's model file and copy the *exact* neuron
   update used at the `in_proj` and `out_proj` inputs (class, `decay_input`, `v_reset`, surrogate,
   `detach_reset`, how `tau` is parameterised, whether `theta` is learnable, how the T-axis is laid
   out). Write those equations into `METHODS.md` under "Reference neuron". The reference method
   must reproduce *that* neuron over the token axis, not a textbook LIF.
4. **Read the two site widths from the checkpoint**: `D_in` = the input width of `in_proj`,
   `D_out` = the input width of `out_proj` (from the state-dict shapes). Use those everywhere this
   file says "both sites".
5. **Create `bench_inputs.pt`** (see "Fidelity gate") with a fixed seed.
6. **Initialise `bench_results.tsv`** with the header row, then `#` comment lines with the start
   time, GPU, library versions, `D_in`/`D_out`, and every fallback taken.
7. **Create `METHODS.md` and `LITERATURE.md`** with their headers and the "Reference neuron"
   entry.
8. **Start the loop immediately.** No plan summary, no question. First run: `serial_fused` at
   `D_in`.

Platform notes: membranes in fp32; no `torch.compile`; SpikingJelly's cupy backend compiles on
first call — warm up before timing; `torch.cuda.synchronize()` around every timing; one process
on the GPU at a time.

## The neuron being computed

Generic form (replace with the exact repo equations from Setup 3):

```
v_t = beta * v_{t-1} + x_t - theta * s_{t-1}        beta from a learnable per-channel tau
s_t = Heaviside(v_t - theta)                         surrogate gradient as in the repo
```

over the token axis, `x` and `spikes` of shape `[L, B, D]`, L = 2048, membrane carried from
token t−1 to t and zero at t = 0. Every method exposes one interface in `neuron.py`:

```
spikes, v_last = method(x, tau, theta, v0=None, **kwargs)
```

`v0` is an optional initial membrane `[B, D]` (the chunked methods depend on it). B = 16 unless a
method's grid says otherwise. Every config runs at both sites, `D_in` and `D_out`.

## Metric, direction, gate

**One metric: `ms_fwd_bwd`, the median wall-clock milliseconds of one forward+backward pass of
one neuron site at `[2048, 16, D]`, lower is better — ranked only among configs that pass the
fidelity gate.** A config that fails the gate is reported with its numbers, never ranked.

**Fidelity gate**, every (method, config, site) measured against `serial_fused` on identical
inputs from `bench_inputs.pt`:

1. **Real input**: activations captured at the `in_proj` and `out_proj` inputs from one forward
   pass of the checkpoint on 16 validation sequences of 2048 tokens, together with the
   checkpoint's learned per-channel `tau` (and `theta` if learnable). If the checkpoint or the
   data cannot be loaded within 30 minutes of trying, fall back to Gaussian input scaled so the
   reference fires at 5–10 % with `tau` = the repo's init value, log the fallback, continue.
2. **Counting input**: a random 0/1 stream with the leak removed (`beta = 1`) and `theta = 2`.
   The exact neuron holds the running parity of the stream in its membrane. Record whether the
   method's cumulative spike count has the reference's parity at every t ≤ 2048. This is the
   property the reset buys; a method that loses it is disqualified for state-tracking work.
3. **Slow-leak input**: Gaussian input with `beta = 0.95` (half-life ≈ 13 tokens) — where
   cross-token memory matters most and truncation hurts most.

Per (method, config, site) record:

- `ms_fwd_bwd` — median of 10 timed passes after 3 warm-ups.
- `ms_fwd` — forward only, same protocol.
- `mem_gb` — peak allocated memory during forward+backward, GB to .1f.
- `spike_agree` — fraction of positions whose spike equals the reference (input 1).
- `v_maxerr` — max |v − v_ref| over all positions (worst of inputs 1 and 3).
- `grad_cos` — cosine similarity to the reference gradient for the loss `(spikes * r).sum()`,
  fixed random `r`, computed w.r.t. `x` **and** w.r.t. `tau`; report the smaller. Gradient to
  `tau` zero or None ⇒ the method does not support a learnable leak ⇒ `grad_cos = 0`.
- `parity_ok` — 1 if input 2 keeps parity at every t, else 0.
- `seq_depth` — the number of sequential steps the method needs (e.g. 2048, C + W, K).
- `status` — `ok`, `crash`, `skip`.

**Pass** = `spike_agree ≥ 0.999` AND `grad_cos ≥ 0.99` AND `parity_ok = 1`.

**Milestones** (write them into `REPORT.md` when reached, then keep going):
- M1: a passing method ≥ 2× faster than `serial_fused` at both sites.
- M2: ≥ 5× at both sites.
- M3: ≥ 5× with `seq_depth ≤ 64`.
No milestone ends the study. The study ends when the process is killed.

## Literature review — the method menu (implement these; add what you find)

The problem — a linear recurrence with one threshold nonlinearity and a reset — has been attacked
in several fields under different names. Before and between implementations, keep a ledger:
`LITERATURE.md`, one entry per paper or idea: *name / where it comes from / the idea in two
lines / which method id it maps to / status (implemented, planned, skipped + why)*. If you have
web search, spend at most one hour in every six on searching, and never on downloading code.
If you do not, the menu below is your literature; work through all of it. Every item below gets
a `LITERATURE.md` entry and, if implementable, a `METHODS.md` entry and a `bench_results.tsv` row.

**A. Spiking-neuron parallelisation (the home field)**
- `serial_fused` — SpikingJelly multi-step neuron, cupy backend: one fused kernel per layer, loop over L inside. The exact reference.
- `serial_python` — same neuron, `backend='torch'`: a Python loop. Time once, for the record.
- `serial_triton` — your own fused Triton kernel (forward + backward) for the exact recurrence, block-size sweep. Does SpikingJelly's kernel leave speed on the table?
- `chunk_cold(C, W)` — chunks of C tokens as extra batch rows, each prefixed with W warm-up tokens, from v = 0; discard the warm-up. The receptive-field-truncation idea from chunked RNN/WaveNet inference. Error ≈ `beta^W`.
- `chunk_warm(C, passes)` — pass 1 under `no_grad` from v = 0 gives chunk end states; pass p reruns every chunk from its predecessor's pass-(p−1) end state via `v0`. Chunk-level Picard; error ≈ `beta^(passes·C)`.
- `fpt(K, relax)` — Fixed-point Parallel Training (Feng et al. 2025): Picard on the spike train — no-reset membrane by parallel scan, threshold, subtract those spikes' resets by another scan, repeat K times; `hard` or `sigmoid` relaxation. K ∈ {1, 2, 3, 5}.
- `newton_soft(K)` — FidLIF-style (2026): Newton on the whole membrane trajectory with sigmoid relaxation; the Jacobian is I − (lower-bidiagonal), so each Newton step is a scan; implicit-differentiation backward (one reverse scan), memory independent of K. K ∈ {2, 3, 5}.
- `decoupled_threshold` — PRF-style (2024): no-reset membrane by scan, reset carried as a dynamic threshold `r_t = beta * r_{t-1} + theta * s_{t-1}` in a fused 1-D loop. Exact; tests whether a scalar-only loop is cheaper.
- `cumsum_closed_form` — PMSN / ANN-to-SNN-conversion identity: for `beta = 1` and non-negative input, spike count up to t = `floor(cumsum(x)/theta)`, spikes = its difference — exact, one prefix sum. Implement it exactly for the leak-free case, then test the approximations that extend it to leak (`beta < 1`, pre-leaked input) and to signed input (ternary spikes with ±theta). Expected to fail the gate on real input; log how badly.
- `sdn_predictor` — SpikingSSMs (AAAI 2025): train a tiny causal 1-D CNN (< 200 parameters) to predict the reset term from the no-reset membrane trajectory, then one scan. Train it on `bench_inputs.pt` for a few minutes; log its accuracy; it is approximate by design.
- `pmbc(iters)` — SPikE-SSM (ICLR 2025): parallel max–min boundary compression — iterate upper and lower bounds on the membrane until spikes are determined; exact when converged. iters ∈ {2, 4, 8}.
- `speculative_chunk(C)` — Bullet Trains (ICML 2026) in discrete time: scan a chunk with no reset, find the first threshold crossing in parallel, apply the reset there, rescan from that point; rounds = number of spikes in the chunk. Exact. Expected slow at 5–10 % firing; measure it anyway.
- `map_scan(C)` — Hard Resets Compose (2026): the map `v_in → v_out` over a chunk of C tokens is piecewise affine (for soft reset: `beta^C · v_in + c − theta · n(v_in)` with the spike count `n` piecewise constant in `v_in`); such maps compose, so chunk maps combine by associative scan; prune to reachable `v_in` ranges. Exact, non-iterative, depth log(L/C). The hardest to implement; do it after the simpler ones are logged.
- `psn`, `sliding_psn`, `spsn`, `psu` — no reset / stochastic reset / probabilistic reset (Fang et al. 2023, Yarga & Wood 2023, Li et al. 2024). Parallel by construction. Expected to fail the gate (they change the neuron); log them as the floor of what "parallel" costs in fidelity.
- `mpe_psn` — membrane-potential-estimation parallel neuron (2024): estimate the membrane without reset, correct statistically. Log as above.

**B. Parallel-in-time numerical methods**
- `parareal(C, iters)` — coarse propagator = the no-reset linear scan (cheap, parallel), fine propagator = exact serial within chunks; iterate the coarse–fine correction on chunk boundary states. Compare with `chunk_warm` (which is parareal with a trivial coarse step).
- `multiple_shooting(C, iters)` — Newton on the chunk boundary states only (L/C unknowns per channel), fine solves inside chunks. Fewer unknowns than `newton_soft`.
- `waveform_relaxation` — Jacobi (all chunks at once) vs Gauss–Seidel (chunks in order, each using the freshest neighbour) iteration on the reset term; Gauss–Seidel converges in fewer sweeps but is sequential over chunks. Log both.
- `deer` / `quasi_deer` / `elk` — Lim et al. 2024 (Newton over the whole trajectory, Jacobian solve as a scan), Gonzalez et al. 2024 (quasi-Newton, O(T·D) memory; trust region via Kalman smoothing). `newton_soft` is DEER specialised to this neuron; implement quasi-DEER if `newton_soft` is memory-bound.

**C. Computer arithmetic — the neuron as a carry chain**
- `carry_select(C, K)` — carry-select adder idea: the carry here is the membrane entering a chunk, bounded in a known interval; precompute each chunk's output for K candidate carries spanning that interval (as extra batch rows), then select the nearest candidate once the true carry is known from the previous chunk. One sequential sweep over L/C chunks of a cheap selection. Error set by K; exact if the spike pattern is identical for all carries in the selected bin.
- `carry_lookahead` — the same with a two-level selection tree (chunks of chunks). Depth log(L/C).

**D. Signal processing — the neuron as a sigma–delta modulator**
- The leak-free soft-reset IF *is* a first-order sigma–delta modulator (1-bit quantisation with error feedback); the leaky one is a leaky sigma–delta. Look-ahead and block-parallel sigma–delta modulators exist (SACD/DSD encoders). `sigma_delta_lookahead(C)` — compute the quantiser outputs for a chunk by exhaustive or pruned look-ahead on the bounded error state; compare with `carry_select`. Log the correspondence in `LITERATURE.md` even if no implementation beats the chunk methods.
- Block/parallel IIR filtering (state-space partitioned IIR, block-recursive filters): the linear part of every method above is an IIR filter; make sure the linear scan is implemented the fastest way (chunked C×C lower-triangular matmul on Tensor Cores + carry) before judging any method that uses it.

**E. Hybrid / piecewise-smooth dynamical systems**
- Firing-map (Poincaré-map) composition: the map from one spike to the next is smooth and composable — the continuous-time ancestor of `map_scan` and `speculative_chunk`. Entry in `LITERATURE.md`; implementation is covered by those two.

**F. GPU engineering (applies to every method)**
- Occupancy: the serial fused kernel parallelises over B·D only; measure A0 vs the best chunked method at B·D ∈ {2k, 8k, 25k, 100k, 400k} — find where the serial kernel starts filling the GPU and parallel-in-time stops paying.
- Fuse the two neuron sites of one layer into one kernel launch; fuse pass 1 and pass 2 of `chunk_warm`; fp16/bf16 membranes (does the gate survive?); kernel block-size sweeps for every Triton method.

Order of work: A (`serial_*`, `chunk_*`, `fpt`, `decoupled_threshold`, `cumsum_closed_form`,
`newton_soft`, `pmbc`, `speculative_chunk`, `sdn_predictor`, the no-reset family, `map_scan`),
then C, then B, then D, then F on the top three passing methods, then anything new from the
literature ledger. When two methods are mathematically the same thing under different names, say
so in `METHODS.md` and run it once.

## Logging — strict, every method, every attempt

`bench_results.tsv` (tabs, not commas), header plus one row per (method, config, site, attempt):

```
method	config	site	D	ms_fwd	ms_fwd_bwd	mem_gb	spike_agree	v_maxerr	grad_cos	parity_ok	seq_depth	status	notes
```

Example:

```
method	config	site	D	ms_fwd	ms_fwd_bwd	mem_gb	spike_agree	v_maxerr	grad_cos	parity_ok	seq_depth	status	notes
serial_fused	-	in	1024	14.10	41.20	1.2	1.0000	0.0e+00	1.0000	1	2048	ok	reference
chunk_cold	C=256,W=128	in	1024	3.30	9.80	1.4	0.9998	3.1e-03	0.9950	0	384	ok	parity lost at chunk boundaries
chunk_warm	C=256,p=2	in	1024	4.20	12.40	1.4	1.0000	2.0e-06	0.9999	1	512	ok	
fpt	K=3,sigmoid	in	1024	5.10	15.10	3.9	0.9970	4.2e-02	0.9800	0	3	ok	drifts after ~1.5k tokens
newton_soft	K=3	in	1024	0.00	0.00	0.0	0.0000	0.0e+00	0.0000	0	3	crash	NaN in Newton step, attempt 1 of 3
```

Crashes: zeros in the numeric columns, `crash`, the reason and the attempt number in notes.
A retried config gets a *new* row — never overwrite a row.

`METHODS.md` — one entry per method, in this shape, written **before** the first line of code and
updated after every attempt:

```
## <method id>
origin:        <paper / field / "own idea">
idea:          <two lines>
exactness:     exact | approximate (error ~ ...) | changes the neuron
seq_depth:     <formula>
learnable tau: yes | no | via <...>
status:        planned | implemented | ok | crash | skip (<why>)
attempts:      <n>, <what went wrong each time>
best config:   <config, site, ms_fwd_bwd, gate columns>
verdict:       <one line>
```

Every method you consider — including ones you decide not to implement — has an entry. "I didn't
log it" is the one failure this study cannot recover from.

`REPORT.md` — rewritten (not appended) at every milestone, after every method's grid completes,
and at least every 6 hours:

1. The ranked table: best passing config per method, sorted by `ms_fwd_bwd` at `D_in`, with the
   gate columns and the `D_out` number beside it.
2. `RECOMMENDED = <method, config>`; `SERIAL_FUSED_MS = <in> / <out>`;
   `RECOMMENDED_MS = <in> / <out>`; `SPEEDUP = <in> / <out>`; milestones reached.
3. Methods that failed the gate and *which input broke them* (parity, slow leak, `tau` gradient).
4. The occupancy-crossover and length-scaling tables once they exist.
5. What is in progress, and the exact command to resume it.

## The experiment loop

LOOP FOREVER:

1. Pick the next method in the order above; within a method, its configs in grid order; every
   config at both sites.
2. Write or update its `METHODS.md` entry (status `planned` → `implemented`).
3. Before an edit to `neuron.py` that could break a method already logged `ok`, copy it to
   `neuron_backup_<n>.py`. Then implement.
4. Run: `python bench_neuron.py --method <id> --config "<...>" --site <in|out> > bench.log 2>&1`
   (redirect everything; do NOT tee or let kernel output flood your context).
5. Read out: `grep "^RESULT" bench.log` — exactly one `RESULT` line per run in the TSV schema.
   Append it (prefix stripped) to `bench_results.tsv`.
6. Empty grep = crash: `tail -n 40 bench.log`, fix dumb bugs (typo, shape, dtype), re-run — a
   new row each time. After **three attempts or 45 minutes** on one method, log `crash` with the
   reason, restore `neuron.py` from the latest backup if you broke something, update
   `METHODS.md`, move on.
7. When a method's grid is complete, add a `#` comment to `bench_results.tsv` with its best
   passing config, update `METHODS.md` (`verdict`), and rewrite `REPORT.md` if the ranking changed.

**Timeout**: one config should finish in under 2 minutes; kill anything past 6 minutes and log
`crash` with `timeout` in notes. `serial_python` and `speculative_chunk` may legitimately be
slow — give those 15 minutes.

**Fairness**: same `bench_inputs.pt`, same seed, same B, same `detach_reset`, same surrogate for
every config. If you find the reference itself is wrong (reset applied in the wrong order relative
to the leak, a dtype mismatch with the repo's neuron), fix it, rerun `serial_fused`, and rerun
every method logged so far — everything is measured against the reference.

**Simplicity criterion**: when two configs pass the gate within noise, prefer the one with less
code and say so in the notes. A reshape of the existing kernel beats 200 lines of fragile kernel.

**Extended study** — once the method menu is exhausted, in this order, logged the same way:
length scaling (L ∈ {512, 1024, 2048, 4096, 8192}) for the top three passing methods; the
occupancy crossover (F above); a leak sweep (`beta` ∈ {0.5, 0.9, 0.95, 0.99}) checking the
measured `v_maxerr` against the predicted `beta^W` / `beta^(passes·C)`; `detach_reset=False`;
the 4·L tick axis (T = 4 ticks per token, membrane carried across all 8192 steps); then tune the
constants (C, W, passes, K, block size) of the top three for the single fastest passing config at
each site. Then go back to `LITERATURE.md`, pick the most promising unimplemented entry, and
implement it. Then combine the two best ideas. Repeat.

**NEVER STOP, NEVER ASK**: from the moment you read this file there is no one to talk to. Do NOT
pause to ask whether to continue, whether a result is interesting, whether a fallback is
acceptable, or whether this is a good stopping point — a question is an idle GPU until the
process is killed. Decide, log the decision, continue. If you believe the menu is exhausted,
re-read `METHODS.md` for a method whose `verdict` says "almost" and push it; re-read
`LITERATURE.md` for a skipped entry; combine near-misses; make the linear scan faster, since
every parallel method is bounded by it. The loop runs until the process is killed, period.

As a use case: you are started on a free GPU and nobody looks at the screen for a day or more.
`bench_results.tsv`, `METHODS.md`, `LITERATURE.md` and `REPORT.md` are the only things that will
be read — not your messages, not `bench.log` — so everything worth knowing must be in those
files, and a method that was tried but not logged was, for this study, never tried.