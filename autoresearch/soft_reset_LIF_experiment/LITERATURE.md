# LITERATURE — the literature ledger

One entry per paper or idea: name / where it comes from / the idea in two lines / the method id it maps
to / status. "verified" says whether the primary source was found by web search during this study.
Entries marked `menu` come from `program.md` and are not yet checked against the source.

Status words: `implemented` (has rows in `bench_results.tsv`), `planned`, `skipped (why)`.

## A. Spiking-neuron parallelisation

Web survey of 2026-11-22 (primary sources read; "partially" = only part of the source could be read).
Notation: `h_t = beta v_{t-1} + u_t`, `s_t = [h_t >= theta]`, `v_t = h_t - theta s_t`.

Two exact rewrites that several papers start from:
- **Decoupled forward**: `k_t = beta k_{t-1} + u_t` (membrane without reset, a linear filter),
  `r_t = beta (r_{t-1} + theta s_{t-1})` (reset trace), `h_t = k_t - r_t`.
- **Backward with detached reset**: `e_t = g_t sg(h_t - theta)`, `lambda_t = e_t + beta lambda_{t+1}`,
  `dL/du_t = lambda_t`. A constant-coefficient linear recurrence. Only the forward pass is hard.

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| Fused multi-step neuron | SpikingJelly 2.0.0, `_ops/plif/triton.py` (code read in the venv); PR #770 removed cupy in Oct 2026 | yes | One kernel per layer, loop over T inside, parallel over neurons only. | `serial_fused` | implemented |
| Torch-loop neuron | SpikingJelly Torch provider | yes | Python loop over T. | `serial_python` | implemented |
| FPT, Fixed-point Parallel Training | Feng, Gao, Du, Shi, Zhao, Wu, Miao, ICML 2025, PMLR 267:16711, arXiv:2506.12087 | yes | Picard on the spike train with a sigmoid: `u_(k) = Lambda c - V_th (Lambda - I) s_(k-1)`, `s_(k) = sigmoid(alpha_k (u_(k) - V_th))`, K = 3, alpha rising, e.g. (1, 3, 12). The paper uses a dense T x T matrix, not a scan. Approximate. Its own contraction condition needs a small alpha when the leak is slow. | `fpt` | implemented |
| FidLIF | He, Wu, arXiv:2610.16924 (Oct 2026), no venue | yes | Newton on the whole membrane trajectory of a sigmoid-relaxed **hard**-reset LIF. Each Newton step is an affine scan with elements `(a, b)`; K = 3. Backward: one reverse affine scan. Exact only for the relaxed neuron. Soft-reset form (derived here, not in the paper): `phi(h) = beta (h - theta sigma(alpha (h - theta))) + u`. | `newton_soft` | implemented |
| PRF, decoupled reset | Huang, Liu, Feng, Lin, Ren, Fu, Zhou, Xing, Cheng, arXiv:2410.03530 | yes | Membrane without reset by FFT convolution; the reset becomes a dynamic threshold `A_t = beta (A_{t-1} + s_{t-1})` in a serial loop. Exact rewrite, but the reset loop stays serial. Reported gain is almost all in the backward pass (forward 1.77x, backward 132x). | `decoupled_threshold` | implemented |
| PMSN | Chen, Wu, Ma, Yan, Wu, Tan, arXiv:2408.14917 (TNNLS 2026) | yes | Output compartment without leak and with non-negative input: `S_t = 1 iff floor(Csum_t/theta) > floor(Csum_{t-1}/theta)`. With signed input the last compartment must run serially (their ablation). | `cumsum_closed_form` | implemented |
| SpikingSSMs, SDN | Shen, Wang, Huang, Zhong, Guo, Lu, Zhang, Leng, AAAI 2025, arXiv:2408.14909 | yes | A frozen 1-D CNN (< 200 parameters, 8 channels, kernel 8) predicts the membrane with reset from the input. Trained for a hard reset and one tau. Spike accuracy falls from 99.95 % (tau 0.2) to 85.9 % (tau 1.0). | `sdn_predictor` | implemented |
| SPikE-SSM, PMBC | Zhong, Zhao, Wang, Guo, Zhang, Lu, Leng, arXiv:2410.17268 (IEEE TCDS 2026) | yes | Upper and lower bound on the spike train, tightened by convolutions: `s_low_t = 1 if k_t > m_up_t`, `s_up_t = 0 if k_t < m_low_t`. Exact only at convergence. Undetermined spikes after n iterations at tau = 0.1: 8.8 % (1), 4.5 % (5), 0.82 % (30). | `pmbc` | implemented |
| Bullet Trains | Morrill, Pehle, Zador, ICML 2026, arXiv:2603.13283 | yes | Event-based continuous-time LIF with hard reset: scan the affine dynamics over a chunk of input spikes, find the first output spike, reset, restart. Reverts to serial when a neuron spikes in every chunk. Not the discrete neuron. | `speculative_chunk` | implemented |
| Hard Resets Compose | repository `cheshireyang/hard-resets-compose` (README names Dekun Yang, 2026; no arXiv id found; not peer reviewed) | partially | The chunk map `v_in -> v_out` is piecewise affine and such maps compose. Soft reset: all pieces have slope `beta^C`, at most `1 + C(C+1)/2` pieces. Two-level scan: bounds, build the pieces per chunk, carry serially over chunks, replay, repair. Measured on an RTX 5090 against a **tuned serial kernel** (time tiles of 2–32 steps, next tile prefetched): the soft-reset scan reaches 0.16x of serial at N = 1024 and 0.06x at N = 8192. Their statement: time-parallel evaluation pays only when serial evaluation leaves the GPU idle. They also report SpikingJelly 2.0 as 5.3–9.1x slower than their tuned serial kernel. | `map_scan` | implemented |
| PSN, masked PSN, sliding PSN | Fang, Yu, Zhou, Chen, Chen, Ma, Masquelier, Tian, NeurIPS 2023, arXiv:2304.12760 | yes | No reset. `H = W X` with a learnable T x T matrix (sliding: k shared taps), `S = [H >= B]`. A different neuron. | `psn`, `sliding_psn` | implemented |
| SPSN | Yarga, Wood, arXiv:2306.12666 (IJCNN 2023) | yes | Leaky integration by FFT, then `s ~ Bernoulli(sigmoid(u))`. No reset. A different neuron. | `spsn` | implemented |
| PSU / IPSU / RPSU | Li, Sun, He, Dong, Zhao, Zeng, arXiv:2402.00449 | yes | `S = [A I - B sigmoid(A I) >= V_th]`: one reset estimate from the reset-free membrane (one Picard step without iteration). Approximate. | `psu` | implemented |
| MPE-PSN | Chen, Yu, Zhan, Yao, Shao, arXiv:2409.04978 | yes | Estimate the previous membrane from the input alone: `P = sigmoid(I)`, `u_hat = (1 - Bernoulli(P)) I`. Built for T = 4–20. A different neuron. | `mpe_psn` | implemented |
| Tachyon | Miruvor AI web report, July 2026 (no authors, no paper) | partially | Draft with a reset-free scan, verify, commit up to the first mismatch, restart there; reverse affine scan for the adjoint. Hard reset only. Wins reported only for narrow, sparse settings. | `speculative_chunk` | covered by that entry |
| DSN, dynamic decay neuron | Huang, Yao, Pan, Lv, Xu, Zheng, Xu, Li, arXiv:2602.01133 | yes | Replaces the reset by an input-dependent decay: a gated linear recurrence, run by a selective-scan kernel. A different neuron. | — | skipped (changes the neuron; no reset to compute) |
| PFSN | Yang, Peng, Xie, Yang, Zhan, Liu, NeurIPS 2026 poster 151458 (abstract and README only) | partially | LIF as one fixed-point map with a learnable temporal operator; 1–5 iterations. Same family as FPT. | `fpt` | covered by that entry |
| Blocks | Taylor, King, Harper, NeurIPS 2023, arXiv:2311.11390 | yes | With an absolute refractory period of T_R steps a neuron fires at most once per T_R steps, so each block of T_R steps is parallel. Exact for that neuron only. | — | skipped (needs a refractory period the neuron does not have) |
| NeuronSpark | Tang, Zhu, Chen, Tao, arXiv:2603.16148 | yes | Spiking SSM language model with soft-reset PLIF carried over time. It keeps a fused serial Triton kernel and says the affine prefix scan does not solve the PLIF recurrence. | `serial_triton` | supports the serial choice |
| EXODUS / SLAYER | Bauer, Lenz, Haghighatshoar, Sheik, Frontiers in Neuroscience 2023, arXiv:2205.10242 | yes | Backward pass of a soft-reset LIF in closed form; with a detached reset the factor per step is the leak. Confirms the linear backward recurrence. | backward of every method | implemented in the own kernels |
| IF closed forms from ANN-to-SNN conversion | Bu et al., arXiv:2303.04347 (QCFS); You et al., ICML 2024, arXiv:2406.03470 (SpikeZIP-TF, ST-BIF); Hao et al., ICML 2025, arXiv:2412.13610 | yes | Non-leaky IF: total spike count = clipped floor of the total input. These hold for totals, not for the order of spikes with signed input. Signed-input extension (derived in this study, not from a source): if `beta = 1`, `v_0 < theta`, `u_t < theta`: `n_t = max(0, cummax(floor(cumsum(u)_t / theta)))`. | `cumsum_closed_form` | implemented |
| MF-PSN | Xue, Fang, Ma, Huang, Zhou, Tian, Masquelier, Zhou, arXiv:2501.14490 | yes | Sliding PSN with power-of-two taps. No reset. | `sliding_psn` | covered by that entry |
| D-RF | Zhang et al., arXiv:2509.17186 | yes | Oscillator branches by FFT; the reset is replaced by a threshold that rises after crossings of the reset-free state. A different neuron. | — | skipped (changes the neuron) |
| Reset-free spiking SSMs | Stan & Rhodes, Scientific Reports 2024; SHaRe-SSM arXiv:2510.14386; P-SpikeSSM arXiv:2406.02923; S5-RF arXiv:2504.00719 | partially | Threshold the SSM output directly; the SSM state is the membrane; no reset. The usual way spiking SSMs avoid this problem. | `psn` family | covered (no-reset floor) |
| Per-token micro-step neurons | SpikingMamba arXiv:2510.04595; MAR arXiv:2601.21503 | yes | The membrane evolves over a few micro-steps inside one token and is not carried across tokens. This is what the repo trains today (T = 4, reset per token). | — | not a method here: the study asks for the membrane carried over tokens |
| Fixed refractory by convolution | Zhang, arXiv:2504.17751 | yes | Membrane without reset; sparsity by a fixed refractory window applied as a convolution. A different neuron. | — | skipped (changes the neuron) |
| GPU SNN simulators (GeNN, Brian2CUDA, NeuronGPU, CARLsim, EDLUT) | search summaries only | no | Parallel over neurons and synapses; none found that is parallel over time. | — | nothing to implement |

Speed-up caveat from the survey: PRF, SDN, PMBC, FPT, SPSN and EXODUS report speed-ups against BPTT
without a fused serial kernel. A tuned fused serial baseline was found only in Hard Resets Compose and
Tachyon. This study always compares against a fused serial kernel.

## B. Parallel-in-time numerical methods and parallel evaluation of recurrences

Web survey of 2026-11-22. Short answer of the survey: no published method gives an exact parallel forward
pass in a few iterations when beta is near 1; the backward pass is one linear reverse scan and needs no
iteration.

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| Parareal | Lions, Maday, Turinici, C. R. Acad. Sci. Paris 332 (2001) 661–668; analysis: Gander & Vandewalle, SIAM J. Sci. Comput. 29 (2007) 556–578; Gander & Hairer, DD17 (2008) | yes (analysis papers) | `U_{n+1}^{k+1} = F(U_n^k) + G(U_n^{k+1}) - G(U_n^k)`. For this neuron: F = C serial steps, G(v) = beta^C v + c, so the correction is `beta^C (U_n^{k+1} - U_n^k)`. Chunk n is exact after n iterations for any G. The fixed point is bit-identical to the serial result. The rate theorems need F - G Lipschitz, which fails at the threshold. With beta = 1 there is no damping: about N sweeps. | `parareal` | implemented |
| Newton across the steps | Bellen & Zennaro 1989; Chartier & Philippe 1993 (via Gander, "50 Years of Time Parallel Time Integration", 2015) | partially | Write the recurrence as a fixed point of the whole trajectory and apply Newton. With slope beta: freeze the spikes, solve the linear recurrence, threshold again. | `fpt` (same iteration) | implemented |
| Multiple shooting | classical; Gander & Hairer show parareal = Newton multiple shooting with the Jacobian replaced by a G difference | yes | Newton on the chunk boundary states. The Jacobian of a chunk is beta^C almost everywhere, so with the affine G this is the same iteration as parareal. | `parareal` | same method, run once |
| MGRIT | Falgout, Friedhoff, Kolev, MacLachlan, Schroder, SIAM J. Sci. Comput. 36 (2014) C635 | partially | Multigrid on the time axis; the two-level case with F-relaxation is parareal. The coarse problem here is already a scalar affine recurrence, so more levels add nothing. | `parareal` | covered |
| PFASST | Emmett & Minion 2012 | partially | Parareal with spectral deferred correction sweeps. Needs quadrature nodes and smooth solutions. | — | skipped (not applicable to a discrete map) |
| Waveform relaxation | Lelarasmee, Ruehli, Sangiovanni-Vincentelli, IEEE Trans. CAD 1 (1982) 131 | partially | Iterate on whole trajectories; Jacobi uses old data everywhere, Gauss–Seidel the newest. For one scalar recurrence the split is linear part versus reset: Jacobi = the Picard iteration on the spikes, Gauss–Seidel in time = the serial loop. | `fpt` (Jacobi), `serial_triton` (Gauss–Seidel) | same methods, see `waveform_relaxation` in METHODS.md |
| Jacobi / block Jacobi–Gauss–Seidel for sequential computation | Song, Meng, Liao, Ermon, ICML 2021, arXiv:2002.03629 | yes | A sequential computation is a triangular system; Jacobi reaches the sequential result in at most T iterations, the block form in at most M block iterations, with no smoothness needed. Block form with blocks = chunks is `chunk_warm`. | `chunk_warm` | implemented |
| DEER | Lim, Zhu, Selfridge, Kasim, ICLR 2024, arXiv:2309.12252 | yes | Newton on the whole trajectory; each step is a linear recurrence solved by a prefix scan. Quadratic convergence needs a Lipschitz Jacobian; the backward pass needs one linear solve and no iterations. | `newton_soft` | implemented |
| quasi-DEER / ELK | Gonzalez, Warrington, Smith, Linderman, NeurIPS 2024, arXiv:2407.19115 | yes | Diagonal Jacobian (elementwise scan); ELK adds damping as a Kalman smoother. Newton ends in at most T iterations for any finite stand-in Jacobian. The neuron state is scalar, so quasi-DEER = DEER here; the slope is beta <= 1, so damping is not needed. Their own timing: sequential 96 ms, quasi-DEER 642 ms at T = 10 000. | `newton_soft` | same method |
| Predictability | Gonzalez, Kozachkov, Zoltowski, Clarkson, Linderman, NeurIPS 2025, arXiv:2508.16817 | yes | Newton steps needed depend on the largest Lyapunov exponent. For this neuron the exponent is ln beta < 0 but the Jacobian is not Lipschitz at the threshold, so the theory does not transfer. | — | background |
| Unifying framework | Gonzalez et al., arXiv:2509.21716 | yes | Newton, quasi-Newton, Picard and Jacobi are one iteration with a different stand-in Jacobian. | `fpt`, `newton_soft` | background |
| ParaRNN | Danieli, Rodriguez, Sarabia, Suau, Zappella, arXiv:2510.21450 | yes | Newton plus parallel reduction for GRU/LSTM cells in a fused CUDA kernel, 3 iterations. Gains over an existing scan kernel are 1.1–2.6x. | — | engineering template only |
| Prefix-scan networks | Kogge & Stone 1973; Ladner & Fischer 1980; Brent & Kung 1982; Sklansky 1960; Blelloch 1990 | partially | Any associative operator can be prefix-combined in log depth; for `x_t = a_t x_{t-1} + b_t` the element is the pair (a, b). A scan reorders float additions, so it is not bit-identical to the serial loop. | linear scan | see section D |
| Parallel linear recurrence on GPU | Martin & Cundy, ICLR 2018, arXiv:1709.04057 | yes | CUDA kernel for linear recurrences. Speed-up over a serial-in-time CUDA kernel at batch 1: 0.22–0.86x at T = 256, 1.0–3.4x at T = 4096, 17–42x at T = 65 536. | linear scan | supports "serial wins at L = 2048 with many neurons" |
| Mamba / Mamba-2 / Mamba-3 scans | Gu & Dao arXiv:2312.00752 (code read); Dao & Gu arXiv:2405.21060; Lahoti et al. arXiv:2603.15569 | yes | Mamba: one CUDA block per (batch, channel), `cub::BlockScan` on (a, b) pairs, chunks of 2048 steps. Mamba-2/3: chunked matrix products (gain comes from a matrix-valued state). | linear scan | not better for a scalar recurrence |
| Anderson acceleration | Walker & Ni 2011; Bian, Chen, Kelley 2021 | partially | Mix the last iterates of a fixed-point map. A mix of binary spike trains is not binary and the map is discontinuous. | — | skipped (not applicable to the exact problem) |
| Time-parallel simulation with fix-up | Lin & Lazowska 1991; Greenberg, Lubachevsky, Mitrani 1991 (via Fujimoto's course slides) | partially | Guess each interval's start state, simulate all intervals in parallel, repair with the neighbour's final state until nothing changes. The classical name for speculate, verify, repair. | `spec_verify` | implemented |
| Not found | — | — | Convergence theory of parareal/MGRIT for state-triggered jumps; parareal applied to spiking neurons; an exact parallel soft-reset LIF with beta near 1 in O(1) iterations. | — | — |

## C. Computer arithmetic — the neuron as a carry chain

Web survey of 2026-11-22.

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| Carry look-ahead, parallel-prefix adders | Weinberger & Smith 1958; Ladner & Fischer, JACM 27 (1980) 831; Brent & Kung 1982 | yes (Ladner–Fischer) | Each bit turns the incoming carry into the outgoing one by kill, generate or propagate; these maps are closed under composition. For the neuron the carry is the real-valued membrane and the block map must stay in a small closed family: true for beta = 1 with bounded input (pair: block sum, largest prefix sum), not for beta < 1. | `cumsum_closed_form`, `map_scan` | see those entries |
| Carry-select, conditional-sum | Bedrij 1962; Sklansky 1960 | partially | Compute each block for every possible carry-in and select when the carry arrives. Here: one candidate per affine piece of the block map; `v_out = v_out,j + beta^C (v_in - g_j)`. | `carry_select` | implemented |
| Carry-skip | Lehman & Burla 1961 | partially | If a block only propagates, the carry jumps over it. Here: a block is "silent" if `beta^t v_max + X_t < theta` for all t. A cheap filter for silent neurons. | — | skipped (no gain in a dense tensor kernel; the real input fires at 6–12 %) |
| Speculative adders with error detection | Nowick 1996; Verma, Brisk, Ienne, DATE 2008; Pippenger 2002 | partially | The longest carry chain is short on average, so guess, flag an error, correct in one more step. Here: the influence of the membrane decays as beta^k; start each block W steps early from 0; error flag = block start state differs from the previous block's end state; correction = re-run. | `spec_verify` | implemented |
| Saturated accumulation | Papadantonakis, Kapre, Chan, DeHon, IEEE Trans. Comput. 58 (2009) 208 | yes | `y -> min(max(y + x, m), M)` is closed under composition, so a clamped accumulator is a prefix scan with 3 numbers per element. The neuron does not clamp the state, and its level form is not closed (survey's own argument). | bounds in `carry_select` | used for the membrane bounds |
| Kogge–Stone recurrence class | Kogge & Stone, IEEE Trans. Comput. C-22 (1973) 786 | yes | `x_i = f(b_i, g(a_i, x_{i-1}))` is solved in log depth if f is associative and g distributes over f. Running maximum and clamped sums are in the class; "affine, then conditional subtract" is not, with constant size. | — | explains why only beta = 1 has a closed form |

Derivations from the survey (checked there on CPU; not found as published results):
- **Running-maximum formula, beta = 1, signed input**: `S_t = v_0 + cumsum(u)_t`, `N_t = max(0, floor(cummax(S)_t / theta))`,
  `s_t = N_t - N_{t-1}`. Exact iff every increment is 0 or 1 (true when `u_t < theta`, `v_0 < theta`). Related to
  the Lindley recursion (1952) and the Skorokhod map. → `cumsum_closed_form mode=cummax`, implemented.
- **Min-plus dual, beta = 1, non-negative input of any size**: `N_t = t + min(0, min_{s<=t}(floor(S_s/theta) - s))`.
  → `cumsum_closed_form mode=minplus`, implemented (exact on the counting input; fails on the signed real input).
- **beta < 1, pieces of a block map**: on a start interval no wider than theta, C steps give at most C + 1 affine
  pieces, all with slope beta^C. Measured mean piece count: 1.2 (beta 0.5), 3–4 (beta 0.9), 5–13 (beta 0.99).
- **Validity interval + affine map** (constant size, associative, with a FAIL flag): run a block once from a guess g;
  `lo = g + max over spike steps of (theta - h_t)/beta^t`, `hi = g + min over non-spike steps of (theta - h_t)/beta^t`;
  inside `[lo, hi)` the block is `v_out = beta^C v_in + b`. → implemented: the interval certificate of `spec_verify`
  (`cert=interval`) and the candidate maps of `map_scan`.
- **Look-back speculation with a bit-equality certificate**: start every block W steps early from 0 with the serial
  float operations; a block is correct if its start state is bit-equal to the previous block's end state. Measured
  coalescence time in float32 (median / 99.9 %): 25 / 36 steps (beta 0.5), 74 / 125 (0.8), 158 / 302 (0.9),
  328 / 725 (0.95), ~1900 (0.99). → implemented as `spec_verify` (`cert=bit`).

## D. Signal processing — the neuron as a sigma–delta modulator

Web survey of 2026-11-22.

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| First-order sigma–delta | Güntürk, Lagarias, Vaishampayan, IEEE Trans. Inf. Theory 47 (2001) 1735; Gray 1990 | yes (Güntürk) | `u_{n+1} = u_n + x_n - q(u_n)`: the leak-free soft-reset IF neuron. For input in [0, theta]: `N_t = floor((v_0 + sum u)/theta)`. Signed input or input above theta is "overload". | `cumsum_closed_form` | implemented |
| Leaky sigma–delta | Feely & Chua, IEEE Trans. Circuits Syst. 38 (1991) 1293 (ERL memo M90/116 read) | yes | The neuron with constant input, exactly: `u_{n+1} = p u_n + g (x - sgn(u_n))`. The n-step map has at most n + 1 straight segments, each of slope p^n; for p < 1 one globally stable limit cycle (mode locking); p = 1 is a circle translation. | `map_scan` | gives the piece bound |
| Contracted rotations | Laurent & Nogueira, arXiv:1704.05130 (Ding & Hemmer 1987; Bugeaud 1993) | yes | `x -> {lambda x + delta}` is the neuron for constant input below theta. Closed-form firing rate as a devil's staircase in the input. Not usable for time-varying input. | — | background |
| Time-interleaved delta–sigma | Khoini-Poorfard, Lim, Johns, IEEE Trans. Circuits Syst. II 44 (1997) 634 | yes | Rewrite the loop as an M-input block filter with M cross-coupled modulators. Some coupling paths have no delay, so the M quantizers still form a chain: the dependency is not broken. | `sigma_delta_lookahead` | skipped (exact but no gain) |
| Unrolled / speculative digital modulators | Hatami et al. 2014; Johansson, MSc thesis Chalmers 2014; Tanio et al. 2016; Dinis et al. 2018 | partially | Unroll N steps and precompute both outcomes of each bit, then select "in the manner of a carry-select adder": 2^N modulators. The neuron allows enumerating state pieces (at most N + 1) instead of bit patterns. | `carry_select` | implemented (same idea) |
| Look-ahead / trellis sigma–delta (SACD) | Kato, AES 112 (2002); Harpe, Reefman, Janssen, AES 114 (2003); Janssen & van Roermund 2011 | partially | Choose output bits by a tree search over a future error cost. The bit stream is different by design. | `sigma_delta_lookahead` | skipped (changes the output) |
| Look-ahead and block IIR | Parhi & Messerschmitt, IEEE Trans. ASSP 37 (1989) 1099 and 1118 | partially | `y_n = a^M y_{n-M} + sum_{k<M} a^k u_{n-k}`: the affine pass-through of a block of the linear part. | linear scan | used in every chunk method |
| Quantizer and decision-feedback loops | Parhi 1991; Kasturia & Winters 1991; Parhi 2004 | partially | Pipeline a loop with a finite-level quantizer by precomputing for each possible decision, then selecting. The neuron is a decision-feedback loop with an infinite feedback filter and one real state. | `carry_select` | same idea |
| Modulo in a loop = filter + compensation | Gu & Parhi 2007 | partially | A filter with a modulo in the loop equals a linear filter whose input carries a finite-level compensation signal. This is the decoupled-reset rewrite. | `decoupled_threshold` | implemented |
| Fastest GPU linear recurrence (PLR) | Maleki & Burtscher, ASPLOS 2018; Merrill & Garland, NVIDIA TR 2016-002 | yes | Solve each chunk from zero state, then add carry times precomputed factors d, d^2, ...; one pass, about 2n memory moves; reaches memcpy throughput. The correction is skipped once d^k is below precision. | linear scan | the serial unrolled kernel does the same at the bandwidth limit |
| Viterbi sliding-block decoding | Fettweis & Meyr 1989; Black & Meng 1997 | partially | Decode overlapping blocks independently because survivor paths merge. The overlap is the look-back W. | `chunk_cold`, `spec_verify` | implemented / planned |

Survey conclusion on "scan versus matmul versus FFT" for a first-order IIR: the scan is already at the memory
limit; a block matmul costs O(L C) and an FFT O(L log L), so neither can beat a memory-bound O(L) pass. No GPU
benchmark of the three was found in the literature. This study measured them (METHODS.md, "D: the linear scan"):
fused serial pass 0.150 / 0.289 ms, chunked matmul 0.55 / 1.01 ms, chunked cumsum 0.85 / 1.72 ms, FFT 1.8 / 3.9 ms.
The survey's conclusion holds.

## E. Hybrid dynamical systems and speculation over state machines

Web survey of 2026-11-22.

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| Firing maps of IF neurons | Keener, Hoppensteadt, Rinzel, SIAM J. Appl. Math. 41 (1981) 503; Carrillo & Ongay 2001 | partially | The map from one firing time to the next; origin of the Poincaré-map view. | `map_scan`, `speculative_chunk` | background |
| Lyapunov exponent of IF neurons | Coombes, Phys. Lett. A 255 (1999) 49 | yes | A perturbation shrinks between spikes and is multiplied by a saltation factor at each spike. In the discrete soft-reset map the factor is 1, so the exponent is exactly ln beta; the theta-jumps are a finite-gap effect. | — | explains the contraction |
| No chaos in a single IF neuron | Brette, J. Math. Biol. 48 (2004) 38; Zhou, Rangan, Sun, Cai, Phys. Rev. E 80 (2009) 031918 | partially / yes | The spike map is increasing on its range; a single linear IF neuron under any input cannot be chaotic. The perfect integrator is conjugate to a rotation and never synchronises. | — | explains why beta = 1 defeats every truncation |
| Reliability of spike timing | Mainen & Sejnowski, Science 268 (1995) 1503; Brette & Guigon, Neural Comput. 15 (2003) 279 | partially | Aperiodic input makes spike times reproducible under a change of the initial condition. No convergence rate given. | `chunk_cold`, `spec_verify` | the property those methods rely on |
| Count mismatch at most 1 | Dou, Chen, Lin, Xiao, arXiv:2603.10854 | yes | Two copies of a 1-D IF neuron with common input: spike times interlace, so the spike counts differ by at most 1 (hard reset, continuous time). The discrete soft-reset version was derived in this study. | — | background |
| Piecewise contractions | Catsigeras, Guiraud, Meyroneinc, arXiv:1702.03599; Bruin & Deane 2009; Nogueira, Pires, Rosales 2014 | yes / partially | For an injective piecewise contraction with N pieces the number of itineraries of length n is at most (N-1) n + 1; almost all such maps are asymptotically periodic. Autonomous maps only. | `map_scan` | supports "few pieces" |
| FSM prefix over transition tables | Hillis & Steele, CACM 29 (1986) 1170; Ladner & Fischer 1980 | yes | Replace each input symbol by its state-to-state table and compose tables with a parallel prefix. | `carry_lookahead` | implemented |
| Data-parallel FSMs | Mytkowicz, Musuvathi, Schulte, ASPLOS 2014 | yes | Run each chunk from every start state; most machines converge to at most 16 active states; three phases: chunk maps in parallel, start states serially, outputs in parallel. The neuron is the continuous version: the active set is an interval that shrinks by beta^C. | `carry_select`, `map_scan` | implemented |
| Speculation then validation | Luchaup, Smith, Estan, Jha 2011; Zhao, Wu, Shen, ASPLOS 2014; Qiu et al., ASPLOS 2021 | yes / partially | Guess each chunk's start state, scan in parallel, let the previous chunk run on until its state equals the stored one. The gain depends on validation ending fast. | `spec_verify` | implemented |
| Rank convergence | Maleki, Musuvathi, Mytkowicz, PPoPP 2014 | yes | Start each processor from an arbitrary vector; a fix-up loop re-runs from the neighbour's result until the vectors agree. Worst case serial. | `spec_verify` | implemented |
| Symbolic chunk summaries | Raychev, Musuvathi, Mytkowicz, SOSP 2015 | partially | Run a chunk with the unknown start state as a symbol; keep the path condition and an expression for the end state. Path condition = validity interval, expression = affine map. | `map_scan` | implemented |
| Segmented scan | Blelloch 1989 | partially | A scan that restarts at flagged positions. Needs the reset positions before the scan; here they depend on the accumulated value. | — | skipped (not applicable) |
| Saltation matrix | Kong, Payne, Zhu, Johnson, Proc. IEEE 112 (2024) | partially | First-order map of a perturbation across a reset. In discrete time the event-time term becomes a spike at the threshold; the surrogate gradient plays that role in the backward pass. | — | background |

Own measurements quoted by the surveys (CPU toy checks, Gaussian input, not from the literature): two copies of
the neuron with the same input become bit-identical in float32 after 24 / 76 / 169 / 388 steps (median) for
beta 0.5 / 0.8 / 0.9 / 0.95, and never for beta = 1.

## F. GPU engineering

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| Latency hiding | Volkov, PhD thesis UC Berkeley, UCB/EECS-2016-143; GTC 2010 talk "Better Performance at Lower Occupancy" | yes | Little's law: needed concurrency = latency x throughput. Memory latency is ~400 cycles against 6–24 for arithmetic. More independent work per thread reaches the same speed with fewer warps. | `serial_triton` | implemented (U loads issued ahead) |
| Tuned serial kernel beats time-parallel scans | Hard Resets Compose repository (RTX 5090, Triton 3.4); Laszlo, Giles, Appleyard, ACM TOMS 42 (2016) art. 31 | partially / yes | One thread per recurrence, time tiles of 2–32 steps with the next tile prefetched. For batches of tridiagonal systems the serial algorithm, one system per thread, beats the parallel reduction up to sizes in the thousands. | `serial_triton` | implemented, confirmed here |
| Memory-latency hiding, own probe | this study, 2026-11-22 | measured | SpikingJelly's kernel waits ~0.4 us for one load per time step. Issuing 8–32 loads before using them cuts the forward pass from 0.78 ms to 0.19–0.23 ms, bit-identical. | `serial_triton` | implemented |
| Bandwidth floor, own probe | this study, 2026-11-22 | measured | Copy of `[2048,16,1536]` fp32: 0.267 ms (~1.5 TB/s). In the forward kernel every 200 MB stream costs ~0.14 ms. RTX 5090: 512-bit GDDR7 at 28 Gbps = 1792 GB/s pin bandwidth, 170 SMs (survey, from NVIDIA pages and reviews). | all | — |
| Triton facts | Triton docs and source (survey) | yes | `tl.associative_scan`, `tl.cumsum`, `static_range`; `enable_fp_fusion` default differs between Triton versions, so it must be set explicitly to match a reference bit for bit. | all kernels | fusion is switched off in every kernel here |
| accelerated-scan | github.com/proger/accelerated-scan (README) | yes | Ready kernels for `x[t] = gate[t] x[t-1] + token[t]`, forward and backward, chunked with warp shuffles. | linear scan | not used (no downloads) |

## G. The backward pass: tapes, reversible networks, online gradients

Fifth search round (2026-11-22), on the part of the cost that the forward-pass literature does not cover: what the
backward pass must read.

| name | source | verified | idea | method id | status |
|---|---|---|---|---|---|
| Activation compressed training (ActNN) | Chen et al., ICML 2021, PMLR 139:1803–1813, arXiv 2104.14129 | yes | Save the activations for the backward pass in about 2 bits each, with random rounding, so the gradient is an unbiased estimate. | tape of `serial_triton` | implemented as `tape=u8` (deterministic rounding, decode at the cell midpoint); fewer bits: see "how small can the tape be" in METHODS.md |
| Backprop with approximate activations | Chakrabarti & Moseley, NeurIPS 2019, arXiv 1901.07988 | yes | Keep the exact activation only for the forward pass, keep a low-precision copy (about 4 bits) for the backward pass. Errors do not pile up over layers, because every forward value is exact. | tape of `serial_triton` | implemented: the same split, float32 forward and 8-bit tape |
| Lossy compressed checkpoints for adjoints | Kukreja et al., Euro-Par 2019 (arXiv 1810.05268); Kukreja et al., Geosci. Model Dev. 15 (2022) 3815 (arXiv 2009.12623) | yes | Reverse-mode solvers of wave equations save the forward states compressed with loss and recompute the rest from checkpoints. | — | background. Checkpoint + recompute does not fit here: a recompute needs x again, 4 bytes per step against 1 byte of tape. |
| Reversible RNNs | MacKay, Vicol, Ba, Grosse, NeurIPS 2018, arXiv 1810.10999 | yes | Rebuild the hidden states backwards instead of saving them. A network that forgets cannot be reversed exactly, so a few bits per step are stored. | `reversible_backward` | skipped (probe): the rebuilt membrane is off by more than θ after 22 steps (`in`) / 52 steps (`out`), because every reverse step divides by β |
| Online training through time (OTTT) | Xiao, Meng, Zhang, He, Lin, NeurIPS 2022, arXiv 2210.04195 | yes | Learn forward in time from presynaptic traces and the instantaneous gradient; constant memory in the number of time steps. | `truncated_carry_backward` | skipped (probe): without the temporal carry the cosine to the exact gradient is 0.86 (`in`), 0.60 (`out`), 0.31 (slow leak) |
| Spatial learning through time (SLTT) | Meng et al., ICCV 2023, arXiv 2302.14311 | yes | Drop the temporal path of backpropagation through time; keep only the path through the layers at each time step. | `truncated_carry_backward` | same probe |
