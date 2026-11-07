# Autoresearch report: spiking Mamba-3 on WikiText-2 (branch `autoresearch/nov6`)

Working notes. Runs on 2026-11-06, 15:09 to 21:30, on one RTX 5090 (32 GB).
Sources: `results.tsv`, `git log master..autoresearch/nov6`, and the run logs.
Estimates are marked **(est.)**.

## 1. Summary

| | val_ppl | spike_rate | rate in_proj | rate out_proj |
|---|---|---|---|---|
| Spiking baseline (`2de3d1f`) | 155.55 | 0.056 | 0.080 | 0.044 |
| **Best (`ab9fa2c`)** | **108.43** | **0.076** | **0.065** | **0.082** |
| Dense teacher (from `program.md`, not re-measured) | 142.9 | n/a | n/a | n/a |

- 70 runs: 25 keep, 45 discard, 0 crash, 0 over the 0.20 rate cap.
- val_ppl fell by 47 points (−30 %). Spike rate stayed at 0.076, far under the 0.20 cap.
- About 44 of the 47 points (over 90 %) came from the training recipe (KD, LR schedule, optimizer), not from the neuron.
- Safe claim: "Spiking Mamba-3 61M with 7.6 % spike rate reaches 108.4 val_ppl on WikiText-2 (740 steps)."
- **Not** a safe claim: "Spiking beats dense." The teacher used a weaker recipe. See section 6.

## 2. Setup

- **Model:** Mamba-3 SISO, d_model 384, 6 layers, d_state 128, expand 2, gated MLP 768, Llama-3.1 vocab of 128,256 (embedding tied with the head). 60.6M parameters, 49.3M of them in the embedding.
- **Spiking:** one SpikingJelly `ParametricLIFNode` (step_mode "m") in front of `in_proj` (384 neurons per layer) and `out_proj` (768 neurons per layer). Each token goes in T=4 times. The membrane resets for each token. The output is spike count / T, in {0, 0.25, 0.5, 0.75, 1}.
- **Data and budget (fixed):** WikiText-2, seq_len 1024, batch 16, 740 steps (5 epochs), seed 0.
- **Metric:** val_ppl on the WikiText-2 validation split. Rule: spike_rate ≤ 0.20.
- **Teacher:** frozen dense 61M Mamba-3 (`runs/teacher-dense-61m-wikitext2/final.pt`), trained with the config schedule: peak LR 1e-3, warmup 5 %, min LR 1e-5.
- **Noise:** the Mamba-3 backward pass is not deterministic. A rerun of `06b39f2` gave 109.98 vs 109.89. The baseline gave 155.55 vs 154.46 in `program.md`. I treat gaps under about 0.5 ppl as noise.
- **Cost per run:** about 140 s and 8.6 GB without KD. About 290 s and 19.4 GB with KD, because of the two 4 × 1024 × 128,256 logit tensors.

## 3. Final recipe (diff of `ab9fa2c` against `master`)

| Knob | master | best |
|---|---|---|
| PEAK_LR | 1e-3 | **7e-3** |
| WARMUP_FRACTION | 0.05 | **0.50** |
| MIN_LR | 1e-5 | **5e-4** |
| GRAD_CLIP | 1.0 | **0.5** |
| AdamW betas | (0.9, 0.95) | **(0.9, 0.99)** |
| KD_WEIGHT / KD_TEMPERATURE | 0 / 1.0 | **0.5 / 1.0** |
| Gain before `out_proj` neuron | none | **learnable per channel, init 1.5** |
| Unchanged | | T=4, init tau 2.0, v_th 1.0, hard reset to 0, ATan (alpha 2), detach_reset, WD 0.1 |

The code change is about 10 lines: the optional `gain` in `SpikingLinear`, used only for `out_proj`.

## 4. Progress (kept runs only)

```
val_ppl
155.6 ■ baseline
141.5 ■■■■■■■■■■■■■■ KD 0.5
140.8 ■■■■■■■■■■■■■■■ WD 0.3
140.3 ■■■■■■■■■■■■■■■ selective input scale (idea 5)
139.5 ■■■■■■■■■■■■■■■■ ATan alpha 1.0
128.6 ■■■■■■■■■■■■■■■■■■■■■■■■■■■ peak LR 1.5e-3
122.6 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ peak LR 2.5e-3
120.9 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ peak LR 4e-3
116.9 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ warmup 10 %
114.6 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ warmup 20 % / peak LR 5e-3
112.2 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ warmup 30 % → 50 %
       (cleanup: removed selective scale, WD back to 0.1, ATan back to alpha 2: all ties)
111.1 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ min LR 1e-4 → 5e-4
110.3 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ beta2 0.99
110.0 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ grad clip 0.5
109.9 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ peak LR 7e-3
109.4 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ out_proj gain (init 1.0)
109.0 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ out_proj gain init 2.0
108.4 ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ out_proj gain init 1.5
```

Gain by group (approx.):

| Group | val_ppl change |
|---|---|
| KD 0.5 | −14 |
| LR schedule (peak, warmup, min LR) | −29 |
| Optimizer (beta2, clip) | −1 |
| out_proj gain | −1.5 |
| Everything else kept | about 0 (later removed as ties) |

## 5. All experiments by theme

Values are val_ppl / spike_rate. "Ref." is the best kept result at that time.

### 5.1 Distillation

| Run | val_ppl | rate | Ref. | Verdict |
|---|---|---|---|---|
| KD 0.5, temp 1.0 | 141.50 | 0.058 | 155.55 | **keep** |
| KD 0.8 | 142.14 | 0.060 | 141.50 | worse |
| KD temp 2.0 | 148.38 | 0.065 | 141.50 | worse |
| KD 0.7 (tuned LR) | 116.54 | 0.093 | 112.22 | worse |
| KD 0.3 (tuned LR) | 114.40 | 0.093 | 112.22 | worse |
| KD annealed 0.5 → 0 | 121.46 | 0.066 | 109.89 | much worse |
| KD ramped 0.25 → 0.75 | 113.56 | 0.064 | 109.89 | worse |
| KD temp 1.5 | 114.04 | 0.068 | 109.89 | worse |
| KD temp 0.8 | 110.91 | 0.062 | 109.89 | worse |

The student beats the teacher, yet removing KD late costs 11.6 ppl. KD acts as a regularizer, not as a source of better answers.

### 5.2 Learning rate, schedule, optimizer

| Run | val_ppl | rate | Ref. | Verdict |
|---|---|---|---|---|
| peak LR 1.5e-3 | 128.57 | 0.074 | 139.50 | **keep** |
| peak LR 2.5e-3 | 122.61 | 0.085 | 128.57 | **keep** |
| peak LR 4e-3 | 120.94 | 0.097 | 122.61 | **keep** |
| peak LR 6e-3 (warmup 5 %) | 154.65 | 0.168 | 120.94 | unstable |
| warmup 10 % / 20 % / 30 % / 50 % | 116.85 / 114.61 / 112.35 / 112.20 | ~0.085 | | **keep** each |
| peak LR 5e-3 (warmup 20 %) | 114.54 | 0.082 | 114.61 | keep (tie) |
| min LR 1e-4 / 5e-4 | 111.52 / 111.10 | ~0.095 | 112.22 | **keep** |
| min LR 1e-3 | 111.77 | 0.097 | 111.10 | worse |
| beta2 0.99 | 110.31 | 0.061 | 110.99 | **keep** |
| beta2 0.999 | 112.40 | 0.056 | 110.31 | worse |
| grad clip 0.5 | 109.98 | 0.064 | 110.31 | **keep** |
| grad clip 0.25 | 110.35 | 0.064 | 109.98 | worse |
| peak LR 7e-3 | 109.89 | 0.064 | 109.98 | **keep** |
| peak LR 1e-2 | 110.18 | 0.064 | 109.89 | tie |
| WD 0.3 | 140.81 | 0.056 | 141.50 | keep, later reverted to 0.1 (tie) |
| WD 1.0 | 143.35 | 0.051 | 140.81 | worse |
| Neuron params LR × 10 | 142.65 | 0.054 | 140.26 | worse |
| Label smoothing 0.1 | 113.18 | 0.074 | 110.99 | worse |
| Dropout 0.1 on neuron input | 110.82 | 0.066 | 109.89 | worse |
| EMA of weights 0.99 for eval | 110.61 | 0.065 | 109.89 | worse |

Peak LR above 5e-3 needs the long warmup. With 5 % warmup, 6e-3 diverged and the out_proj rate jumped.

### 5.3 Neuron dynamics

| Run | val_ppl | rate | Ref. | Verdict |
|---|---|---|---|---|
| v_th 0.5 | 163.35 | 0.156 | 155.55 | worse |
| v_th 0.75 (tuned) | 116.98 | 0.148 | 111.10 | worse |
| v_th 1.5 | 145.66 | 0.033 | 140.26 | worse |
| decay_input False | 148.94 | 0.132 | 140.26 | worse |
| init tau 1.5 | 115.88 | 0.129 | 111.10 | worse |
| init tau 3.0 | 111.79 | 0.079 | 111.10 | noise, not kept |
| out_proj init tau 2.5 | 108.95 | 0.073 | 108.43 | worse |
| soft reset | 143.71 / 111.58 | 0.054 / 0.066 | | worse both times |
| detach_reset False | 110.14 | 0.066 | 109.98 | tie |
| ATan alpha 1.0 | 139.50 | 0.068 | 140.26 | keep, later back to default |
| ATan alpha 0.5 | 142.17 | 0.094 | 139.50 | worse |
| ATan alpha 1.5 → default 2.0 | 110.89 → 110.99 | | | tie, default kept |
| Sigmoid surrogate (alpha 4) | 109.16 | 0.079 | 108.43 | worse |

### 5.4 Time steps T

| Run | val_ppl | rate | Ref. | Verdict |
|---|---|---|---|---|
| T=8 | 146.16 | 0.069 | 141.50 | worse |
| T=2 | 142.46 / 113.85 | 0.045 / 0.083 | 141.50 / 112.22 | worse at tuned LR |
| T=6 | 109.72 | 0.074 | 109.89 | noise, 1.7× spikes per token |
| T=3 (with gain) | 109.75 | 0.068 | 108.43 | worse |

### 5.5 Input scaling in front of the neurons

| Run | val_ppl | rate | Ref. | Verdict |
|---|---|---|---|---|
| Learnable gain before out_proj, init 1.0 | 109.38 | 0.065 | 109.89 | **keep** |
| Same, init 2.0 | 108.99 | 0.084 | 109.38 | **keep** |
| Same, init 1.5 | **108.43** | 0.076 | 108.99 | **keep (best)** |
| Same, init 3.0 | 114.72 | 0.098 | 108.43 | worse |
| Fixed v_th 0.5 at out_proj (no gain) | 113.48 | 0.111 | 108.99 | worse |
| RMSNorm before out_proj neuron | 112.34 | 0.085 | 111.10 | worse |
| Per-channel shift (bias) | 108.52 | 0.078 | 108.43 | tie |
| in_proj gain init 1.25 | 114.42 | 0.104 | 108.43 | worse |
| in_proj gain init 0.85 | 109.88 | 0.055 | 108.43 | worse |

### 5.6 Idea 5: selective threshold / gate

| Run | val_ppl | rate | Ref. | Verdict |
|---|---|---|---|---|
| Softplus input scale, rank 32, both sites | 140.26 | 0.056 | 140.81 | keep at LR 1e-3 |
| Removed at tuned LR (ablation) | 112.26 vs 112.20 | | | tie, removed |
| Sigmoid gate, rank 32, both sites | 109.70 | 0.059 | 109.89 | noise, not kept |
| Sigmoid gate at out_proj only | 109.39 | 0.070 | 108.43 | worse |

The early −0.5 ppl gain at LR 1e-3 disappeared once the schedule was tuned.

### 5.7 ON/OFF neuron pair (probe, needs your ruling)

| Run | val_ppl | rate | Ref. |
|---|---|---|---|
| Two stock PLIFs per out_proj channel, on +x and −x; projection gets W·(r₊ − r₋) | 108.35 | 0.092 | 109.89 |

- −1.5 ppl against the best at that time, before the gain existed. Not yet tried together with the gain.
- 2.9× more spikes per channel.
- Logged as discard: the projection effectively sees signed values, and the rule says every value is in {0, …, 1}.
- Patch: `/tmp/claude-1000/-home-karthik-Desktop-mamba/59c5aa3b-b855-481c-bf80-475fbab9db93/scratchpad/probe-on-off-pair.patch`. It is in a temp folder, so copy it out if you want to keep it.

## 6. Analysis

**Recipe beats neuron changes.** The 1e-3 schedule from the dense config was far too weak for the spiking model. The model takes a 7× higher peak LR once warmup is long. My guess (not tested): the surrogate gradient shrinks the gradients, so a larger step is needed. All neuron-level tweaks together gave about 2 ppl.

**Dead out_proj layers.** The out_proj neuron sees the raw gated SSM output y·silu(z), which has no norm. With gain 1, the out_proj neurons of layers 0 and 1 never fired, so those two mixers sent nothing on. A learnable per-channel gain revives them. Init 1.5 is best: 3.0 fires too much and is worse by 6 ppl.

**More spikes usually hurt.** Every change that raised the rate a lot (v_th 0.5/0.75, tau 1.5, decay_input False, in_proj gain 1.25, out_proj gain 3.0) made val_ppl worse. The best region is a rate of about 0.06 to 0.08. The only rate increase that helped was waking up the dead layers.

**Half the out_proj signal is lost.** The out_proj input is mostly negative, and a PLIF never fires on negative input. The ON/OFF probe recovers it and gains 1.5 ppl. This is the largest open lever, but it needs the rule decision in 5.7.

**Firing rates (best model, average over 6 layers):** in_proj 6.5 %, out_proj 8.2 %, overall 7.6 % (weighted by neuron count: 384 vs 768 per layer). Rates per layer and per neuron are not measured. The saved checkpoint `runs/autoresearch/latest.pt` comes from the last discarded run, not from `ab9fa2c`.

## 7. Efficiency (est.)

MACs per token, from the checkpoint shapes:

| Part | Spiking? | MACs per token | Share |
|---|---|---|---|
| `lm_head` 384 → 128,256 | no | 49.25 M | 81 % |
| `in_proj` 384 → 1860, × 6 | yes | 4.29 M | 7 % |
| `out_proj` 768 → 384, × 6 | yes | 1.77 M | 3 % |
| MLP fc1 + fc2, × 6 | no | 5.31 M | 9 % |
| Total (SSM scan ignored) | | 60.6 M | |

Energy, using the usual 45 nm numbers (MAC 4.6 pJ, AC 0.9 pJ, Horowitz 2014). Spiking ops = MACs × rate × T:

| | Dense Mamba-3 | Spiking (best) |
|---|---|---|
| in_proj + out_proj | 27.9 µJ | 1.5 µJ (1.70 M ACs) |
| Rest (lm_head + MLP) | 251.0 µJ | 251.0 µJ |
| **Total per token** | **278.8 µJ** | **252.5 µJ (−9 %)** |

- The spiking layers save about 94 %. The whole model saves only about 9 %, because the dense `lm_head` does 81 % of the work.
- These numbers ignore memory access, neuron update cost and the SSM scan. The saving exists only on hardware that skips zeros.
- On a GPU, the spiking model costs more: it runs a full dense matmul on spike counts, plus 4 PLIF steps per token.
- Largest levers for real efficiency: a smaller vocab or a cheaper output head, then spiking the MLP.

## 8. Limits

1. **Dense comparison is unfair.** The teacher used LR 1e-3, warmup 5 % and no KD. Most of the student's gain came from the schedule. Needed: dense with the tuned schedule, with and without KD.
2. **One seed per run.** Noise is about ±0.5 to 1 ppl. The last three kept steps (109.4 → 109.0 → 108.4) are each near noise. Only the overall trend is solid.
3. **Small setting.** 740 steps, 12M tokens, WikiText-2 validation split. Rankings may change with longer training or more data. Spikes may act as a regularizer that helps only when data is small.
4. **Greedy search.** One change at a time, kept when better. Interactions (for example T and the gain, or KD and the LR) were only partly tested.
5. **Teacher ppl not re-measured.** 142.9 comes from `program.md`.
6. **Lint not run** (ruff, mypy) on the final `spiking_mamba3.py`.

## 9. Next steps (my order)

1. Rule on the ON/OFF pair. If allowed, test it together with the out_proj gain.
2. Dense control runs: tuned schedule, with and without KD.
3. Rerun `ab9fa2c` with 2 more seeds and save per-layer and per-neuron firing histograms.
4. Re-evaluate the teacher checkpoint with `eval.py`.
5. Ideas not yet tried: higher LR for the gain parameters only, KD 0.6, warmup 0.4, out_proj gain init 1.25.

## 10. Housekeeping

- `program.md` (your uncommitted edits) is commit `2de3d1f` on the branch, so `git reset --hard` in the loop could not erase it.
- Only `spiking_mamba3.py` changed in the experiments. `master` is untouched. Nothing was pushed.
- `results.tsv` and this report are untracked.
