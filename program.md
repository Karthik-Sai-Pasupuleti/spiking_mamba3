# Spiking Mamba3

This is an experiment in autonomous research: you, the agent, convert Mamba-3 SISO into a
spiking model and improve it by yourself, one experiment at a time, while the human is away.

## The idea

1. **Neuron placement.** Following the SpikingMamba literature (arXiv 2510.04595), spiking
   neurons sit in front of `in_proj` and `out_proj` of every Mamba-3 block. The architecture
   stays intact, but both projections receive sparse spikes instead of dense activations.
2. **PLIF, multi-step.** The neuron is SpikingJelly's PLIF,
   `spikingjelly.activation_based.neuron.ParametricLIFNode` with `step_mode="m"` (parametric
   LIF with a learnable time constant, all T steps in one call on a `[T, N, D]` input). Each
   token's activation is fed to the neuron for T steps; the membrane integrates over the T
   steps and the projection receives the spike count / T. The network is not repeated over
   the T steps: the model already works in the token domain, and the SSM hidden state carries
   the memory along the sequence. The neuron state is reset for every token.
3. **Validate with perplexity.** A spiking model trained with T = k steps is judged by its
   validation perplexity on WikiText-2, under a cap on its firing rate (see "The goal").
4. **Knowledge distillation.** A frozen dense Mamba-3 61M teacher is available for logit
   distillation (`KD_WEIGHT`, `KD_TEMPERATURE` in `spiking_mamba3.py`; off in the baseline).
5. **Selective decay and threshold.** Make the effective threshold and/or decay depend on
   the token (computed from the neuron's input, like Mamba's input-dependent Δ), so that the
   neuron passes little to the Mamba block for filler tokens and acts as one more layer that
   selects the key features of the data. The stock PLIF has one learnable tau per neuron layer
   and one fixed threshold, so build this *around* it, on its input: scaling the input by
   1 / θ(token) is the same as a per-token threshold θ(token), and a per-token gate g(token) in
   [0, 1] on the input silences filler tokens.

## Setup

Do this once, together with the human, before the loop starts:

1. **Agree on a run tag** from today's date (e.g. `nov6`). The branch `autoresearch/<tag>` must
   not exist yet.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `master`.
3. **Read the in-scope files** for full context:
   - `program.md`: these instructions.
   - `spiking_mamba3.py`: the file you edit (spiking layer, training loop, summary).
   - `models/layers/mamba3.py`: the upstream Mamba-3 layer that `SpikingMamba3` subclasses. Read only.
   - `train.py`, `eval.py`, `dataset/dataset.py`, `dataset/dataloader.py`: optimizer, LR schedule,
     the fixed evaluation and the data pipeline that `spiking_mamba3.py` imports. Read only.
   - `configs/mamba3-siso-61m-wikitext2.yaml`: the fixed model size, data, batch and step budget.
     Read only.
4. **Verify data and teacher exist**: `data-wikitext2/train.bin`, `data-wikitext2/validation.bin`
   and `runs/teacher-dense-61m-wikitext2/final.pt`. If any is missing, stop and tell the human
   (data: run `prepare_wikitext2.py`; teacher: train the 61M config into that folder).
5. **Initialize `results.tsv`** with only its header row (see "Logging results").
6. **Confirm and go.**

## Experimentation

Each experiment trains the spiking 61M model for a **fixed budget of 740 steps** (5 epochs of
WikiText-2, 16 samples x 1024 tokens per step, 12.1M tokens) on one GPU, then evaluates it on
the full validation split. The baseline takes about 2.5 minutes. Launch it as:

`.venv/bin/python spiking_mamba3.py > run.log 2>&1`

For a 20-step check that new code runs at all, add `--smoke` (about 30 s; its numbers mean nothing).

**What you CAN do** (only in `spiking_mamba3.py`):
- Change T (1-8) and the arguments of the `ParametricLIFNode`: `init_tau`, `v_threshold`,
  `v_reset` (0.0 = hard reset, None = soft reset), `decay_input`, `detach_reset` and
  `surrogate_function` (any of `spikingjelly.activation_based.surrogate`, with its `alpha`).
- Change what feeds the neuron: a norm, a learnable gain, or a token-dependent scale or gate
  computed from the neuron's input (idea 5).
- Change the training: the LR, schedule, warmup, weight decay, gradient clipping, the training
  loss, and knowledge distillation with the fixed teacher.
- Add new classes or functions to the file.

**What you CANNOT do:**
- Edit any file except `spiking_mamba3.py`. In particular, never edit the upstream Mamba-3
  files (`models/layers/`, `models/kernels/`, `utils/determinism.py`, `utils/torch.py`), the
  config, `train.py`, `eval.py` or `dataset/`. They define the model and the measurement.
- Change the fixed parts of `spiking_mamba3.py`: the "Fixed" constants (`BASE_CONFIG`,
  `TEACHER_CHECKPOINT`, `RATE_CAP`, `TOTAL_STEPS`, `OUTPUT_CHECKPOINT`), the function
  `firing_rate_and_val_loss`, and the summary block that main() prints. Do not override the
  model size, data, seq_len, batch size, step count or validation set from the config.
- Replace the neuron. Every neuron is a SpikingJelly `ParametricLIFNode` with
  `step_mode="m"`, called once on the `[T, N, D]` input. Do not write your own neuron,
  subclass it, or change its dynamics or its `backend`.
- Remove, bypass or move the neurons: they stay in front of `in_proj` and `out_proj`, and no
  dense (non-spike) path may reach those projections. The neuron output stays a spike count / T
  of binary spikes, so every value is in {0, 1/T, ..., 1}.
- Install packages or add dependencies. Use what is in `pyproject.toml` (torch, spikingjelly, ...).

**The goal: get the lowest `val_ppl` with `spike_rate` at most 0.20.** A run with
`rate_ok: no` never counts as an improvement, however low its perplexity. Because the step
budget is fixed, slower code is allowed, as long as a run finishes within the timeout below.

Reference points (same data, steps and evaluation):

| Model | val_ppl | spike_rate |
|---|---|---|
| Dense Mamba-3 SISO 61M (the teacher) | 142.9 | — |
| Spiking baseline: T=4, tau 2.0, v_th 1.0, no KD | 154.5 | 0.056 |

**VRAM** is a soft limit. The GPU has 32 GB; the baseline uses 8.6 GB, distillation about
20 GB (the student and teacher logits are each 4 x 1024 x 128,256 floats). Stay well below 32 GB.

**Simplicity criterion.** All else equal, simpler is better. A tiny gain that adds ugly code
is not worth it; an equal result with less code is a win. Weigh the complexity of a change
against the size of the gain.

**The first run** is always the unmodified file, to record the baseline on this machine.

## Output format

At the end, `spiking_mamba3.py` prints a summary block:

```
---
val_ppl:            154.462071
val_loss:           5.039949
spike_rate:         0.056074
rate_ok:            yes
rate_in_proj:       0.080212
rate_out_proj:      0.044005
T:                  4
num_steps:          740
training_seconds:   137.9
total_seconds:      139.9
peak_vram_mb:       8637.0
```

`spike_rate` is the total spikes divided by the total neuron-steps of all PLIF neurons over
the whole validation split. Read the key lines with:

`grep "^val_ppl:\|^spike_rate:\|^rate_ok:\|^peak_vram_mb:" run.log`

## Logging results

Log every experiment to `results.tsv`, tab-separated (not comma-separated: commas break
descriptions). Never commit it; it stays untracked. Header and 6 columns:

```
commit	val_ppl	spike_rate	memory_gb	status	description
```

1. short git commit hash (7 characters)
2. val_ppl (e.g. 154.462071); 0.000000 for crashes
3. spike_rate (e.g. 0.056074); 0.000000 for crashes
4. peak memory in GB, one decimal (peak_vram_mb / 1024); 0.0 for crashes
5. status: `keep`, `discard`, `crash` or `over_cap` (rate_ok: no)
6. a short description of what the experiment tried

Example:

```
commit	val_ppl	spike_rate	memory_gb	status	description
a1b2c3d	154.462071	0.056074	8.4	keep	baseline T=4 tau 2.0 v_th 1.0
b2c3d4e	151.200000	0.071000	8.4	keep	v_th 0.8
c3d4e5f	149.900000	0.243000	8.4	over_cap	v_th 0.5
d4e5f6a	0.000000	0.000000	0.0	crash	token-dependent tau (shape bug)
```

## The experiment loop

The loop runs on the branch `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Change `spiking_mamba3.py` to test one idea.
3. `git commit -am "<short description>"`.
4. Run: `.venv/bin/python spiking_mamba3.py > run.log 2>&1`. Redirect everything; do not use
   tee and do not print training logs into your context.
5. Read the result: `grep "^val_ppl:\|^spike_rate:\|^rate_ok:\|^peak_vram_mb:" run.log`.
6. If the grep is empty, the run crashed: `tail -n 50 run.log` shows the error. Fix it if it is
   a simple bug, and re-run; give up on the idea after a few attempts.
7. Add a row to `results.tsv`.
8. If `rate_ok` is `yes` and `val_ppl` is lower than the best kept result, keep the commit
   (status `keep`): the branch advances.
9. Otherwise, `git reset --hard HEAD~1` back to the last kept commit (status `discard`, `crash`
   or `over_cap`).

You are a fully autonomous researcher: if an idea works, keep it; if it does not, discard it.
Rewind further back than one commit only rarely, if ever.

**Timeout.** An experiment takes about 2.5 minutes at T=4 and should stay under 6 minutes at
T=8. If a run takes more than 10 minutes, kill it and treat it as a failure.

**Crashes.** Out of memory, a bug, a NaN gradient (the loop stops with "non-finite gradient"):
fix simple mistakes (a typo, a shape, a missing import) and re-run. If the idea itself is
broken, log `crash` and move on.

**NEVER STOP.** Once the loop has begun, do not pause to ask the human whether to continue, and
do not ask "is this a good stopping point?". The human may be asleep and expects you to work
until they stop you. If you run out of ideas, think harder: re-read `spiking_mamba3.py` and
`models/layers/mamba3.py` for new angles, re-read "The idea" above, combine earlier near-misses,
or try a more radical neuron design. The loop runs until the human interrupts you, period.

At about 2.5 minutes per experiment you can run about 20 per hour, about 150 in a night.

## Ideas to start from

Not a plan, only a cold start. Use your judgement.

- T: does more integration (T = 6, 8) buy perplexity, or is T = 2 almost as good?
- Threshold and tau: the baseline rate is only 5.6%; a lower `v_threshold` or a scaled input lets
  more information through while staying under the 20% cap.
- A norm or a learnable gain in front of each neuron, so its input scale fits the threshold.
- Distillation: `KD_WEIGHT` 0.3-0.7, `KD_TEMPERATURE` 1-2.
- Selective threshold or gate, idea 5: a small projection of the neuron input, through a
  softplus (scale 1/θ) or sigmoid (gate g), per token and channel, applied before the PLIF.
- Surrogate gradient: `surrogate.ATan(alpha=...)`, `surrogate.Sigmoid(alpha=...)`, and others in
  `spikingjelly.activation_based.surrogate`.
- Soft reset (`v_reset=None`, subtract the threshold) instead of hard reset to 0.
- `decay_input=False` (the input is added to the membrane without the 1/tau factor).
- A higher LR or a longer warmup for the neuron parameters only.
