# spiking-mamba3

All Mamba-3 files of the official Mamba repository (SISO and MIMO kernels), plus a language
model and training code. SISO runs on Triton. MIMO runs on TileLang and `Mamba3.step()` on CuTe;
both need the optional extra `mimo` (see Setup).

On the RTX 5090 the upstream TileLang MIMO backward kernel returns NaN or garbage gradients
(reproduced with the unchanged upstream code), so the configs here use SISO.

## Layout

```
configs/         mamba3-siso-61m-wikitext103-spiking.yaml: the training config
data/            all datasets, git-ignored
  wikitext103/   raw/ (parquet from Hugging Face) and train.bin, validation.bin, test.bin
                 (uint32 tokens, made by src/prepare_wikitext103.py)
src/             all code; run every command from the repo root
  configs/       config.py: loads a YAML from configs/ (--smoke shrinks it)
  dataset/       dataset.py: LMDataset (from FlashAttention) over a memory-mapped .bin file
                 dataloader.py: fault-tolerant sampler (from FlashAttention), DataLoader, step counts
                 tokenizer.py: text <-> token ids
  models/
    model.py     MambaLMHeadModel: backbone + tied output head
    backbone.py  embedding, stack of blocks, final norm
    layers/      mamba3.py (the Mamba-3 mixer), block.py, mlp.py (upstream)
    kernels/     upstream kernels: triton/ (SISO, norms, MIMO helpers), tilelang/ (MIMO),
                 cute/ (one-token step)
  utils/         checkpoint.py, run_files.py, distributed.py
                 determinism.py, torch.py (upstream)
  train.py       train loop and the `train` command
  eval.py        validation loss and the `eval` command
  test.py        text generation from a checkpoint
  spiking_mamba3.py  spiking Mamba-3 layer (PLIF before in_proj and out_proj)
  prepare_wikitext103.py  raw WikiText-103 parquet -> token .bin files
LICENSE-mamba, AUTHORS-mamba   license and authors of the upstream Mamba files
LICENSE-flash-attention        license of the FlashAttention data code
runs/            outputs, git-ignored
```

`src/` is on `sys.path` in two ways: Python adds the folder of the script it runs, and the
editable install (`uv sync`) adds it for the notebooks. So the code imports `models`,
`dataset`, `utils` and `configs` without a `src.` prefix.

## Upstream files

- Repository: https://github.com/state-spaces/mamba
- Commit: `e9594ce1c732d97440f0332fdc43170a2294dbfa`
- License: Apache-2.0, see `LICENSE-mamba`. Authors in `AUTHORS-mamba`.
- Our only change: the import lines and the file paths. `mamba_ssm.ops.` became
  `models.kernels.`, `mamba_ssm.modules.` became `models.layers.`, and `mamba_ssm.utils.` became `utils.`. The check at the end of this file proves nothing else changed.

| File | Role |
|---|---|
| `src/models/layers/mamba3.py` | the `Mamba3` layer |
| `src/models/kernels/triton/mamba3/mamba3_siso_combined.py`, `_fwd.py`, `_bwd.py` | SISO forward and backward |
| `src/models/kernels/triton/mamba3/mamba3_siso_step.py` | SISO one-token decode (no gradients) |
| `src/models/kernels/triton/mamba3/angle_dt.py`, `utils.py` | rotary angles, math helpers |
| `src/models/kernels/triton/mamba3/mamba3_mimo_rotary_step.py` | rotary step; `mamba3.py` imports it at the top |
| `src/models/kernels/triton/layernorm_gated.py` | RMSNorm on B and C |
| `src/models/layers/block.py`, `src/models/layers/mlp.py` | upstream `Block` and `GatedMLP`, used by the LM |
| `src/models/kernels/triton/layer_norm.py`, `src/utils/determinism.py`, `src/utils/torch.py` | fused residual-add + RMSNorm of `Block` |
| `src/models/kernels/tilelang/mamba3/mamba3_mimo.py`, `_fwd.py`, `_bwd.py`, `_fwd_varlen.py`, `_bwd_varlen.py` | MIMO forward and backward (TileLang) |
| `src/models/kernels/triton/mamba3/mamba3_mimo_utils.py`, `grouped_head_reduction.py` | Triton helpers of the MIMO kernels |
| `src/models/kernels/cute/mamba3/mamba3_step_fn.py` | one-token decode for `Mamba3.step()` (CuTe) |

Not copied: Mamba-1, Mamba-2, attention, upstream's `models/` and `__init__.py`, and the tests.
Import the layer as `from models.layers.mamba3 import Mamba3`.

Without the `mimo` extra, `is_mimo=True` and `Mamba3.step()` fail with an assert. For one-token
SISO decoding without it, call `mamba3_siso_step` directly with the four saved states.

## Data loading (from FlashAttention)

The Mamba repository has no data or training code; its authors point to the training code of
FlashAttention (state-spaces/mamba issue #1). Two classes are copied unchanged from
Dao-AILab/flash-attention at commit `34a06b9` (BSD-3-Clause, see `LICENSE-flash-attention`):

| Here | FlashAttention file |
|---|---|
| `src/dataset/dataset.py`: `LMDataset` | `training/src/datamodules/datasets/lm_dataset.py` |
| `src/dataset/dataloader.py`: `RandomFaultTolerantSampler`, `FaultTolerantDistributedSampler` | `training/src/datamodules/fault_tolerant_sampler.py` |

Not copied: `LMDataModule` (`language_modeling_hf.py`), which needs PyTorch Lightning,
`transformers` and `datasets` and tokenizes with HF on first use. Here the data is already
tokenized: `data.train_bin` and `data.val_bin` are flat `uint32` token files (WikiText-103,
Llama-3.1 tokenizer, made by `src/prepare_wikitext103.py`).

`LMDataset` cuts the token file into samples of `seq_len` inputs and targets; neighbouring
samples share one token. Training uses `FaultTolerantDistributedSampler` on one or several
GPUs. On resume, the sampler skips the samples already trained on, so a resumed run sees the
same samples as a run without a break.

## Setup

```bash
uv sync --extra dev
uv sync --extra dev --extra mimo   # also TileLang MIMO and the CuTe step kernel
```

## Prepare data

Put the parquet files of `wikitext-103-raw-v1` (Hugging Face `Salesforce/wikitext`) in
`data/wikitext103/raw/`, then tokenize them once (about 15 s, 480 MB):

```bash
.venv/bin/python src/prepare_wikitext103.py --tokenizer <Llama-3.1 tokenizer.json>
```

## Train

```bash
.venv/bin/python src/train.py --config configs/mamba3-siso-61m-wikitext103-spiking.yaml --smoke  # 20-step check
.venv/bin/python src/train.py --config configs/mamba3-siso-61m-wikitext103-spiking.yaml
```

Each run writes `runs/<run_name>-<timestamp>/`: `config.json` (config, seed, environment, step
counts), `train.log`, `metrics.csv`, `last.pt`, `final.pt` and `best.pt`. Every epoch ends
with a validation; `best.pt` holds the weights of the epoch with the lowest validation loss
so far (with its `step`, `epoch` and `val_loss`). The run stops on the first non-finite
gradient and logs which parameters have it. The terminal shows a progress bar per
epoch. For spiking models, every validation also measures the spike rate (`spike_rate`,
`spike_rate_in_proj`, `spike_rate_out_proj` in `metrics.csv`).

## Experiment tracking (MLflow)

`src/train.py` logs every run to MLflow (`src/utils/tracking.py`), one experiment per
`run_name`:

- params: every config value (`optim.peak_lr`, `model.ssm_cfg.plif_steps`, ...), step counts,
  parameter count;
- tags: git commit, branch and `git_dirty`, command line, GPU, torch, CUDA and Python versions;
- artifacts: the YAML config (`config/`), `config.json`, `train.log`, `metrics.csv` (`run/`),
  and for a dirty working tree a zip of `src/`, `configs/`, `pyproject.toml`, `uv.lock` as they
  were on disk (`code/source.zip`);
- metrics: everything in `metrics.csv`, per step.

To repeat a run: check out its `git_commit` (or unzip `code/source.zip` if `git_dirty` is
true) and run `src/train.py` with its config artifact. Commit before real runs, so the commit
alone identifies the code.

The store is `mlflow.db` and `mlartifacts/` in the repo root (git-ignored). Open the UI with

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

and go to http://localhost:5000. To log to another server instead, set `MLFLOW_TRACKING_URI`.

Several GPUs on one node: `torchrun --standalone --nproc_per_node=N src/train.py ...`.
`batch_size` must be a multiple of `micro_batch_size x N`. With `--run-dir runs/<name>` the run
uses one fixed folder and resumes from its `last.pt` automatically.

## Evaluate and generate

```bash
.venv/bin/python src/eval.py --config configs/<name>.yaml --checkpoint runs/<run>/final.pt
.venv/bin/python src/test.py --config configs/<name>.yaml --checkpoint runs/<run>/final.pt \
    --tokenizer <tokenizer.json> --prompt "The capital of France is"
```

## Check that the upstream files are unchanged

Each file is compared with upstream after renaming the imports back to the `mamba_ssm` paths. No
`DIFFERS` line means only the import names changed. The count must be 24 (the empty `__init__.py` files are not compared).

```bash
UP=<path to a fresh clone of github.com/state-spaces/mamba>
P=.
n=0
while read -r ours up; do
  sed -E 's/^(\s*)from models\.kernels\./\1from mamba_ssm.ops./;
          s/^(\s*)from models\.layers\.mamba3/\1from mamba_ssm.modules.mamba3/;
          s/^(\s*)from models\.layers\./\1from mamba_ssm.modules./;
          s/^(\s*)from utils\./\1from mamba_ssm.utils./' "$P/$ours" \
    | cmp -s - <(git -C "$UP" show "e9594ce:$up") || echo "DIFFERS: $ours"
  n=$((n + 1))
done < <(
  echo "LICENSE-mamba LICENSE"; echo "AUTHORS-mamba AUTHORS"
  echo "src/models/layers/mamba3.py mamba_ssm/modules/mamba3.py"
  echo "src/models/layers/block.py mamba_ssm/modules/block.py"
  echo "src/models/layers/mlp.py mamba_ssm/modules/mlp.py"
  echo "src/utils/determinism.py mamba_ssm/utils/determinism.py"
  echo "src/utils/torch.py mamba_ssm/utils/torch.py"
  (cd "$P/src/models/kernels" && find triton tilelang cute -name '*.py' ! -name __init__.py ! -path '*/__pycache__/*') \
    | while read -r f; do echo "src/models/kernels/$f mamba_ssm/ops/$f"; done
)
echo "checked $n files"
```
