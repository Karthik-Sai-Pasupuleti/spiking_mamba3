# spiking-mamba3

All Mamba-3 files of the official Mamba repository (SISO and MIMO kernels), plus a language
model and training code. SISO runs on Triton. MIMO runs on TileLang and `Mamba3.step()` on CuTe;
both need the optional extra `mimo` (see Setup).

On the RTX 5090 the upstream TileLang MIMO backward kernel returns NaN or garbage gradients
(reproduced with the unchanged upstream code), so the configs here use SISO.

## Layout

```
configs/       one YAML per experiment, and config.py (loads it; --smoke shrinks it)
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
spiking_mamba3.py  spiking Mamba-3 (PLIF before in_proj/out_proj); the file the autoresearch agent edits
program.md     instructions for the autoresearch agent
notebooks/     dataloader-end-to-end.ipynb: the data pipeline step by step
scripts/       prepare_wikitext2.py: WikiText-2 parquet -> token .bin files
licenses/      LICENSE-mamba, AUTHORS-mamba (upstream Mamba), LICENSE-flash-attention (data code)
reports/       results of autoresearch runs (autoresearch-nov6/: report, results.tsv, run.log)
runs/          outputs, git-ignored
```

## Upstream files

- Repository: https://github.com/state-spaces/mamba
- Commit: `e9594ce1c732d97440f0332fdc43170a2294dbfa`
- License: Apache-2.0, see `licenses/LICENSE-mamba`. Authors in `licenses/AUTHORS-mamba`.
- Our only change: the import lines and the file paths. `mamba_ssm.ops.` became
  `models.kernels.`, `mamba_ssm.modules.` became `models.layers.`, and `mamba_ssm.utils.` became `utils.`. The check at the end of this file proves nothing else changed.

| File | Role |
|---|---|
| `models/layers/mamba3.py` | the `Mamba3` layer |
| `models/kernels/triton/mamba3/mamba3_siso_combined.py`, `_fwd.py`, `_bwd.py` | SISO forward and backward |
| `models/kernels/triton/mamba3/mamba3_siso_step.py` | SISO one-token decode (no gradients) |
| `models/kernels/triton/mamba3/angle_dt.py`, `utils.py` | rotary angles, math helpers |
| `models/kernels/triton/mamba3/mamba3_mimo_rotary_step.py` | rotary step; `mamba3.py` imports it at the top |
| `models/kernels/triton/layernorm_gated.py` | RMSNorm on B and C |
| `models/layers/block.py`, `models/layers/mlp.py` | upstream `Block` and `GatedMLP`, used by the LM |
| `models/kernels/triton/layer_norm.py`, `utils/determinism.py`, `utils/torch.py` | fused residual-add + RMSNorm of `Block` |
| `models/kernels/tilelang/mamba3/mamba3_mimo.py`, `_fwd.py`, `_bwd.py`, `_fwd_varlen.py`, `_bwd_varlen.py` | MIMO forward and backward (TileLang) |
| `models/kernels/triton/mamba3/mamba3_mimo_utils.py`, `grouped_head_reduction.py` | Triton helpers of the MIMO kernels |
| `models/kernels/cute/mamba3/mamba3_step_fn.py` | one-token decode for `Mamba3.step()` (CuTe) |

Not copied: Mamba-1, Mamba-2, attention, upstream's `models/` and `__init__.py`, and the tests.
Import the layer as `from models.layers.mamba3 import Mamba3`.

Without the `mimo` extra, `is_mimo=True` and `Mamba3.step()` fail with an assert. For one-token
SISO decoding without it, call `mamba3_siso_step` directly with the four saved states.

## Data loading (from FlashAttention)

The Mamba repository has no data or training code; its authors point to the training code of
FlashAttention (state-spaces/mamba issue #1). Two classes are copied unchanged from
Dao-AILab/flash-attention at commit `34a06b9` (BSD-3-Clause, see `licenses/LICENSE-flash-attention`):

| Here | FlashAttention file |
|---|---|
| `dataset/dataset.py`: `LMDataset` | `training/src/datamodules/datasets/lm_dataset.py` |
| `dataset/dataloader.py`: `RandomFaultTolerantSampler`, `FaultTolerantDistributedSampler` | `training/src/datamodules/fault_tolerant_sampler.py` |

Not copied: `LMDataModule` (`language_modeling_hf.py`), which needs PyTorch Lightning,
`transformers` and `datasets` and tokenizes with HF on first use. Here the data is already
tokenized: `data.train_bin` and `data.val_bin` are flat `uint32` token files (FineWeb-Edu,
Llama-3.1 tokenizer).

`LMDataset` cuts the token file into samples of `seq_len` inputs and targets; neighbouring
samples share one token. Training uses `FaultTolerantDistributedSampler` on one or several
GPUs. On resume, the sampler skips the samples already trained on, so a resumed run sees the
same samples as a run without a break.

## Setup

```bash
uv sync --extra dev
uv sync --extra dev --extra mimo   # also TileLang MIMO and the CuTe step kernel
```

## Train

```bash
.venv/bin/python train.py --config configs/mamba3-siso-187m-small.yaml --smoke  # 20-step check
.venv/bin/python train.py --config configs/mamba3-siso-187m-small.yaml
```

Each run writes `runs/<run_name>-<timestamp>/`: `config.json` (config, seed, environment, step
counts), `train.log`, `metrics.csv`, `last.pt` and `final.pt`. The run stops on the first
non-finite gradient and logs which parameters have it.

Several GPUs on one node: `torchrun --standalone --nproc_per_node=N train.py ...`.
`batch_size` must be a multiple of `micro_batch_size x N`. With `--run-dir runs/<name>` the run
uses one fixed folder and resumes from its `last.pt` automatically.

## Spiking Mamba-3-SISO-180M on FineWeb-Edu 100B (cluster)

Spiking 187M model (`configs/spiking-mamba3-siso-187m-fineweb100b.yaml`): PLIF neurons before
`in_proj` and `out_proj` (`models/layers/spiking.py`), started from the pretrained
`state-spaces/mamba3-siso-187m` weights (`train.init_from`), trained on FineWeb-Edu sample-100BT
(139 files, ~102B Llama-3.1 tokens, 2K context, ~0.5M tokens/step, ~193K steps; est.).
`configs/mamba3-siso-187m-fineweb100b.yaml` is the dense reference with the same data.

On the cluster, from the repo root:

```bash
bash scripts/cluster/setup.sh                      # once: uv, .venv, runs/slurm/, GPU check
# copy the pretrained weights to weights/mamba3-siso-187m/pytorch_model.bin (setup.sh shows how)
sbatch scripts/cluster/prepare_data.sbatch         # once: download + tokenize (~400 GB); resumable
.venv/bin/python eval.py --config configs/mamba3-siso-187m-fineweb100b.yaml \
    --checkpoint weights/mamba3-siso-187m/pytorch_model.bin          # dense reference ppl
sbatch scripts/cluster/train.sbatch configs/spiking-mamba3-siso-187m-fineweb100b.yaml
```

Fill in the `TODO` lines (CPU partition, time limits) in the two `.sbatch` files first. Each
training job resumes from `runs/<config name>/last.pt`; `train.sbatch` shows how to queue jobs
in a row. `metrics.csv` has a `spike_rate` column (validation firing rate).

**Known issue (2026-11-07):** with the nov6 neuron settings (threshold 1.0, tau 2.0, T=4),
the pretrained model's activations are too small to fire: in a 40-step local test the rate was
0.0002 and val loss rose from 2.75 (dense, ppl 15.65) to 6.5. Calibrate the neurons before the
full run.

## Evaluate and generate

```bash
.venv/bin/python eval.py --config configs/<name>.yaml --checkpoint runs/<run>/final.pt
.venv/bin/python test.py --config configs/<name>.yaml --checkpoint runs/<run>/final.pt \
    --tokenizer <tokenizer.json> --prompt "The capital of France is"
```

## Check that the upstream files are unchanged

Each file is compared with upstream after renaming the imports back to the `mamba_ssm` paths. No
`DIFFERS` line means only the import names changed. The count must be 24 (the empty `__init__.py` files are not compared).

```bash
UP=/home/karthik/Desktop/mamba   # a clone of state-spaces/mamba
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
  echo "licenses/LICENSE-mamba LICENSE"; echo "licenses/AUTHORS-mamba AUTHORS"
  echo "models/layers/mamba3.py mamba_ssm/modules/mamba3.py"
  echo "models/layers/block.py mamba_ssm/modules/block.py"
  echo "models/layers/mlp.py mamba_ssm/modules/mlp.py"
  echo "utils/determinism.py mamba_ssm/utils/determinism.py"
  echo "utils/torch.py mamba_ssm/utils/torch.py"
  (cd "$P/models/kernels" && find triton tilelang cute -name '*.py' ! -name __init__.py ! -path '*/__pycache__/*') \
    | while read -r f; do echo "models/kernels/$f mamba_ssm/ops/$f"; done
)
echo "checked $n files"
```
