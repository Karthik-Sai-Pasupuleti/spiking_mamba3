#!/usr/bin/env bash
# Train one config on all GPUs of one node (DDP via torchrun). Submit from the repo root:
#   sbatch scripts/cluster/train.sbatch configs/spiking-mamba3-siso-187m-fineweb100b.yaml
# The run folder is runs/<config name>/. A job that hits its time limit loses at most
# `save_every` steps: submit the same command again and it resumes from last.pt. Once the run
# has finished (final.pt exists), a new job stops at once. To queue several jobs in a row:
#   jid=""; for i in $(seq 1 12); do
#     jid=$(sbatch --parsable ${jid:+--dependency=afterany:$jid} scripts/cluster/train.sbatch <config>)
#   done
# batch_size in the config must be a multiple of micro_batch_size x GPUs (258 = 2 x 3 x 43).
#SBATCH --job-name=spiking-187m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=TODO                    # the partition's limit: check `sinfo -p gpu -o "%l"`
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/cluster/train.sbatch configs/<name>.yaml [train.py args]}"
shift
cd "$SLURM_SUBMIT_DIR"
NAME="$(basename "$CONFIG" .yaml)"
NGPU="$(nvidia-smi -L | wc -l)"
echo "job $SLURM_JOB_ID on $(hostname): $NGPU GPUs, config $CONFIG, run dir runs/$NAME"
.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$CONFIG" --run-dir "runs/$NAME" "$@"
