#!/usr/bin/env bash
# Download FineWeb-Edu sample-100BT (140 parquet files, 286 GB) and tokenize it into
# data-fineweb-100bt/{train,val}/*.bin (~400 GB). CPU only. Submit from the repo root:
#   sbatch scripts/cluster/prepare_data.sbatch
# The script resumes: if the job hits its time limit, submit it again; finished files are skipped.
# Measured on the workstation: ~30 CPU-minutes per file, ~69 CPU-hours for all 140 (est. 1-2 h on
# 32-64 cores, plus download time). Peak memory ~5 GB per worker.
#SBATCH --job-name=prep-fineweb100b
#SBATCH --partition=TODO               # a CPU partition: check `sinfo`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=TODO                    # e.g. 12:00:00; check `sinfo -p <partition> -o "%l"`
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.venv/bin/python -m scripts.prepare_fineweb --out data-fineweb-100bt --workers 8 --delete-parquet
ls data-fineweb-100bt/train | wc -l
ls data-fineweb-100bt/val
