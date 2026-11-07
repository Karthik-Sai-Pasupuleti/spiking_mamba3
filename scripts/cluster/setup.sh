#!/usr/bin/env bash
# One-time setup on the cluster, run on the login node from anywhere inside the repo:
#   bash scripts/cluster/setup.sh
# Installs uv if missing, builds .venv from uv.lock, creates runs/slurm/ for job logs, and
# checks that the pretrained checkpoint is in place. The GPU check runs in a short srun job.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra dev --frozen
mkdir -p runs/slurm

.venv/bin/python -c "import torch, triton, spikingjelly; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| triton', triton.__version__)"

CKPT=weights/mamba3-siso-187m/pytorch_model.bin
if [ -f "$CKPT" ]; then
    echo "pretrained checkpoint found: $CKPT"
else
    echo "MISSING: $CKPT"
    echo "  copy it from your workstation, e.g.:"
    echo "  rsync -av /home/karthik/Desktop/mamba/weights/mamba3-siso-187m/pytorch_model.bin <cluster>:$(pwd)/$CKPT"
fi

# torch here is built for CUDA 13.0, which needs a recent NVIDIA driver on the GPU nodes.
srun -p gpu --gres=gpu:nvidia_a100-pcie-40gb:1 --time=00:05:00 \
    .venv/bin/python -c "import torch; print('GPU visible:', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')" \
    || echo "GPU check failed: see the srun error above (driver too old for CUDA 13.0, or no free GPU)"
