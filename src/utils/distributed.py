"""Multi-GPU process setup for training with `torchrun` (one process per GPU).

`torchrun` starts one process per GPU and sets RANK, WORLD_SIZE and LOCAL_RANK in each
process's environment. `init_distributed` reads them, picks this process's GPU and joins
the NCCL process group. Without these variables (a plain `python train.py`), it
returns a single-process setup and starts no process group.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    """This process's place in the job: global rank, number of processes, local GPU."""

    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        """True for rank 0, the process that logs, validates and saves."""
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Join the NCCL process group if started by `torchrun`; else single process."""
    if "RANK" not in os.environ:
        return DistInfo(rank=0, world_size=1, local_rank=0)
    info = DistInfo(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
    )
    torch.cuda.set_device(info.local_rank)
    device = torch.device("cuda", info.local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return info


def close_distributed() -> None:
    """Leave the process group, if one was started."""
    if dist.is_initialized():
        dist.destroy_process_group()
