"""Batches for pretraining: FlashAttention's fault-tolerant samplers plus a torch DataLoader.

`RandomFaultTolerantSampler` and `FaultTolerantDistributedSampler` below are copied
unchanged from Dao-AILab/flash-attention, file
training/src/datamodules/fault_tolerant_sampler.py at commit 34a06b9 (BSD-3-Clause, see
LICENSE-flash-attention). They shuffle like torch's samplers, but `load_state_dict` can
skip the samples of an epoch that a run already trained on, so a restart continues
mid-epoch.

The rest is ours. Training always uses `FaultTolerantDistributedSampler`, also on one GPU
(`world_size == 1`), so there is one code path: the order of the samples depends only on
the seed and the epoch. `micro_batches` restores the sampler position from the step number,
as FlashAttention does (`fast_forward_batches * batch_size`), so a checkpoint only needs
the step.
"""

from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

# Adapted from https://github.com/Lightning-AI/lightning/blob/2845e7565dbe6b765ae32870e7d2bc456529c30a/tests/tests_pytorch/utilities/test_auto_restart.py#L1397
from typing import Iterator
import math

import torch
from torch.utils.data import RandomSampler, DistributedSampler


class RandomFaultTolerantSampler(RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # generator = torch.Generator().manual_seed(seed)
        # super().__init__(*args, generator=generator, **kwargs)
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called before hand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        # self.start_counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    # def __len__(self):
    #     # We need a separate self.start_counter because PL seems to call len repeatedly.
    #     # If we use len(self.data_source) - self.counter then PL will think the epoch ends
    #     # when we're only half way through.
    #     return len(self.data_source) - self.start_counter

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        # self.start_counter = self.counter

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
        # self.start_counter = self.counter


class FaultTolerantDistributedSampler(DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        # self.start_counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    # def __len__(self) -> int:
        # return self.num_samples - self.start_counter

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        # self.start_counter = self.counter

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
        # self.start_counter = self.counter


@dataclass
class TrainPlan:
    """Step counts of a run.

    One step uses `batch_size` samples in total, `batch_size // world_size` on each GPU.
    Samples left over after the last full step of an epoch are not used.
    """

    steps_per_epoch: int
    total_steps: int
    warmup_steps: int


def build_train_sampler(
    dataset: Dataset, rank: int, world_size: int, seed: int
) -> FaultTolerantDistributedSampler:
    """Shuffling sampler that gives GPU `rank` its share of every epoch (1 GPU: all)."""
    return FaultTolerantDistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
    )


def plan_training(
    samples_per_rank: int,
    per_rank_step: int,
    epochs: int,
    max_steps: int | None,
    warmup_fraction: float,
) -> TrainPlan:
    """Count the steps of `epochs` passes, cut at `max_steps` if set.

    `samples_per_rank` is `sampler.num_samples`. The warmup is a fraction of all steps.
    """
    steps_per_epoch = samples_per_rank // per_rank_step
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    warmup_steps = max(1, round(warmup_fraction * total_steps))
    return TrainPlan(steps_per_epoch, total_steps, warmup_steps)


def build_loader(
    dataset: Dataset,
    micro_batch: int,
    num_workers: int,
    sampler: FaultTolerantDistributedSampler | None = None,
) -> DataLoader:
    """DataLoader of (inputs, targets) micro-batches in pinned memory, without partial batches."""
    return DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_val_loader(
    dataset: Dataset, n_samples: int, micro_batch: int, num_workers: int
) -> DataLoader:
    """DataLoader of the first `n_samples` samples, in order (the same every validation)."""
    return build_loader(Subset(dataset, range(n_samples)), micro_batch, num_workers)


def micro_batches(
    loader: DataLoader,
    sampler: FaultTolerantDistributedSampler,
    plan: TrainPlan,
    start_step: int,
    per_rank_step: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield the micro-batches of steps `start_step` to `plan.total_steps - 1`, in order.

    Each step takes `per_rank_step // loader.batch_size` micro-batches. Before each epoch
    the sampler is moved to that epoch and to the first sample not yet trained on, so a
    resumed run sees exactly the samples a run without a break would have seen.
    """
    per_step = per_rank_step // loader.batch_size
    step = start_step
    while step < plan.total_steps:
        epoch = step // plan.steps_per_epoch
        done_in_epoch = step % plan.steps_per_epoch
        sampler.load_state_dict({"epoch": epoch, "counter": done_in_epoch * per_rank_step})
        epoch_end = min((epoch + 1) * plan.steps_per_epoch, plan.total_steps)
        batches = iter(loader)
        for _ in range((epoch_end - step) * per_step):
            yield next(batches)
        step = epoch_end


if __name__ == "__main__":
    # Smoke check: 41 dummy tokens, seq_len 4 -> 10 samples; 2 GPUs (rank 0 shown),
    # 2 samples per GPU and step -> 2 steps per epoch, 3 steps in total.
    from dataset.dataset import LMDataset

    tokens = torch.arange(41).numpy()
    dataset = LMDataset(tokens, seq_len=4)
    sampler = build_train_sampler(dataset, rank=0, world_size=2, seed=0)
    plan = plan_training(sampler.num_samples, 2, epochs=2, max_steps=3, warmup_fraction=0.1)
    loader = build_loader(dataset, micro_batch=1, num_workers=0, sampler=sampler)
    full = [x[0, 0].item() // 4 for x, _ in micro_batches(loader, sampler, plan, 0, 2)]
    resumed = [x[0, 0].item() // 4 for x, _ in micro_batches(loader, sampler, plan, 1, 2)]
    print(plan, "samples", full, "resumed at step 1", resumed, resumed == full[2:])
