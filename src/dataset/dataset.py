"""Token dataset for language-model pretraining: FlashAttention's `LMDataset`.

`LMDataset` below is copied unchanged from Dao-AILab/flash-attention, file
training/src/datamodules/datasets/lm_dataset.py at commit 34a06b9 (BSD-3-Clause, see
LICENSE-flash-attention). It cuts one flat token array into samples of `seq_len` inputs
and `seq_len` targets. Sample `i` reads tokens `i * seq_len` to `i * seq_len + seq_len + 1`,
so neighbouring samples share one token and no token is lost.

`load_tokens` is ours: it memory-maps a pre-tokenized `.bin` file (flat uint32 token ids,
here FineWeb-Edu with the Llama-3.1 tokenizer), so the file is never loaded whole.
`dataset.dataloader` turns the dataset into batches.
"""

from pathlib import Path

# Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt/datasets.py
# Except we don't pad the last block and don't use overlapping eval
# And we return both the input and the target
import math
import numpy as np

import torch


class LMDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, seq_len, drop_last=True):
        """tokens should be a numpy array
        """
        self.seq_len = seq_len
        ntokens = len(tokens)
        if drop_last:
            ntokens = ((ntokens - 1) // seq_len) * seq_len + 1
        self.ntokens = ntokens
        # We're careful not to slice tokens, since it could be a memmap'ed array or H5 dataset,
        # and slicing would load it to memory.
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / self.seq_len)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.ntokens - 1 - start_idx)
        data = torch.as_tensor(self.tokens[start_idx:(start_idx + seq_len + 1)].astype(np.int64))
        return data[:-1], data[1:].clone()


def load_tokens(path: Path) -> np.ndarray:
    """Memory-map a `.bin` file of uint32 token ids as a read-only 1-D array."""
    return np.memmap(path, dtype=np.uint32, mode="r")


if __name__ == "__main__":
    # Smoke check: 21 dummy tokens, seq_len 4 -> 5 samples; sample 1 is tokens 4..8.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dummy_file = Path(tmp) / "tokens.bin"
        np.arange(21, dtype=np.uint32).tofile(dummy_file)
        dataset = LMDataset(load_tokens(dummy_file), seq_len=4)
        x, y = dataset[1]
        print(len(dataset), x.tolist(), y.tolist())
