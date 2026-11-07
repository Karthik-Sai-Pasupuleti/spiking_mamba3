"""Download FineWeb-Edu sample-100BT and tokenize it into uint32 token files for `train.py`.

Source: Hugging Face dataset HuggingFaceFW/fineweb-edu, folder sample/100BT (140 parquet files,
286 GB), pinned to revision REVISION. Tokenizer: Llama-3.1 `tokenizer.json` from
NousResearch/Meta-Llama-3.1-8B (an ungated copy), pinned to TOKENIZER_REVISION.

Output, one `.bin` per parquet file (flat uint32 token ids, ~4 bytes per token):
    <out>/train/<name>.bin   139 files
    <out>/val/<name>.bin     VAL_FILE only; never used for training

The recipe is the one of `data/tokens/` and `scripts/prepare_wikitext2.py`: every document
becomes `[<|begin_of_text|>] + tokenize(text)`, all documents back to back, no padding, no
end-of-text token.

The script can be stopped and started again: a finished `.bin` is skipped, and a file is
written as `.bin.tmp` and renamed only when it is complete. With `--delete-parquet` each
parquet file is deleted after its `.bin` is written, so the disk holds ~400 GB of tokens
instead of ~690 GB at the peak.

Run from the project root:
    .venv/bin/python -m scripts.prepare_fineweb --out data-fineweb-100bt --workers 16 --delete-parquet
Test on local parquet files without downloading:
    .venv/bin/python -m scripts.prepare_fineweb --out <dir> --parquet-dir <dir with *.parquet> --limit 1
"""

import argparse
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from dataset.tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SUBSET = "sample/100BT/"
VAL_FILE = "013_00009"
TOKENIZER_REPO = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"


def remote_parquet_files() -> list[str]:
    """Paths of the sample-100BT parquet files in the dataset repo, sorted."""
    files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    return sorted(f for f in files if f.startswith(SUBSET) and f.endswith(".parquet"))


def output_path(out: Path, name: str) -> Path:
    """`<out>/val/<name>.bin` for VAL_FILE, else `<out>/train/<name>.bin`."""
    split = "val" if name == VAL_FILE else "train"
    return out / split / f"{name}.bin"


def tokenize_parquet(parquet: Path, out_bin: Path, tokenizer: TextTokenizer) -> int:
    """Write the token ids of every document in `parquet` to `out_bin`; return the count.

    Reads one row group at a time, so memory stays small. Encodes in batches, which gives
    the same ids as `tokenizer.encode` per document.
    """
    tmp = out_bin.with_suffix(".bin.tmp")
    n_tokens = 0
    pf = pq.ParquetFile(parquet)
    with tmp.open("wb") as f:
        for group in range(pf.num_row_groups):
            texts = pf.read_row_group(group, columns=["text"]).column("text").to_pylist()
            encodings = tokenizer.tokenizer.encode_batch(texts, add_special_tokens=False)
            parts = [np.array([tokenizer.bos_id] + e.ids, dtype=np.uint32) for e in encodings]
            ids = np.concatenate(parts)
            ids.tofile(f)
            n_tokens += len(ids)
    tmp.replace(out_bin)
    return n_tokens


_tokenizer: TextTokenizer | None = None


def _init_worker(tokenizer_path: Path, threads: int) -> None:
    """Per worker process: limit the tokenizer's threads, then load the tokenizer once."""
    global _tokenizer
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    _tokenizer = TextTokenizer(tokenizer_path)


def _process(job: tuple[str, Path | None, Path, Path, bool]) -> tuple[str, int, float]:
    """Download (unless local) and tokenize one parquet file. Returns (name, tokens, seconds)."""
    remote, local, raw_dir, out_bin, delete = job
    start = time.perf_counter()
    if local is None:
        parquet = Path(hf_hub_download(DATASET, remote, repo_type="dataset",
                                       revision=REVISION, local_dir=raw_dir))
    else:
        parquet = local
    assert _tokenizer is not None
    n_tokens = tokenize_parquet(parquet, out_bin, _tokenizer)
    if delete and local is None:
        parquet.unlink()
    return out_bin.stem, n_tokens, time.perf_counter() - start


def main() -> None:
    """Fetch the tokenizer, then download and tokenize every file that has no .bin yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data-fineweb-100bt"))
    parser.add_argument("--workers", type=int, default=8, help="files processed in parallel")
    parser.add_argument("--delete-parquet", action="store_true")
    parser.add_argument("--parquet-dir", type=Path, default=None,
                        help="tokenize local *.parquet files instead of downloading")
    parser.add_argument("--limit", type=int, default=None, help="only the first N files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP request otherwise

    for split in ("train", "val"):
        (args.out / split).mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(hf_hub_download(TOKENIZER_REPO, "tokenizer.json",
                                          revision=TOKENIZER_REVISION, local_dir=args.out))
    raw_dir = args.out / "raw"

    if args.parquet_dir is None:
        sources = [(remote, None) for remote in remote_parquet_files()]
    else:
        sources = [(p.name, p) for p in sorted(args.parquet_dir.glob("*.parquet"))]
    sources = sources[: args.limit]
    jobs = []
    for remote, local in sources:
        out_bin = output_path(args.out, Path(remote).stem)
        if not out_bin.exists():
            jobs.append((remote, local, raw_dir, out_bin, args.delete_parquet))
    logger.info("%d files, %d already done, %d to do, %d workers",
                len(sources), len(sources) - len(jobs), len(jobs), args.workers)

    # CPUs this process may use (a SLURM job gets fewer than the node has).
    threads = max(1, len(os.sched_getaffinity(0)) // args.workers)
    total, done = 0, 0
    with Pool(args.workers, initializer=_init_worker, initargs=(tokenizer_path, threads)) as pool:
        for name, n_tokens, seconds in pool.imap_unordered(_process, jobs):
            total, done = total + n_tokens, done + 1
            logger.info("%s: %d tokens in %.0f s (%d/%d files, %.2fB tokens this run)",
                        name, n_tokens, seconds, done, len(jobs), total / 1e9)


if __name__ == "__main__":
    main()
