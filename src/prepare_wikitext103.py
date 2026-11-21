"""Make the WikiText-103 token files that the training config reads.

Reads the raw parquet shards of `wikitext-103-raw-v1` (Hugging Face `Salesforce/wikitext`)
from `data/wikitext103/raw/`, groups the lines into articles, tokenizes each article with
the Llama-3.1 tokenizer behind one `<|begin_of_text|>` token, and writes one flat uint32
array per split: `data/wikitext103/{train,validation,test}.bin`.
`dataset.dataset.load_tokens` reads this format. The steps are the same as in notebooks/wikitext103-data-pipeline.ipynb.

Run from the repo root:
    .venv/bin/python src/prepare_wikitext103.py --tokenizer <Llama-3.1 tokenizer.json>
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataset.tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
# A level-1 heading such as " = Valkyria Chronicles III = \n" starts a new article.
ARTICLE_HEADING = re.compile(r" = [^=].* = \n")


def read_lines(raw_dir: Path, split: str) -> list[str]:
    """All rows of one split, shard after shard. One row is one line of an article."""
    shards = sorted(raw_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no {split}-*.parquet files in {raw_dir}")
    lines: list[str] = []
    for shard in shards:
        lines += pq.read_table(shard).column("text").to_pylist()
    return lines


def split_articles(lines: list[str]) -> list[str]:
    """Join the lines into articles, each starting at a level-1 heading.

    Empty rows are '' and vanish in the join; the '\\n' at the end of each text line
    stays. Articles with only whitespace are dropped.
    """
    articles: list[list[str]] = []
    for line in lines:
        if ARTICLE_HEADING.fullmatch(line) or not articles:
            articles.append([])
        articles[-1].append(line)
    texts = ["".join(article) for article in articles]
    return [text for text in texts if text.strip()]


def tokenize_articles(
    articles: list[str], tokenizer: TextTokenizer, chunk: int = 1000
) -> np.ndarray:
    """One flat uint32 array: `[bos] + ids` of every article, back to back.

    `encode_batch` gives the same ids as `TextTokenizer.encode`, but tokenizes in
    parallel. Chunks keep memory low, because each Encoding also stores offsets and piece
    strings.
    """
    parts = []
    for start in range(0, len(articles), chunk):
        batch = articles[start:start + chunk]
        for encoding in tokenizer.tokenizer.encode_batch(batch, add_special_tokens=False):
            parts.append(np.array([tokenizer.bos_id] + encoding.ids, dtype=np.uint32))
    return np.concatenate(parts)


def main() -> None:
    """Tokenize all three splits and write their .bin files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Llama-3.1 tokenizer.json"
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/wikitext103/raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/wikitext103")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing .bin files"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    outputs = {split: args.out_dir / f"{split}.bin" for split in SPLITS}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"already exist (use --overwrite to replace): {existing}")

    tokenizer = TextTokenizer(args.tokenizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, path in outputs.items():
        articles = split_articles(read_lines(args.raw_dir, split))
        tokens = tokenize_articles(articles, tokenizer)
        tmp = path.with_suffix(".tmp")
        tokens.tofile(tmp)
        tmp.replace(path)
        logger.info(
            "%s: %d articles, %d tokens -> %s (%.1f MB)",
            split, len(articles), len(tokens), path, path.stat().st_size / 1e6,
        )


if __name__ == "__main__":
    main()
