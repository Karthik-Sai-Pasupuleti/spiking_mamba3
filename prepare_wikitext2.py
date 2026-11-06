"""Tokenize WikiText-2 (raw) into the flat uint32 token files that `train.py` reads.

Input: the three parquet files of `Salesforce/wikitext`, config `wikitext-2-raw-v1`, in
`data-wikitext2/raw/` (train, validation, test). Each parquet row is one line of text.
Output: `data-wikitext2/{train,validation,test}.bin`.

The recipe is the one of the FineWeb-Edu files in `data/tokens/`: every document becomes
`[<|begin_of_text|>] + tokenize(text)`, all documents back to back, no padding, no
end-of-text token. Here a document is one Wikipedia article. A new article starts at each
level-1 heading line such as " = Robert Boulter = ". The raw text is kept as it is,
including WikiText's own spacing (" @-@ ", spaces before punctuation).

Run: .venv/bin/python prepare_wikitext2.py --tokenizer <path to Llama-3.1 tokenizer.json>
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataset.tokenizer import TextTokenizer

logger = logging.getLogger(__name__)

SPLITS = ["train", "validation", "test"]
ARTICLE_HEADING = re.compile(r" = [^=].* = \n")


def split_articles(lines: list[str]) -> list[str]:
    """Group the lines of one split into articles; each starts at a level-1 heading.

    Blank lines before the first heading would form an empty article; they are dropped.
    """
    articles: list[list[str]] = []
    for line in lines:
        if ARTICLE_HEADING.fullmatch(line) or not articles:
            articles.append([])
        articles[-1].append(line)
    texts = ["".join(article) for article in articles]
    return [text for text in texts if text.strip()]


def tokenize_articles(articles: list[str], tokenizer: TextTokenizer) -> np.ndarray:
    """Concatenate `[bos] + ids` of every article into one uint32 array."""
    parts = [np.array(tokenizer.encode(text), dtype=np.uint32) for text in articles]
    return np.concatenate(parts)


def main() -> None:
    """Write one .bin file per split and log its size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data-wikitext2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tokenizer = TextTokenizer(args.tokenizer)
    for split in SPLITS:
        lines = pq.read_table(args.data_dir / "raw" / f"{split}.parquet").column("text")
        articles = split_articles(lines.to_pylist())
        tokens = tokenize_articles(articles, tokenizer)
        out = args.data_dir / f"{split}.bin"
        tokens.tofile(out)
        logger.info("%s: %d articles -> %d tokens -> %s", split, len(articles), len(tokens), out)


if __name__ == "__main__":
    main()
