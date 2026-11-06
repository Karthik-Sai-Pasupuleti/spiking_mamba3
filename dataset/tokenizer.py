"""Turn text into token ids and back, with a Hugging Face `tokenizer.json` file.

The training data and the pretrained Mamba-3 checkpoints use the Llama-3.1 tokenizer,
and every document starts with `<|begin_of_text|>` (id 128000). `TextTokenizer.encode`
puts that token in front exactly once, so a prompt looks like the start of a training
document. Only the local file is read; nothing is downloaded.
"""

from pathlib import Path

from tokenizers import Tokenizer


class TextTokenizer:
    """Wrapper around `tokenizers.Tokenizer` with an explicit begin-of-text token.

    Build it from the path of a `tokenizer.json`. `encode` returns `[bos_id] + ids`;
    `decode` drops special tokens.
    """

    def __init__(self, path: Path, bos_token: str = "<|begin_of_text|>") -> None:
        self.tokenizer = Tokenizer.from_file(str(path))
        bos_id = self.tokenizer.token_to_id(bos_token)
        if bos_id is None:
            raise ValueError(f"{bos_token!r} is not in {path}")
        self.bos_id: int = bos_id

    def encode(self, text: str) -> list[int]:
        """Token ids of `text`, with the begin-of-text id in front."""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return [self.bos_id] + ids

    def decode(self, ids: list[int]) -> str:
        """Text of `ids`, without special tokens."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    # Smoke check: a tiny word-level tokenizer in a temp file, encode and decode once.
    import tempfile

    from tokenizers import models, pre_tokenizers

    vocab = {"<|begin_of_text|>": 0, "[UNK]": 1, "hello": 2, "world": 3}
    tiny = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tiny.pre_tokenizer = pre_tokenizers.Whitespace()
    tiny.add_special_tokens(["<|begin_of_text|>"])
    with tempfile.TemporaryDirectory() as tmp:
        tiny_file = Path(tmp) / "tokenizer.json"
        tiny.save(str(tiny_file))
        tok = TextTokenizer(tiny_file)
        encoded = tok.encode("hello world")
        print(encoded, repr(tok.decode(encoded)))
