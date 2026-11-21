"""Chat with a trained model in the terminal: type a prompt, the model continues it.

The models here are base language models trained on WikiText-103 (Wikipedia articles),
not chat assistants. They continue text in Wikipedia style and do not follow
instructions: "The Eiffel Tower is" works better than "What is the Eiffel Tower?".
Every prompt is independent; the model does not see earlier prompts or replies.

A reply stops at the end of a paragraph, at `<|begin_of_text|>` (the start of a new
article in the training data) or after `--max-new-tokens`. Each new token runs the full
forward pass over the text so far, as in `test.generate`; that is fine for short replies.
Ctrl-C stops a reply; /quit or Ctrl-D ends the chat.

Run from the repo root (default: the best checkpoint of the spiking WikiText-103 run):
    .venv/bin/python src/chat.py [--checkpoint runs/<run>/best.pt]
"""

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn

from configs.config import load_config
from dataset.tokenizer import TextTokenizer
from models.model import MambaConfig, MambaLMHeadModel
import spiking_mamba3  # noqa: F401  registers the "SpikingMamba3" layer for configs
from test import pick_next
from utils.checkpoint import load_model_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "mamba3-siso-187m-wikitext103-spiking.yaml"
# Best epoch (92) of the 100-epoch spiking run: val ppl 20.30, test ppl 21.06.
DEFAULT_CHECKPOINT = (
    ROOT / "runs" / "mamba3-siso-187m-wikitext103-spiking" / "best.pt"
)
CONTEXT = 2048  # tokens; the length the models were trained on
QUIT_COMMANDS = {"/quit", "/exit"}


def default_tokenizer() -> Path | None:
    """The Llama-3.1 tokenizer.json in the local Hugging Face cache, or None."""
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    pattern = "models--NousResearch--Meta-Llama-3.1-8B/snapshots/*/tokenizer.json"
    found = sorted(cache.glob(pattern))
    return found[-1] if found else None


def load_model(config_path: Path, checkpoint: Path) -> nn.Module:
    """Build the model of `config_path` on the GPU, load `checkpoint`, set eval mode."""
    cfg = load_config(config_path)
    model = MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")
    model.load_state_dict(load_model_state(checkpoint))
    return model.eval()


@torch.no_grad()
def stream_tokens(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_ids: set[int],
    generator: torch.Generator,
) -> Iterator[int]:
    """Yield new token ids one at a time, continuing `prompt_ids`.

    Stops before a token in `stop_ids` or after `max_new_tokens`. The model sees at most
    the last CONTEXT tokens.
    """
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = torch.tensor([ids[-CONTEXT:]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(window)[0, -1].float()
        next_id = pick_next(logits, temperature, top_k, generator)
        if next_id in stop_ids:
            return
        ids.append(next_id)
        yield next_id


def print_reply(tokenizer: TextTokenizer, tokens: Iterator[int]) -> None:
    """Print the reply while it is generated; stop at its first line break."""
    new_ids: list[int] = []
    shown = 0
    for token_id in tokens:
        new_ids.append(token_id)
        text = tokenizer.decode(new_ids)
        if text.endswith("�"):  # half of a UTF-8 character: wait for the next byte
            continue
        if "\n" in text:
            print(text[shown:text.index("\n")], end="", flush=True)
            break
        print(text[shown:], end="", flush=True)
        shown = len(text)
    print()


def main() -> None:
    """Load the model and answer prompts until /quit or Ctrl-D."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=default_tokenizer())
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=50, help="0 = no top-k cut")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.tokenizer is None:
        parser.error("no Llama-3.1 tokenizer.json in the HF cache; pass --tokenizer")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}; pass --checkpoint")

    model = load_model(args.config, args.checkpoint)
    tokenizer = TextTokenizer(args.tokenizer)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    print(f"Loaded {args.checkpoint}.")
    print("Type the start of a text; /quit or Ctrl-D to leave.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in QUIT_COMMANDS:
            break
        if not prompt:
            continue
        # Training lines start with a space, so a prompt reads like one.
        prompt_ids = tokenizer.encode(" " + prompt)
        print(f"model> {prompt}", end="", flush=True)
        tokens = stream_tokens(
            model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k,
            stop_ids={tokenizer.bos_id}, generator=generator,
        )
        try:
            print_reply(tokenizer, tokens)
        except KeyboardInterrupt:
            print(" [stopped]")


if __name__ == "__main__":
    main()
