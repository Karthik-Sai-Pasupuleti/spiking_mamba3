"""Inference check: generate text from a checkpoint, one token at a time.

Run it with `python src/test.py --config <yaml> --checkpoint <pt> --tokenizer
<tokenizer.json> --prompt "..."`.

Extend a token sequence one token at a time.

Each new token comes from a full forward pass over the whole sequence so far, with the
same chunked SISO kernels as training. So it tests the trained weights through the
training forward path. The work grows with the square of the length, which is fine for
short checks. The recurrent one-token kernel (`mamba3_siso_step`) is not used here.
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn


def pick_next(
    logits: Tensor, temperature: float, top_k: int, generator: torch.Generator
) -> int:
    """Choose the next token id from a 1-D logits vector.

    `temperature == 0` takes the most likely token (greedy). Otherwise the logits are
    divided by `temperature`, cut to the `top_k` largest (if `top_k > 0`), and one token
    is sampled with `generator`.
    """
    if temperature == 0:
        return int(logits.argmax())
    scaled = logits / temperature
    if top_k > 0:
        values, indices = torch.topk(scaled, top_k)
        choice = torch.multinomial(torch.softmax(values, dim=-1), 1, generator=generator)
        return int(indices[choice])
    choice = torch.multinomial(torch.softmax(scaled, dim=-1), 1, generator=generator)
    return int(choice)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_id: int | None,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> list[int]:
    """Return up to `max_new_tokens` new token ids that continue `prompt_ids`.

    Stops early when the model picks `stop_id`; that id is not returned. The model runs on
    the GPU under autocast with `dtype`. Sampling uses its own generator seeded with
    `seed`, so a run is repeatable and does not change the global random state.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.tensor([prompt_ids], device="cuda")
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=dtype):
            logits = model(ids)[0, -1].float()
        next_id = pick_next(logits, temperature, top_k, generator)
        if next_id == stop_id:
            break
        new_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device="cuda")], dim=1)
    model.train(was_training)
    return new_ids


def main() -> None:
    """Load a checkpoint, continue a text prompt, and print the result."""
    from configs.config import load_config
    from dataset.tokenizer import TextTokenizer
    from models.model import MambaLMHeadModel, MambaConfig
    from utils.checkpoint import load_model_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = MambaLMHeadModel(MambaConfig(**cfg["model"]), device="cuda")
    model.load_state_dict(load_model_state(args.checkpoint))
    tokenizer = TextTokenizer(args.tokenizer)
    new_ids = generate(
        model, tokenizer.encode(args.prompt), args.max_new_tokens,
        args.temperature, args.top_k, stop_id=None, seed=args.seed,
    )
    print(args.prompt + tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
