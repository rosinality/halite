from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F


@dataclass
class TokenResult:
    token: int
    text: str
    logprobs: Optional[List[float]] = None


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(
    tokens,
    model,
    tokenizer,
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,
    max_batch_size=32,
    max_seq_len=2048,
):
    prompt_tokens = tokens

    bsz = len(tokens)
    assert bsz <= max_batch_size, (bsz, max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)

    if max_prompt_len >= max_seq_len:
        print(f"Out of token budget {max_prompt_len} vs {max_seq_len}", "red")
        return

    total_len = min(max_gen_len + max_prompt_len, max_seq_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id

    if echo:
        for i, t in enumerate(tokens):
            yield TokenResult(
                token=t,
                text=tokenizer.decode([t]),
                logprobs=(token_logprobs[0, i : i + 1].tolist() if logprobs else None),
            )

    stop_tokens = torch.tensor(tokenizer.stop_tokens)
    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

        if cur_pos == min_prompt_len:
            print(logits[:, -1])

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token

        target = tokens[:, prev_pos + 1 : cur_pos + 1]

        if logprobs:
            token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=target,
                reduction="none",
                ignore_index=pad_id,
            )
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            torch.isin(next_token, stop_tokens)
        )
        yield TokenResult(
            token=next_token[0].item(),
            text=tokenizer.decode(next_token.tolist()),
            logprobs=(
                token_logprobs[:, cur_pos : cur_pos + 1][0].tolist()
                if logprobs
                else None
            ),
        )

        prev_pos = cur_pos
        if all(eos_reached):
            break
