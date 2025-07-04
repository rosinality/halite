from dataclasses import dataclass

import torch
from torch import nn

from halite.transformers.infer.engine.batch import ForwardMode


@dataclass
class LogitsProcessorOutput:
    next_token_logits: torch.Tensor
    next_token_logprobs: torch.Tensor = None

    normalized_prompt_logprobs: torch.Tensor = None
    input_token_logprobs: torch.Tensor = None

    input_top_logprobs: list = None
    output_top_logprobs: list = None

    def slice(self, batch_size):
        return LogitsProcessorOutput(
            self.next_token_logits[:batch_size],
            self.next_token_logprobs[:batch_size]
            if self.next_token_logprobs is not None
            else None,
            self.normalized_prompt_logprobs[:batch_size]
            if self.normalized_prompt_logprobs is not None
            else None,
            self.input_token_logprobs[:batch_size]
            if self.input_token_logprobs is not None
            else None,
            self.input_top_logprobs[:batch_size]
            if self.input_top_logprobs is not None
            else None,
            self.output_top_logprobs[:batch_size]
            if self.output_top_logprobs is not None
            else None,
        )


class LogitsProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, hidden_states, weight, batch):
        if batch.mode == ForwardMode.DECODE:
            last_index = None
            last_hidden = hidden_states

        else:
            last_index = torch.cumsum(batch.extend_lens, dim=0) - 1
            last_hidden = hidden_states[last_index]

        last_logits = last_hidden @ weight.T
        last_logits = last_logits.float()

        return LogitsProcessorOutput(last_logits)
