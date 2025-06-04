from flashinfer import top_k_top_p_sampling_from_probs
import torch
from torch import nn
from torch.nn import functional as F


def calc_tokens_and_logprobs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature is not None:
        logits.div_(temperature.unsqueeze(-1))

    probs = torch.softmax(logits, -1)
    logits = None
    del logits

    probs = probs.clone()

    top_ks = torch.where(greedy_mask, 1, 1 << 30)

    next_token_ids = top_k_top_p_sampling_from_probs(
        probs, top_ks, 1.0, filter_apply_order="joint"
    ).to(torch.int64)

    # TODO: because this is all in fp32, I think the numerics are ok here.
    chosen_probs = probs.gather(dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)
    chosen_logprobs = chosen_probs.log()

    return next_token_ids, chosen_logprobs


class LogitsProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, hidden_states, weight, batch_state):
        if batch_state.lm_head_indices.numel() == 0:
            next_token_ids = torch.empty(
                0, device=hidden_states.device, dtype=torch.long
            )
            chosen_logprobs = torch.empty(
                0, device=hidden_states.device, dtype=torch.float32
            )

        else:
            hidden_states = hidden_states[batch_state.lm_head_indices].contiguous()

            logits = hidden_states @ weight.T
            logits = logits.float()

            next_token_ids, chosen_logprobs = calc_tokens_and_logprobs(
                logits,
                temperature=batch_state.sampling_params.temperature,
                greedy_mask=batch_state.sampling_params.greedy_mask,
            )

        batch_state.output_ids = next_token_ids
        batch_state.logprobs = chosen_logprobs

        return batch_state
