from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class UnpadParams:
    cu_seqlens_q: torch.Tensor
    max_length_q: int

    batch: int | None = None
    seqlen: int | None = None
    indices_q: torch.Tensor | None = None
    indices_k: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_length_k: int | None = None

    def __post_init__(self):
        if self.indices_k is None:
            self.indices_k = self.indices_q

        if self.cu_seqlens_k is None:
            self.cu_seqlens_k = self.cu_seqlens_q

        if self.max_length_k is None:
            self.max_length_k = self.max_length_q


@dataclass
class TransformerDecoderOutput:
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    loss_dict: dict[str, torch.Tensor] | None = None
    last_hidden_state: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None
    cache: Any | None = None
    aux_outputs: Any | None = None
