import os

from typing import (
    List,
    Sequence,
)
from transformers import AutoTokenizer


class HFTokenizer:
    def __init__(self, model_path: str):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.additional_stop_token_ids = set()

    def postprocess_tokens(
        self, tokens: Sequence[int], bos: bool = False, eos: bool = False
    ):
        if bos:
            tokens.insert(0, self.bos_id)

        if eos:
            tokens.append(self.eos_id)

        return tokens

    def encode(
        self,
        s: str,
        *,
        add_special_tokens: bool = True,
        **postprocess_kwargs,
    ) -> List[int]:
        t = self.tokenizer.encode(s, add_special_tokens=add_special_tokens)

        if add_special_tokens:
            t = self.postprocess_tokens(t, **postprocess_kwargs)

        return t

    def decode(self, t: Sequence[int]) -> str:
        return self.tokenizer.decode(t)
