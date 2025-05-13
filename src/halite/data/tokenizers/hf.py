from typing import (
    List,
    Sequence,
)
from transformers import AutoTokenizer


class HFTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_id = self.tokenizer.eos_token_id
        self.additional_stop_token_ids = set()

    def encode(self, s: str, *, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=add_special_tokens)

    def decode(self, t: Sequence[int]) -> str:
        return self.tokenizer.decode(t)
