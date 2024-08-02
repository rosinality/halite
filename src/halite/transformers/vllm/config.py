from dataclasses import dataclass
from typing import Optional, Tuple

from meshfn.utils import get_torch_dtype


class ModelConfig:
    def __init__(
        self,
        transformer_config,
        tokenizer,
        tokenizer_mode,
        trust_remote_code,
        download_dir,
        load_format,
        dtype,
        seed,
        revision=None,
        max_model_len=None,
    ):
        self.model = "meshfn"
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.dtype = get_torch_dtype(dtype)

        self.config = transformer_config

        self.max_model_len = max_model_len

    def get_hidden_size(self):
        return self.config.dim

    def get_head_size(self):
        return self.config.head_dim

    def get_num_heads(self, parallel_config):
        return self.config.n_heads // parallel_config.tensor_parallel_size

    def get_max_model_len(self):
        max_len = self.config.max_length

        if max_len is None:
            max_len = float("inf")

        return max_len

    def get_num_layers(self, parallel_config):
        return self.config.n_layers // parallel_config.pipeline_parallel_size
