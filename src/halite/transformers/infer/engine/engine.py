from typing import Any

from torch import nn

from halite.transformers.infer.engine.model_runner import ModelConfig, ModelRunner
from halite.transformers.infer.engine.scheduler import Scheduler, ServerConfig


class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_config: ModelConfig,
        server_config: ServerConfig | None = None,
    ):
        self.model_runner = ModelRunner(model, model_config)

        if server_config is None:
            server_config = ServerConfig()

        self.scheduler = Scheduler(self.model_runner, tokenizer, server_config)

        self.model_config = model_config
        self.server_config = server_config
        self.tokenizer = tokenizer

    def infer_batch(self, requests):
        requests = self.scheduler.build_batch_request(requests)

        return self.scheduler.infer_batch(requests)
