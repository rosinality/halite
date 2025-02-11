import copy
from typing import Any
import uuid

from torch import nn

from halite.transformers.infer.engine.model_runner import ModelConfig, ModelRunner
from halite.transformers.infer.engine.scheduler import Scheduler, ServerConfig


class InferenceResult:
    def __init__(self, id: int, input_ids: list[int], output_ids: list[list[int]]):
        self.id = id
        self.input_ids = input_ids
        self.output_ids = output_ids

    def to_dict(self):
        return {
            "id": self.id,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
        }

    def __repr__(self):
        if len(self.output_ids) > 2:
            output_ids = [self.output_ids[0], "...", self.output_ids[-1]]

        else:
            output_ids = self.output_ids

        output_ids = ", ".join(str(out) for out in output_ids)

        return f"InferenceResult(id={self.id}, input_ids={self.input_ids}, output_ids=[{output_ids}])"


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

        self.initialized = True

    def initialize(self):
        if self.initialized:
            return

        self.model_runner.initialize()
        self.scheduler.initialize()

        self.initialized = True

    def cleanup(self):
        if not self.initialized:
            return

        self.model_runner.cleanup()
        self.scheduler.cleanup()

        self.initialized = False

    def load_state_dict(self, state_dict, assign=True):
        self.model_runner.load_state_dict(state_dict, assign=assign)

    def infer_batch(self, requests):
        requests = self.scheduler.build_batch_request(requests)

        sample_per_req = requests[0].sampling_params.n
        assert all(
            req.sampling_params.n == sample_per_req for req in requests
        ), "all requests must have the same number of samples"

        if sample_per_req == 1:
            generated = self.scheduler.infer_batch(requests)
            generated = sorted(generated, key=lambda x: x[0])

            return [
                InferenceResult(id, input_ids, [output_ids])
                for id, (input_ids, output_ids) in generated
            ]

        prefix_reqs = []
        for req in requests:
            prefix_req = copy.copy(req)
            prefix_req.id = uuid.uuid4().hex
            prefix_req.sampling_params = copy.copy(prefix_req.sampling_params)
            prefix_req.sampling_params.max_new_tokens = 0

        self.scheduler.infer_batch(prefix_reqs)

        req_ids = {}
        gen_reqs = []
        for req in requests:
            for _ in range(sample_per_req):
                gen_req = copy.copy(req)
                gen_req.id = uuid.uuid4().hex
                req_ids[gen_req.id] = req.id
                gen_reqs.append(gen_req)

        generated = self.scheduler.infer_batch(gen_reqs)
        output_dict = {}
        for id, input_id, output_id in generated:
            req_id = req_ids[id]

            if req_id not in output_dict:
                output_dict[req_id] = (input_id, [])

            output_dict[req_id][1].append(output_id)

        outputs = sorted(output_dict.items(), key=lambda x: x[0])

        return [
            InferenceResult(id, input_ids, output_ids)
            for id, (input_ids, output_ids) in outputs
        ]
