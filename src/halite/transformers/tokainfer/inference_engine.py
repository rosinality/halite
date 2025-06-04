import os
import uuid
import time
from typing import Sequence

from slickconf import instantiate
from torch import distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor
from torch import multiprocessing as mp

from halite.transformers.tokainfer.server_types import (
    SubmittedRequest,
    CancelledRequest,
)
from halite.transformers.tokainfer.types import (
    Engine,
    ServerConfig,
    ProcessInfo,
    UpdateStateDict,
    Initialize,
    Cleanup,
    SamplingParams,
    TokasaurusRequest,
    ServerState,
)
from halite.transformers.tokainfer.entry import get_basic_model_process_dict
from halite.transformers.tokainfer.engine.manager import start_manager


class InferenceEngine:
    def __init__(self, model_config, tokenizer_config, server_config=None):
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.server_config = (
            server_config if server_config is not None else ServerConfig()
        )

        self.tokenizer = instantiate(tokenizer_config)

        self._initialized_distributed = False

    def initialize(
        self,
        master_addr=None,
        master_port=None,
        local_rank=None,
        rank=None,
        world_size=None,
    ):
        if self._initialized_distributed:
            self._reinitialize()

            return

        if master_addr is None:
            master_addr = os.environ["MASTER_ADDR"]

        if world_size is None:
            world_size = int(os.environ["WORLD_SIZE"])

        if local_rank is None:
            local_rank = int(os.environ["LOCAL_RANK"])

        if rank is None:
            rank = int(os.environ["RANK"])

        if self.server_config.placement == "colocated":
            self._initialize_colocated(
                master_addr, master_port, local_rank, rank, world_size
            )

        else:
            self._initialize_dedicated(
                master_addr, master_port, local_rank, rank, world_size
            )

        self._initialized_distributed = True

    def _initialize_colocated(
        self,
        master_addr,
        master_port,
        local_rank,
        rank,
        world_size,
    ):
        mp.set_start_method("spawn", force=True)

        self._engines = [
            self._build_engine(master_addr, master_port, local_rank, rank, world_size)
            for _ in range(1)
        ]

        self._state = ServerState(engines=self._engines)

        pooled_proc_dict = {}
        for engine in self._engines:
            for proc_name, proc_info in engine.proc_dict.items():
                pooled_proc_dict[proc_name] = proc_info

        barrier = mp.Barrier(len(pooled_proc_dict) + 1)

        for proc_name, proc_info in pooled_proc_dict.items():
            proc_info.kwargs["barrier"] = barrier
            proc_info.kwargs["process_name"] = proc_name

        self._processes = []

        for proc_info in pooled_proc_dict.values():
            p = proc_info.make_process()
            p.start()

            self._processes.append(p)

        barrier.wait()

    def _build_engine(
        self,
        master_addr,
        master_port,
        local_rank,
        rank,
        world_size,
    ):
        q_manager_to_model = mp.Queue()
        q_model_to_manager = mp.Queue()
        q_server_to_manager = mp.Queue()
        q_manager_to_server = mp.Queue()

        if master_port is None:
            raise ValueError(
                "master_port is required for colocated placement, and it should be distinct from port of training group"
            )

        process_dict, q_server_to_models = get_basic_model_process_dict(
            self.model_config,
            self.server_config,
            q_manager_to_model,
            q_model_to_manager,
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            local_rank=local_rank,
            rank=rank,
            group_ranks=[],
        )

        process_dict["manager"] = ProcessInfo(
            target=start_manager,
            kwargs={
                "tokenizer": self.tokenizer_config,
                "config": self.server_config,
                "q_manager_to_model": q_manager_to_model,
                "q_model_to_manager": q_model_to_manager,
                "q_server_to_manager": q_server_to_manager,
                "q_manager_to_server": q_manager_to_server,
            },
        )

        worker_barrier = mp.Barrier(len(process_dict))

        for proc_info in process_dict.values():
            proc_info.kwargs["worker_barrier"] = worker_barrier

        engine = Engine(
            q_server_to_manager=q_server_to_manager,
            q_manager_to_server=q_manager_to_server,
            proc_dict=process_dict,
            q_server_to_models=q_server_to_models,
        )

        return engine

    def _reinitialize(self):
        [engine.q_server_to_manager.put(Initialize()) for engine in self._engines]

    def load_state_dict(self, state_dict, assign=True):
        keys = sorted(state_dict.keys())

        if self.server_config.placement == "colocated":
            [
                engine.q_server_to_manager.put(UpdateStateDict(method="queue"))
                for engine in self._engines
            ]

            for key in keys:
                tensor = state_dict[key]

                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()

                for engine in self._engines:
                    [
                        q_server_to_model.put(tensor)
                        for q_server_to_model in engine.q_server_to_models
                    ]

    def cleanup(self):
        [engine.q_server_to_manager.put(Cleanup()) for engine in self._engines]

    def destroy(self):
        for process in self._processes:
            process.kill()

        for process in self._processes:
            process.join()

    def _build_batch_request(self, requests):
        results = []

        for req in requests:
            rid = str(uuid.uuid4())

            if isinstance(req, str):
                req = TokasaurusRequest(
                    id=rid,
                    input_ids=self.tokenizer.encode(req),
                    sampling_params=SamplingParams(),
                )

            elif isinstance(req, Sequence):
                text_or_tokens, sampling_params = req

                input_ids = text_or_tokens

                if isinstance(text_or_tokens, str):
                    input_ids = self.tokenizer.encode(text_or_tokens)

                if not isinstance(sampling_params, SamplingParams):
                    sampling_params = SamplingParams(**sampling_params)

                req = TokasaurusRequest(
                    id=rid,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                )

            elif not isinstance(req, TokasaurusRequest):
                raise ValueError(f"Invalid request type: {type(req)}")

            if isinstance(req.sampling_params.stop, str):
                req.sampling_params.stop = [req.sampling_params.stop]

            results.append(req)

        return results

    def _submit_request(self, req: TokasaurusRequest):
        min_requests = min(self._state.requests_per_engine)
        engine_id = self._state.requests_per_engine.index(min_requests)
        engine = self._engines[engine_id]
        submitted = SubmittedRequest(request=req, engine_index=engine_id)
        # self._state.rid_to_req[req.id] = submitted
        engine.q_server_to_manager.put(req)
        self._state.requests_per_engine[engine_id] += 1

        return submitted

    def _cancel_request(self, submitted: SubmittedRequest):
        engine = self._engines[submitted.engine_index]
        engine.q_server_to_manager.put(CancelledRequest(req_id=submitted.request.id))
        self._state.requests_per_engine[submitted.engine_index] -= 1

    def infer_batch(self, requests):
        requests = self._build_batch_request(requests)
        rid_to_id = {i: req.id for i, req in enumerate(requests)}

        for req in requests:
            self._submit_request(req)

        outputs = {}

        all_finished = False

        while True:
            did_something = False

            for engine in self._engines:
                if not engine.q_manager_to_server.empty():
                    output = engine.q_manager_to_server.get()
                    outputs[output.id] = output

                    if all_finished:
                        print(os.environ["RANK"], output)

                    did_something = True

            if len(outputs) == len(requests):
                all_finished = True

                break

            if not did_something:
                time.sleep(0.01)

        outputs = [outputs[rid_to_id[id]] for id in range(len(requests))]

        for id, output in enumerate(outputs):
            output.id = id

        return outputs
