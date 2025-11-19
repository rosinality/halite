import asyncio
import copy
import math
import uuid
from typing import NamedTuple

from monarch.actor import Actor, endpoint
from monarch.rdma import RDMABuffer
import torch
from torch.distributed.tensor import DTensor, Placement

import wandb

from halite.projects.common.rollout import Rollout, Response


class FinishedGeneration:
    pass


class Empty:
    pass


class TensorSpec(NamedTuple):
    name: str
    shape: tuple[int]
    dtype: torch.dtype
    placements: tuple[Placement] | None = None
    full_shape: tuple[int] | None = None
    stride: tuple[int] | None = None
    rdma_buffer: RDMABuffer | None = None

    @staticmethod
    def from_tensor(name: str, tensor: torch.Tensor | DTensor, use_rdma: bool = False):
        rdma_buffer = None

        if use_rdma:
            rdma_tensor = tensor

            if isinstance(tensor, DTensor):
                rdma_tensor = tensor.to_local()

            rdma_buffer = RDMABuffer(rdma_tensor.view(torch.uint8).flatten())

        if isinstance(tensor, DTensor):
            return TensorSpec(
                name=name,
                shape=tensor.to_local().shape,
                dtype=tensor.dtype,
                placements=tensor.placements,
                full_shape=tensor.shape,
                stride=tensor.stride(),
                rdma_buffer=rdma_buffer,
            )

        return TensorSpec(
            name=name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            rdma_buffer=rdma_buffer,
        )

    @property
    def is_dtensor(self):
        return self.placements is not None


def get_peer_ranks(rank, size, target_size, overlap_targets=False):
    if size == target_size:
        return [rank]

    elif size < target_size:
        step = math.ceil(size / target_size)
        ranks = list(range(rank * step, (rank + 1) * step))
        ranks = [rank for rank in ranks if rank < size]

        return ranks

    else:
        if not overlap_targets:
            if rank < target_size:
                return [rank]

            else:
                return []

        rank_groups = []
        for i in range(target_size):
            rank_groups.append([i] * (size // target_size))

        for i in range(size % target_size):
            rank_groups[i].append(rank_groups[i][0])

        ranks = sum(rank_groups, [])
        ranks = [ranks[rank]]

        return ranks


def chunk_list(x, n_chunk):
    size = len(x) // n_chunk
    remain = len(x) % n_chunk

    chunks = []
    for i in range(n_chunk):
        start = i * size
        end = (i + 1) * size + (1 if i < remain else 0)
        chunks.append(x[start:end])

    return chunks


def infinite_loader(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


class GeneratorWorker:
    def __init__(
        self,
        inference_engine,
        trajectory_queue,
        global_pg,
        mesh_train,
        rank,
        use_rdma=False,
    ):
        self.scheduler = inference_engine.scheduler
        self.model = inference_engine.model_runner.model

        self.trajectory_queue = trajectory_queue.slice(gpus=0)
        self.training_condition = asyncio.Condition()
        self.training_finished = False

        self.global_pg = global_pg
        self.mesh_train = mesh_train
        self.device = self.model.device
        self.rank = rank
        self.use_rdma = use_rdma

        self.update_event = asyncio.Event()

    def set_state_dict_plans(self, state_dict_plans):
        self.state_dict_plans = state_dict_plans

    async def update_state_dict(self):
        self.update_event.set()

    def update_state_dict_recv(self):
        received_tensors = {}

        self.global_pg.group_start()

        for tensor_spec in self.state_dict_plans:
            tensor = torch.empty(
                tensor_spec.shape, dtype=tensor_spec.dtype, device=self.device
            )
            self.global_pg.recv(tensor, self.rank)
            received_tensors[tensor_spec.name] = (tensor, tensor_spec)

        self.global_pg.group_end()

        state_dict = self.model.state_dict()
        received_state_dict = {}

        for name, (tensor, tensor_spec) in received_tensors.items():
            if not tensor_spec.is_dtensor:
                received_state_dict[name] = tensor

                continue

            tensor = DTensor.from_local(
                tensor,
                self.mesh_train,
                tensor_spec.placements,
                shape=tensor_spec.full_shape,
                stride=tensor_spec.stride,
            )

            target_tensor = state_dict[name]

            if not isinstance(target_tensor, DTensor):
                received_state_dict[name] = tensor.full_tensor()

            else:
                received_state_dict[name] = tensor

        self.model.load_state_dict(received_state_dict)

    def update_state_dict_rdma(self):
        received_tensors = {}

        for tensor_spec in self.state_dict_plans:
            tensor = torch.empty(
                tensor_spec.shape, dtype=tensor_spec.dtype, device=self.device
            )
            tensor_spec.rdma_buffer.read_into(tensor.view(torch.uint8).flatten())
            received_tensors[tensor_spec.name] = (tensor, tensor_spec)

        state_dict = self.model.state_dict()
        received_state_dict = {}

        for name, (tensor, tensor_spec) in received_tensors.items():
            if not tensor_spec.is_dtensor:
                received_state_dict[name] = tensor

                continue

            tensor = DTensor.from_local(
                tensor,
                self.mesh_train,
                tensor_spec.placements,
                shape=tensor_spec.full_shape,
                stride=tensor_spec.stride,
            )

            target_tensor = state_dict[name]

            if not isinstance(target_tensor, DTensor):
                received_state_dict[name] = tensor.full_tensor()

            else:
                received_state_dict[name] = tensor

        self.model.load_state_dict(received_state_dict)

    async def finished_training(self):
        async with self.training_condition:
            self.training_finished = True
            self.training_condition.notify_all()

    async def wait_for_finish(self):
        async with self.training_condition:
            await self.training_condition.wait_for(lambda: self.training_finished)

    async def run(self):
        finished = False
        request_ids = {}
        output_params = {}
        output_buffer = {}
        prefix_req_ids = set()

        get_next_requests = True
        fetched_requests = False

        while True:
            if self.update_event.is_set():
                if self.use_rdma:
                    self.update_state_dict_rdma()

                else:
                    self.update_state_dict_recv()

                self.update_event.clear()

            if get_next_requests or finished:
                requests = await self.trajectory_queue.get_input.call_one(
                    nowait=not finished
                )

                if isinstance(requests, FinishedGeneration):
                    break

                fetched_requests = False
                if isinstance(requests, Empty):
                    batch_requests = []

                else:
                    fetched_requests = True

                    batch_requests = self.scheduler.build_batch_request(requests)

                    for request in batch_requests:
                        output_params[request.id] = request.sampling_params

            else:
                batch_requests = []

            prefix_reqs = []
            for request in batch_requests:
                if request.sampling_params.n == 1:
                    continue

                prefix_req = copy.copy(request)
                prefix_req.id = uuid.uuid4().hex
                prefix_req.sampling_params = copy.copy(prefix_req.sampling_params)
                prefix_req.sampling_params.max_new_tokens = 0
                prefix_reqs.append(prefix_req)
                prefix_req_ids.add(prefix_req.id)

            batch_output = []

            if len(prefix_reqs) > 0:
                _, prefix_output = self.scheduler.infer(prefix_reqs)
                batch_output.extend(prefix_output)

            gen_reqs = []
            for request in batch_requests:
                for _ in range(request.sampling_params.n):
                    gen_req = copy.copy(request)
                    gen_req.id = uuid.uuid4().hex
                    request_ids[gen_req.id] = request.id

                    gen_reqs.append(gen_req)

            finished, gen_output = self.scheduler.infer(gen_reqs)
            batch_output.extend(gen_output)

            get_next_requests = not fetched_requests
            if len(batch_output) > 0:
                get_next_requests = True

                for output in batch_output:
                    if output[0] in prefix_req_ids:
                        prefix_req_ids.remove(output[0])

                        continue

                    req_id = request_ids[output[0]]
                    del request_ids[output[0]]

                    if req_id not in output_buffer:
                        output_buffer[req_id] = [output]

                    else:
                        output_buffer[req_id].append(output)

                    if len(output_buffer[req_id]) != output_params[req_id].n:
                        continue

                    await self.trajectory_queue.put_output.call_one(
                        Response(
                            id=req_id,
                            responses=[output[2] for output in output_buffer[req_id]],
                            response_logprobs=[
                                output[3] for output in output_buffer[req_id]
                            ],
                        )
                    )

                    del output_buffer[req_id]
                    del output_params[req_id]


class EnvironmentWorker:
    def __init__(self, rollout_generator, environment_queue):
        self.rollout_generator = rollout_generator
        self.environment_queue = environment_queue.slice(gpus=0)

    def interact(
        self,
        rollouts: list[Rollout],
        generated_samples: list[list[list[int]]],
        generated_logprobs: list[list[list[float]]],
    ) -> tuple[list[Rollout], list[Rollout]]:
        unfinished_rollouts = []

        finished_rollouts = self.rollout_generator.build_rollout(
            rollouts, generated_samples, generated_logprobs
        )

        return finished_rollouts, unfinished_rollouts

    async def run(self):
        while True:
            input = await self.environment_queue.get_input.call_one()

            if isinstance(input, FinishedGeneration):
                break

            rollouts, samples, logprobs = input

            finished_rollouts, unfinished_rollouts = self.interact(
                rollouts, samples, logprobs
            )

            await self.environment_queue.put_output.call_one(
                (finished_rollouts, unfinished_rollouts)
            )


class RolloutWorker:
    def __init__(
        self,
        dataloader,
        request_builder,
        trajectory_queue,
        environment_queue,
        replay_buffers,
        batch_size,
        max_iter,
        n_generators,
    ):
        self.n_generators = n_generators

        self.trajectory_queue = trajectory_queue.slice(gpus=0)
        self.environment_queue = environment_queue.slice(gpus=0)

        self.replay_buffers = replay_buffers

        self.buffer_ranks = list(range(self.replay_buffers.size()))

        self.batch_size = batch_size
        self.max_iter = max_iter
        self.dataloader = dataloader
        self.request_builder = request_builder

    async def run(self):
        loader = infinite_loader(self.dataloader)

        rollout_buffer: list[Rollout] = []
        n_sent_requests = 0
        n_generated_samples = 0
        n_target_samples = self.batch_size * self.max_iter
        buffer_idx = 0
        rollouts_map: dict[str, Rollout] = {}

        while True:
            if len(rollout_buffer) < self.batch_size:
                batch = next(loader)
                requests = self.request_builder(batch)
                rollout_buffer.extend(requests)

            rollouts_send = rollout_buffer[: self.batch_size - n_sent_requests]

            for rollout in rollouts_send:
                rollouts_map[rollout.id] = rollout

            requests_send = [
                rollout.to_inference_request() for rollout in rollouts_send
            ]

            if len(requests_send) > 0:
                for chunk in chunk_list(requests_send, self.n_generators):
                    await self.trajectory_queue.put_input.call_one(chunk)

            rollout_buffer = rollout_buffer[len(rollouts_send) :]
            n_sent_requests += len(rollouts_send)

            output = await self.trajectory_queue.get_output.call_one(nowait=True)

            if not isinstance(output, Empty):
                n_sent_requests -= 1

                rollout = rollouts_map[output.id]
                generated_rollouts = [rollout]
                generated_samples = [output.responses]
                generated_logprobs = [output.response_logprobs]

                await self.environment_queue.put_input.call_one(
                    (generated_rollouts, generated_samples, generated_logprobs)
                )

            environment_output = await self.environment_queue.get_output.call_one(
                nowait=True
            )

            if isinstance(environment_output, Empty):
                continue

            finished_rollouts, unfinished_rollouts = environment_output

            rollout_buffer = unfinished_rollouts + rollout_buffer

            rollout_by_group = {}
            for rollout in finished_rollouts:
                if rollout.id not in rollout_by_group:
                    rollout_by_group[rollout.id] = []

                rollout_by_group[rollout.id].append(rollout)

            for rollout in rollout_by_group.values():
                buffer = self.replay_buffers.slice(gpus=self.buffer_ranks[buffer_idx])
                await buffer.put.call_one(rollout)

                n_generated_samples += 1

                if rollout[0].id in rollouts_map:
                    del rollouts_map[rollout[0].id]

            buffer_idx = (buffer_idx + 1) % len(self.buffer_ranks)

            if n_generated_samples >= n_target_samples:
                break

        for rank in self.buffer_ranks:
            buffer = self.replay_buffers.slice(gpus=rank)
            await buffer.put.call_one(FinishedGeneration())

        for _ in range(self.n_generators):
            await self.trajectory_queue.put_input.call_one(FinishedGeneration())
            await self.environment_queue.put_input.call_one(FinishedGeneration())


class ReplayBuffer(Actor):
    def __init__(self):
        self.buffer = []
        self.put_condition = asyncio.Condition()
        self.latest_index = 0

    @endpoint
    async def put(self, sample):
        async with self.put_condition:
            self.buffer.append(sample)
            self.put_condition.notify_all()

    @endpoint
    async def sample(self):
        async with self.put_condition:
            await self.put_condition.wait_for(
                lambda: self.latest_index < len(self.buffer)
            )

        sample = self.buffer[self.latest_index]
        self.latest_index += 1

        return [sample]


class BidirectionalQueue(Actor):
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

    @endpoint
    async def initialize(self, name):
        wandb.init(project="halite-ppo", name=name)

    @endpoint
    async def put_input(self, sample):
        await self.input_queue.put(sample)

    @endpoint
    async def get_input(self, nowait=False):
        if nowait:
            try:
                return self.input_queue.get_nowait()

            except asyncio.QueueEmpty:
                return Empty()

        else:
            return await self.input_queue.get()

    @endpoint
    async def put_output(self, sample):
        await self.output_queue.put(sample)

    @endpoint
    async def get_output(self, nowait=False):
        if nowait:
            try:
                return self.output_queue.get_nowait()

            except asyncio.QueueEmpty:
                return Empty()

        else:
            return await self.output_queue.get()
