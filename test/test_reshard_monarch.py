import asyncio
import os

from monarch.actor import Actor, current_rank, current_size, endpoint, proc_mesh
import torch
from torch import nn
from torch.distributed.tensor import DTensor

from halite.distributed import (
    all_reduce_mean,
    find_free_port,
    find_local_ip,
    load_checkpoint,
)
from halite.distributed.process_group import stateless_init_process_group
from halite.parallel import ParallelDims
from halite.parallel.fsdp import apply_fsdp
from halite.projects.common.rollout_monarch import get_peer_ranks


def find_multiple_free_ports(n_ports):
    ports = set()

    while len(ports) < n_ports:
        ports.add(find_free_port())

    return list(ports)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleDict()

        for i in range(10):
            self.blocks[str(i)] = nn.Linear(1, 10)

    def initialize(self):
        offset = torch.arange(10).float() / 10

        for i, block in self.blocks.items():
            i = float(i)

            block.weight.data.fill_(i)
            block.weight.data.add_(offset.unsqueeze(-1))
            block.bias.data.fill_(i)
            block.bias.data.add_(offset)


class TensorSpec:
    def __init__(
        self, name, shape, dtype, placements=None, full_shape=None, stride=None
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.placements = placements
        self.full_shape = full_shape
        self.stride = stride

    @property
    def is_dtensor(self):
        return self.placements is not None


class Trainer(Actor):
    def __init__(self, generator):
        self.rank = current_rank()["gpus"]
        self.world_size = current_size()["gpus"]

        self.generator = generator

    @endpoint
    def initialize(
        self, local_ranks, master_addr, master_port, world_size, rank_offset
    ):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        self.device = torch.device("cuda")
        self.device_id = local_ranks[self.rank]
        self.world_size = world_size

        self.global_pg = stateless_init_process_group(
            master_addr, master_port, self.rank, world_size, self.device_id
        )

        self.pdims = ParallelDims(
            dp_replicate=1,
            dp_shard=4,
            tp=1,
            pp=1,
            local_rank=self.device_id,
            world_size=self.world_size,
        )
        self.pdims.initialize(set_device_id=False)
        self.mesh = self.pdims.build_mesh("cuda")

        model = Model()
        model.initialize()

        self.param_dtype = torch.bfloat16
        self.model = apply_fsdp(
            model, self.mesh, param_dtype=self.param_dtype, reduce_dtype=torch.float32
        )

        self.generator_ranks = get_peer_ranks(
            self.rank, self.world_size, self.generator.size()
        )

        self.generators = []
        for gen_rank in self.generator_ranks:
            self.generators.append(self.generator.slice(gpus=gen_rank))

        self.rank_offset = rank_offset

        print(
            "trainer initialized",
            self.rank,
            self.rank,
            world_size,
            self.device_id,
        )

    @endpoint
    async def nccl_test(self):
        tensor = torch.randn(10, device=self.device)
        self.global_pg.broadcast(tensor, 0)

    @endpoint
    async def send_dtensor_plans(self):
        sd = self.model.state_dict()

        self.plans = []

        with torch.no_grad():
            for k, v in sd.items():
                v = v.to(self.param_dtype)

                if isinstance(v, DTensor):
                    self.plans.append(
                        TensorSpec(
                            k,
                            v.to_local().shape,
                            v.dtype,
                            v.placements,
                            v.shape,
                            v.stride(),
                        )
                    )

                else:
                    self.plans.append(TensorSpec(k, v.shape, v.dtype))

        for generator in self.generators:
            await generator.set_dtensor_plans.call_one(self.plans)

    @endpoint
    async def sync_weights(self):
        sd = self.model.state_dict()

        for generator in self.generators:
            await generator.sync_weights.call_one()

        self.global_pg.group_start()

        for tensor_plan in self.plans:
            name = tensor_plan.name
            tensor = sd[name]

            if isinstance(tensor, DTensor):
                tensor = tensor.to_local()

            for generator_rank in self.generator_ranks:
                self.global_pg.send(tensor, generator_rank + self.rank_offset)

        self.global_pg.group_end()


class Generator(Actor):
    def __init__(self):
        self.rank = current_rank()["gpus"]
        self.world_size = current_size()["gpus"]

        self.update_condition = asyncio.Event()

    @endpoint
    def initialize(
        self, local_ranks, master_addr, master_port, world_size, rank_offset
    ):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        self.device = torch.device("cuda")
        self.device_id = local_ranks[self.rank]
        self.global_size = world_size

        self.global_pg = stateless_init_process_group(
            master_addr,
            master_port,
            self.rank + rank_offset,
            world_size,
            self.device_id,
        )
        torch.cuda.set_device(self.device_id)

        self.pdims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            tp=1,
            pp=1,
            local_rank=self.device_id,
            world_size=self.world_size,
        )

        self.pdims.initialize(set_device_id=False)
        self.mesh = self.pdims.build_mesh("cuda")

        pdims2 = ParallelDims(
            dp_replicate=1,
            dp_shard=4,
            tp=1,
            pp=1,
            local_rank=self.device_id,
            world_size=self.world_size,
        )
        self.mesh_train = pdims2.build_mesh("cuda")

        self.model = Model().to(torch.bfloat16)

        # self.model = apply_fsdp(
        #     Model(), self.mesh, param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        # )

        self.rank_offset = rank_offset

        print(
            "generator initialized",
            self.rank,
            self.rank + rank_offset,
            world_size,
            self.device_id,
        )

    @endpoint
    async def nccl_test(self):
        tensor = torch.randn(10, device=self.device)
        self.global_pg.broadcast(tensor, 0)

        print(tensor)

    @endpoint
    async def set_dtensor_plans(self, plans):
        self.dtensor_plans = plans

    @endpoint
    async def sync_weights(self):
        self.update_condition.set()

    def sync_weights_recv(self):
        received_tensors = {}

        self.global_pg.group_start()

        for tensor_spec in self.dtensor_plans:
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
                # received_state_dict[name] = tensor
                received_state_dict[name] = tensor.redistribute(
                    self.mesh, target_tensor.placements
                )

        self.model.load_state_dict(received_state_dict)

        if self.rank == 0:
            log = []

            for name, tensor in received_state_dict.items():
                log.append(f"{name} {tensor.reshape(-1)}")

            print("\n".join(log))

    @endpoint
    async def run(self):
        while True:
            if self.update_condition.is_set():
                self.sync_weights_recv()
                self.update_condition.clear()

                break

            await asyncio.sleep(0.1)


async def main():
    local_ip = find_local_ip()
    free_ports = find_multiple_free_ports(3)

    generator_mesh = await proc_mesh(
        gpus=4,
        env={
            "MASTER_ADDR": local_ip,
            "MASTER_PORT": str(free_ports[0]),
            # "NCCL_DEBUG": "INFO",
        },
    )
    trainer_mesh = await proc_mesh(
        gpus=4,
        env={"MASTER_ADDR": local_ip, "MASTER_PORT": str(free_ports[1])},
    )

    await trainer_mesh.logging_option(True, None)
    await generator_mesh.logging_option(True, None)

    generator = await generator_mesh.spawn("generator", Generator)
    trainer = await trainer_mesh.spawn("trainer", Trainer, generator)

    local_ranks = [rank % 8 for rank in range(4 + 4)]

    await asyncio.gather(
        generator.initialize.call(local_ranks[4:], local_ip, free_ports[2], 8, 4),
        trainer.initialize.call(local_ranks[:4], local_ip, free_ports[2], 8, 4),
    )

    await trainer.send_dtensor_plans.call()

    await asyncio.gather(trainer.sync_weights.call(), generator.run.call())


if __name__ == "__main__":
    asyncio.run(main())
