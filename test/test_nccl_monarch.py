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


class Generator(Actor):
    def __init__(self):
        self.rank = current_rank()["gpus"]
        self.world_size = current_size()["gpus"]

        self.update_condition = asyncio.Condition()
        self._updating = False

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
            self.device_id,
            world_size,
            self.device_id,
        )

        torch.cuda.set_device(self.device_id)

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


async def main():
    local_ip = find_local_ip()
    free_ports = find_multiple_free_ports(3)

    generator_mesh = await proc_mesh(
        gpus=2,
        env={
            "MASTER_ADDR": local_ip,
            "MASTER_PORT": str(free_ports[0]),
            # "NCCL_DEBUG": "INFO",
        },
    )

    trainer_mesh = await proc_mesh(
        gpus=2,
        env={
            "MASTER_ADDR": local_ip,
            "MASTER_PORT": str(free_ports[1]),
            # "NCCL_DEBUG": "INFO",
        },
    )

    await generator_mesh.logging_option(True, None)

    generator = await generator_mesh.spawn("generator", Generator)
    trainer = await trainer_mesh.spawn("trainer", Generator)

    local_ranks = [rank % 8 for rank in range(4)]

    await asyncio.gather(
        generator.initialize.call(local_ranks[:2], local_ip, free_ports[2], 4, 0),
        trainer.initialize.call(local_ranks[2:], local_ip, free_ports[2], 4, 2),
    )

    await asyncio.gather(
        generator.nccl_test.call(),
        trainer.nccl_test.call(),
    )

    # await asyncio.gather(trainer.sync_weights.call(), generator.sync_weights.call())

    print("finished")


if __name__ == "__main__":
    asyncio.run(main())
