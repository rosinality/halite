import os

import torch
from torch import distributed as dist


class ParallelDims:
    def __init__(self, dp_replicate, dp_shard, tp, pp, world_size=None):
        if world_size is None:
            world_size = int(os.getenv("WORLD_SIZE", "1"))

        for d in (dp_replicate, tp, pp):
            assert d >= 1, "parallel dims must be greater than 0, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must be -1 or >= 1"

        dp = dp_replicate * dp_shard

        if dp < 0:
            dp = world_size // (tp * pp)
            dp_shard = dp // dp_replicate

        self.dp_replicate = dp_replicate
        self.dp_shard = dp_shard
        self.tp = tp
        self.pp = pp

        self.world_size = world_size

    @property
    def is_primary(self):
        return dist.get_rank() == 0

    def initialize():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        backend = "gloo"

        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.device(local_rank))
            backend = "nccl"

        dist.init_process_group(backend)

    def build_mesh(self, device):
        dims = []
        names = []

        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.tp],
            ["pp", "dp_replicate", "dp_shard", "tp"],
            strict=True,
        ):
            if d > 1:
                dims.append(d)

                if (name == "dp_replicate" and self.dp_shard == 1) or (
                    name == "dp_shard" and self.dp_replicate == 1
                ):
                    names.append("dp")

                else:
                    names.append(name)

        names = tuple(names)

        self.mesh = dist.device_mesh.init_device_mesh(
            device, dims, mesh_dim_names=names
        )

        if self.dp_replicate > 1 and self.dp_shard > 1:
            self.mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")

        return self.mesh

    @property
    def dp(self):
        return self.dp_replicate * self.dp_shard

    @property
    def dp_mesh_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1
