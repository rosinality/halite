import torch
from torch import distributed as dist


class DataManager:
    def __init__(self, loader, mesh, device="cuda"):
        self.loader = loader
        self.mesh = mesh
        self.finished = torch.tensor(0, dtype=torch.float32, device=device)

    def __iter__(self):
        self.loader_iter = iter(self.loader)

        return self

    def __next__(self):
        try:
            batch = next(self.loader_iter)

        except StopIteration:
            finished = True

        else:
            finished = False

        self.finished.fill_(float(finished))
        dist.all_reduce(
            self.finished, group=self.mesh.get_group("dp"), op=dist.ReduceOp.MAX
        )
        finished = self.finished.item() > 0

        if finished:
            raise StopIteration

        return batch
