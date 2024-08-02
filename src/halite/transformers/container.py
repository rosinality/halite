from torch import nn

# from meshfn.nn.parallel.strategy import Module


class ModuleList(nn.ModuleList):
    @staticmethod
    def parallelize():
        return {"#": {"_strategy": Module}}
