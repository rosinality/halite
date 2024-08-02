import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

except ImportError:
    pass


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, z_loss=0, fast=False):
        super().__init__()

        self.ignore_index = ignore_index
        self.z_loss = z_loss
        self.fast = fast

    def fast_forward(self, input, target):
        losses, z_losses = cross_entropy_loss(
            input.view(-1, input.shape[-1]),
            target.view(-1),
            lse_square_scale=self.z_loss,
            ignore_index=self.ignore_index,
        )

        loss = losses.mean()
        loss_dict = {"cross entropy": loss.detach()}

        if self.z_loss > 0:
            z_loss = z_losses.mean()
            loss_dict["z-loss"] = z_loss.detach()
            loss_dict["cross entropy"] = (losses.detach() - z_losses.detach()).mean()

        return loss, loss_dict

    def forward(self, input, target):
        if self.fast:
            return self.fast_forward(input, target)

        shift_input = input.contiguous().to(torch.float32)
        shift_target = target.contiguous().to(input.device)

        cross_entropy = F.cross_entropy(
            shift_input.view(-1, input.shape[-1]),
            shift_target.view(-1),
            ignore_index=self.ignore_index,
        )

        loss = cross_entropy
        loss_dict = {"cross entropy": cross_entropy.detach()}

        if self.z_loss > 0:
            z_loss = torch.logsumexp(shift_input, dim=-1).square().mean()
            loss_dict["z-loss"] = z_loss.detach()

            loss = loss + self.z_loss * z_loss

        return loss, loss_dict
