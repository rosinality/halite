import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

except ImportError:
    pass


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, z_loss=0, fast=False, micro_average=False):
        super().__init__()

        self.ignore_index = ignore_index
        self.z_loss = z_loss
        self.fast = fast
        self.micro_average = micro_average

    def fast_forward(self, input, target):
        losses, z_losses = cross_entropy_loss(
            input.view(-1, input.shape[-1]),
            target.view(-1),
            lse_square_scale=self.z_loss,
            ignore_index=self.ignore_index,
        )

        if self.micro_average:
            n_targets = (target != self.ignore_index).sum().clamp(min=1)
            loss = losses.sum() / n_targets

        else:
            loss = losses.mean()

        loss_dict = {"cross entropy": loss.detach()}

        if self.z_loss > 0:
            if self.micro_average:
                z_loss = z_losses.sum() / n_targets

            else:
                z_loss = z_losses.mean()

            loss_dict["z-loss"] = z_loss.detach()

            if self.micro_average:
                loss_dict["cross entropy"] = (
                    losses.detach() - z_losses.detach()
                ).sum() / n_targets

            else:
                loss_dict["cross entropy"] = (
                    losses.detach() - z_losses.detach()
                ).mean()

        if self.micro_average:
            loss_dict["n_targets"] = n_targets

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
            reduction="sum" if self.micro_average else "mean",
        )

        if self.micro_average:
            n_targets = (target != self.ignore_index).sum().clamp(min=1)
            cross_entropy = cross_entropy / n_targets

        loss = cross_entropy
        loss_dict = {"cross entropy": cross_entropy.detach()}

        if self.z_loss > 0:
            if self.micro_average:
                z_loss = torch.logsumexp(shift_input, dim=-1).square().sum() / n_targets
            else:
                z_loss = torch.logsumexp(shift_input, dim=-1).square().mean()

            loss_dict["z-loss"] = z_loss.detach()

            loss = loss + self.z_loss * z_loss

        if self.micro_average:
            loss_dict["n_targets"] = n_targets

        return loss, loss_dict
