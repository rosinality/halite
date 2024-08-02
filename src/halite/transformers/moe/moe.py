from torch import nn


class MoE(nn.Module):
    def __init__(self, router, token_dispatcher, experts):
        super().__init__()

        self.router = router
        self.token_dispatcher = token_dispatcher
        self.experts = experts

    def parallelize(self, ep_group):
        self.token_dispatcher.parallelize(ep_group)

    def forward(self, inputs):
        probs, indices, aux_losses = self.router(inputs)

        if self.token_dispatcher is None:
            output = self.experts(inputs, probs, indices)

            return output, aux_losses

        dispatched_inputs, permute_state = self.token_dispatcher.permute(
            inputs, probs, indices
        )

        output = self.token_dispatcher.unpermute(
            dispatched_inputs, probs, permute_state
        )

        return output, aux_losses
