import torch

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

except ImportError:
    pass

from halite.nn.entropy import entropy_from_logits
from halite.projects.ppo.types import Batch
from halite.transformers.attention import (
    unpad_input,
    unpad_params,
    pad_input,
)


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)


class UnpaddedModel(ModelWrapper):
    def __init__(
        self,
        model,
        pad_id: int = -1,
        calc_log_probs: bool = True,
        calc_entropy: bool = True,
    ):
        super().__init__(model)

        self.pad_id = pad_id

        self.calc_log_probs = calc_log_probs
        self.calc_entropy = calc_entropy

    def __call__(self, batch: Batch):
        unpad = unpad_params(batch.attention_mask)

        input_ids = unpad_input(batch.input_ids, unpad.indices_q).transpose(0, 1)
        position_ids = unpad_input(batch.position_ids, unpad.indices_q).transpose(0, 1)

        out = self.model(input_ids, position_ids=position_ids, unpad_params=unpad)

        logits = out.logits.reshape(-1, out.logits.shape[-1])

        if not self.calc_log_probs and not self.calc_entropy:
            return pad_input(logits, unpad.indices_q, unpad.batch, unpad.seqlen)

        if isinstance(batch.temperatures, torch.Tensor):
            logits = logits / torch.as_tensor(
                batch.temperatures, device=logits.device, dtype=logits.dtype
            ).reshape(-1, 1)
            logit_scale = 1.0

        else:
            logit_scale = 1 / batch.temperatures

        target = torch.roll(input_ids, shifts=-1, dims=1)

        log_probs, _ = cross_entropy_loss(
            logits,
            target.reshape(-1),
            logit_scale=logit_scale,
            ignore_index=self.pad_id,
        )

        log_probs = pad_input(
            -log_probs.unsqueeze(-1), unpad.indices_q, unpad.batch, unpad.seqlen
        ).squeeze(-1)

        response_len = batch.response_ids.shape[-1]

        entropy = None

        if self.calc_entropy:
            entropy = entropy_from_logits(logits, logit_scale=logit_scale)
            entropy = pad_input(
                entropy.unsqueeze(-1), unpad.indices_q, unpad.batch, unpad.seqlen
            ).squeeze(-1)
            entropy = entropy[:, -response_len - 1 : -1]

        return log_probs[:, -response_len - 1 : -1], entropy
