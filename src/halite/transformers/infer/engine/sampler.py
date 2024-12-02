import torch
from flashinfer.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)

from halite.logging import logger


class Sampler:
    def __init__(self, use_nan_detection=True, backend="flashinfer"):
        self.use_nan_detection = use_nan_detection
        self.backend = backend

    def forward(self, logits, sampling_params):
        logits = logits.next_token_logits
        logits = logits.contiguous()

        if self.use_nan_detection:
            is_nan = torch.isnan(logits)

            if torch.any(is_nan):
                logger.warning("Detected NaN in logits")
                logits = torch.where(is_nan, torch.full_like(logits, -1e5), logits)

        if sampling_params.is_all_greedy:
            batch_next_token_ids = torch.argmax(logits, -1)

        else:
            logits.div_(sampling_params.temperatures)
            probs = torch.softmax(logits, -1)
            logits = None
            del logits

            if self.backend == "flashinfer":
                max_top_k_round, batch_size = 32, probs.shape[0]
                uniform_samples = torch.rand(
                    max_top_k_round, batch_size, device=probs.device
                )

                if sampling_params.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_params.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_params.top_ps)
                    batch_next_token_ids, success = min_p_sampling_from_probs(
                        probs, uniform_samples, sampling_params.min_ps
                    )

                else:
                    batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
                        probs,
                        uniform_samples,
                        sampling_params.top_ks,
                        sampling_params.top_ps,
                        filter_apply_order="joint",
                    )

                if not torch.all(success):
                    logger.warning("Detected errors in sampling")
                    batch_next_token_ids = torch.zeros_like(batch_next_token_ids)

        return batch_next_token_ids.to(torch.int32)
