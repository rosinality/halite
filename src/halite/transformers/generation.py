import torch
from torch import distributed as dist


class IndexTracker:
    def __init__(self, n_batch):
        self.map_index = {}

        for i in range(n_batch):
            self.map_index[i] = i

    def update(self, finished):
        indexes = []
        map_index = {}

        for i, flag in enumerate(finished.tolist()):
            if flag == 1:
                indexes.append(self.map_index[i])
                map_index[i] = len(indexes) - 1

        updated = len(self.map_index) != len(map_index)

        self.map_index = map_index

        return (
            torch.as_tensor(indexes, device=finished.device, dtype=torch.int64),
            updated,
        )


class LogitsProcessor:
    additional_kwargs: bool = False


class StoppingCriteriaList(list):
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        return any(criteria(input_ids, scores) for criteria in self)


class MaxLengthCriteria:
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        return input_ids.shape[-1] >= self.max_length


class LogitsProcessorList(list):
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs):
        for processor in self:
            if processor.additional_kwargs:
                scores = processor(input_ids, scores, **kwargs)

            else:
                scores = processor(input_ids, scores)

        return scores


class TemperatureLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature: float):
        if not temperature > 0:
            raise ValueError(f"`temperature` should be > 0, but got {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        return scores / self.temperature


class TopPLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        top_p: float,
        filter_value: float = float("-inf"),
        min_tokens_to_keep: int = 1,
    ):
        if top_p < 0 or top_p > 1:
            raise ValueError("`top_p` has to be in [0, 1], but got {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cum_p = sorted_logits.softmax(-1).cumsum(-1)

        filter_indices = cum_p <= (1 - self.top_p)

        if self.min_tokens_keep > 1:
            filter_indices[..., -self.min_tokens_keep :] = 0

        indices_to_remove = filter_indices.scatter(1, sorted_indices, filter_indices)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores


class TopKLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        top_k: int,
        filter_value: float = float("-inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` should be strictly positive integer, but got {top_k}"
            )

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        top_k = min(self.top_k, scores.shape[-1])
        filter_indices = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(filter_indices, self.filter_value)

        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(
                f"`penalty` should be strictly positive float, but got {penalty}"
            )

        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)

        return scores


class GenerationMixin:
    def generate(
        self,
        input_ids=None,
        mode="sample",
        max_length=None,
        max_new_tokens=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        temperature=1,
        top_k=0,
        top_p=1,
        repetition_penalty=1,
        use_cache=True,
        synced_gpus=False,
        num_return_sequences=1,
        device="cpu",
        streamer=None,
        **kwargs,
    ):
        if input_ids is not None:
            if input_ids.ndim < 2:
                input_ids = input_ids.unsqueeze(-1)

            device = input_ids.device

        attention_mask = kwargs.get("attention_mask", None)
        input_ids, attention_mask = self.prepare_inputs(
            input_ids, attention_mask, bos_token_id, device
        )
        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            **kwargs,
        }

        if streamer is not None:
            streamer.put(input_ids.cpu())

        if max_new_tokens is not None:
            max_new_tokens = input_ids.shape[1] + max_new_tokens

        if max_new_tokens is not None:
            if max_length is None:
                max_length = max_new_tokens

            else:
                max_length = min(max_length, max_new_tokens)

        stopping_criteria = self.get_stopping_criteria(max_length)
        logits_processor = self.get_logits_processor(
            temperature, top_k, top_p, repetition_penalty
        )

        if num_return_sequences > 1:
            input_ids, model_kwargs = self.expand_inputs(
                input_ids, model_kwargs, num_return_sequences
            )

        if mode == "sample":
            return self.sample(
                input_ids,
                eos_token_id,
                pad_token_id,
                stopping_criteria,
                logits_processor,
                synced_gpus,
                streamer,
                greedy=temperature == 0,
                **model_kwargs,
            )

    def sample(
        self,
        input_ids,
        eos_token_id=None,
        pad_token_id=None,
        stopping_criteria=None,
        logits_processor=None,
        synced_gpus=False,
        streamer=None,
        greedy=False,
        **model_kwargs,
    ):
        eos_token_id_tensor = (
            input_ids.new_tensor(eos_token_id).view(-1)
            if eos_token_id is not None
            else None
        )

        unfinished = input_ids.new_ones(input_ids.shape[0], dtype=torch.int64)
        this_peer_finished = False
        scores = None

        slice_index = None
        slice_updated = False
        if eos_token_id is not None:
            tracker = IndexTracker(input_ids.shape[0])

        while True:
            if synced_gpus:
                this_peer_finished = input_ids.new_zeros(1)
                dist.all_reduce(this_peer_finished, op=dist.ReduceOp.SUM)

                if this_peer_finished.item() == 0:
                    break

            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                **model_kwargs,
                slice_index=slice_index if slice_updated else None,
                unfinished=unfinished,
            )
            outs = self(**model_inputs)

            if synced_gpus and this_peer_finished:
                continue

            next_logits = outs.logits[:, -1]
            next_scores = logits_processor(
                input_ids[unfinished.to(torch.bool)], next_logits
            )

            if greedy:
                next_token = next_scores.argmax(1)

            else:
                probs = torch.softmax(next_scores, -1)
                next_token = torch.multinomial(probs, 1).squeeze(1)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "if `eos_token_id` is defined, `pad_token_id` should be defined"
                    )

                next_token_scatter = torch.zeros_like(unfinished)
                next_token_scatter.scatter_(
                    0, unfinished.nonzero().squeeze(), next_token
                )
                next_token = next_token_scatter

                next_token = next_token * unfinished + pad_token_id * (1 - unfinished)

            input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), -1)

            if streamer is not None:
                streamer.put(next_token.cpu())

            model_kwargs = self.update_model_kwargs_for_generation(outs, model_kwargs)

            if eos_token_id_tensor is not None:
                unfinished = unfinished.mul(
                    next_token.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(0)
                )

                slice_index, slice_updated = tracker.update(unfinished)

                if unfinished.max() == 0:
                    this_peer_finished = True

            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        return input_ids

    def expand_inputs(self, input_ids=None, model_kwargs=None, expand_factor: int = 1):
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_factor, dim=0)

        new_kwargs = {}

        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                new_kwargs[k] = v.repeat_interleave(expand_factor, dim=0)

            else:
                new_kwargs[k] = v

        return input_ids, new_kwargs

    def prepare_inputs(
        self, input_ids=None, attention_mask=None, bos_token_id=None, device="cpu"
    ):
        if input_ids is not None:
            return input_ids, attention_mask

        batch_size = 1

        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.int64, device=device
        )

        attention_mask = torch.ones(batch_size, 1, dtype=torch.int64, device=device)

        return input_ids, attention_mask

    def update_model_kwargs_for_generation(self, outs, model_kwargs):
        if outs.cache is not None:
            model_kwargs["cache"] = outs.cache

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                (attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)),
                -1,
            )

        return model_kwargs

    def get_logits_processor(
        self, temperature=1, top_k=0, top_p=1, repetition_penalty=1
    ):
        processors = LogitsProcessorList()

        if repetition_penalty != 1:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        if temperature != 1 and temperature != 0:
            processors.append(TemperatureLogitsProcessor(temperature))

        if top_k > 0:
            processors.append(TopKLogitsProcessor(top_k))

        if top_p < 1:
            processors.append(TopPLogitsProcessor(top_p))

        return processors

    def get_stopping_criteria(self, max_length=None):
        criteria = StoppingCriteriaList()

        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length))

        return criteria
