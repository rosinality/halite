import torch

from halite.data.record import Record


class SFTSample:
    def __init__(
        self,
        prompt_key,
        response_key,
        tokenizer,
        input_key="input",
        target_key="target",
        prompt_map_fn=None,
        response_map_fn=None,
        prompt_tokenizer_kwargs=None,
        response_tokenizer_kwargs=None,
        ignore_index=-100,
    ):
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.target_key = target_key
        self.prompt_map_fn = prompt_map_fn
        self.response_map_fn = response_map_fn
        self.prompt_tokenizer_kwargs = (
            prompt_tokenizer_kwargs if prompt_tokenizer_kwargs is not None else {}
        )
        self.response_tokenizer_kwargs = (
            response_tokenizer_kwargs if response_tokenizer_kwargs is not None else {}
        )
        self.ignore_index = ignore_index

    def __call__(self, iterator):
        for record in iterator:
            yield self.process(record)

    def process(self, record):
        prompt = record[self.prompt_key]
        response = record[self.response_key]

        if self.prompt_map_fn is not None:
            prompt = self.prompt_map_fn(prompt)

        if self.response_map_fn is not None:
            response = self.response_map_fn(response)

        prompt_tokens = self.tokenizer.encode(
            prompt,
            **self.prompt_tokenizer_kwargs,
        )

        response_tokens = self.tokenizer.encode(
            response,
            **self.response_tokenizer_kwargs,
        )

        all_tokens = torch.as_tensor(prompt_tokens + response_tokens, dtype=torch.int64)
        target = torch.cat((all_tokens, torch.tensor([self.ignore_index])))
        target[: len(prompt_tokens)] = self.ignore_index
        target[-1] = self.ignore_index  # eos handling
        record[self.input_key] = all_tokens
        record[self.target_key] = target[1:]

        return record


def length_to_offsets(lengths: list[int]) -> torch.Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """

    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)

    return offsets


def collate_offsets(offsets: list[torch.Tensor], max_len: int) -> torch.Tensor:
    batch = torch.zeros((len(offsets), max_len), dtype=torch.int32)

    for i, offset in enumerate(offsets):
        batch[i, : len(offset)] = offset
        batch[i, len(offset) :] = offset[-1]

    return batch


class SFTSequencePacking:
    def __init__(
        self,
        length,
        input_key="input",
        target_key="target",
        input_pad=0,
        target_pad=-100,
        use_position_ids=False,
        use_document_offsets=False,
        use_rest_of_long_sequence=False,
    ):
        self.length = length
        self.input_key = input_key
        self.target_key = target_key
        self.input_pad = input_pad
        self.target_pad = target_pad

        self.use_position_ids = use_position_ids
        self.use_document_offsets = use_document_offsets
        self.use_rest_of_long_sequence = use_rest_of_long_sequence

        self._input_ids = []
        self._target_ids = []
        self._dataset_ids = []
        self._sample_ids = []

        self._input_ids = []
        self._target_ids = []
        self._dataset_ids = []
        self._sample_ids = []

    def load_state_dict(self, state_dict):
        self._input_ids = state_dict["input_ids"]
        self._target_ids = state_dict["target_ids"]
        self._dataset_ids = state_dict["dataset_ids"]
        self._sample_ids = state_dict["sample_ids"]

    def state_dict(self):
        return {
            "input_ids": self._input_ids.copy(),
            "target_ids": self._target_ids.copy(),
            "dataset_ids": self._dataset_ids.copy(),
            "sample_ids": self._sample_ids.copy(),
        }

    def _pack_record(
        self,
        record,
        packed_input_ids,
        packed_target_ids,
        packed_dataset_ids,
        packed_sample_ids,
    ):
        packed_record = {
            **record,
            self.input_key: torch.cat(packed_input_ids),
            self.target_key: torch.cat(packed_target_ids),
            "_tokenized_keys_": (self.input_key, self.target_key),
        }

        if self.use_document_offsets:
            document_offsets = length_to_offsets(
                [len(sample) for sample in packed_input_ids]
            )
            packed_record["document_offsets"] = document_offsets

        if self.use_position_ids:
            position_ids = torch.cat(
                [torch.arange(len(sample)) for sample in packed_input_ids]
            )
            packed_record["position_ids"] = position_ids

        packed_record["_meta_"].update(
            {
                "dataset_ids": packed_dataset_ids,
                "sample_ids": packed_sample_ids,
            }
        )

        return Record(packed_record)

    def get_padded_slice(self, input_lists, pad_value, *ids_lists):
        buffer = []
        buffer_ids = []
        for _ in range(len(ids_lists)):
            buffer_ids.append([])

        current_len = 0

        for i, input in enumerate(input_lists):
            dtype = input.dtype
            input_len = len(input)

            if current_len + input_len <= self.length:
                buffer.append(input)
                current_len += input_len

                for j, ids in enumerate(ids_lists):
                    buffer_ids[j].append(ids[i])

            elif current_len > 0:
                pad = self.length - current_len
                buffer.append(torch.full((pad,), pad_value, dtype=dtype))

                for j, ids in enumerate(ids_lists):
                    buffer_ids[j].append(ids[i])

                return (
                    buffer,
                    input_lists[i:],
                    buffer_ids,
                    [ids[i:] for ids in ids_lists],
                )

            else:
                buffer.append(input[: self.length])

                for j, ids in enumerate(ids_lists):
                    buffer_ids[j].append(ids[i])

                if self.use_rest_of_long_sequence:
                    return (
                        buffer,
                        [input[self.length :]] + input_lists[i + 1 :],
                        buffer_ids,
                        [[ids[i]] + ids[i + 1 :] for ids in ids_lists],
                    )

                else:
                    return (
                        buffer,
                        input_lists[i + 1 :],
                        buffer_ids,
                        [ids[i + 1 :] for ids in ids_lists],
                    )

        return None, input_lists, [None for _ in range(len(ids_lists))], ids_lists

    def __call__(self, iterator):
        for record in iterator:
            self._input_ids.append(record[self.input_key])
            self._target_ids.append(record[self.target_key])
            self._dataset_ids.append(record["_meta_"]["dataset_id"])
            self._sample_ids.append(record["_meta_"]["sample_id"])

            while True:
                (
                    packed_input_ids,
                    self._input_ids,
                    (packed_dataset_ids, packed_sample_ids),
                    (self._dataset_ids, self._sample_ids),
                ) = self.get_padded_slice(
                    self._input_ids, self.input_pad, self._dataset_ids, self._sample_ids
                )

                if packed_input_ids is None:
                    break

                packed_target_ids, self._target_ids, _, _ = self.get_padded_slice(
                    self._target_ids, self.target_pad
                )

                packed_record = self._pack_record(
                    record,
                    packed_input_ids,
                    packed_target_ids,
                    packed_dataset_ids,
                    packed_sample_ids,
                )

                yield packed_record

            if len(self._input_ids) == 0:
                self._dataset_ids = []
                self._sample_ids = []

        if len(self._input_ids) > 0:
            current_len = sum(len(input) for input in self._input_ids)
            pad_len = self.length - current_len

            self._input_ids.append(
                torch.full((pad_len,), self.input_pad, dtype=torch.int64)
            )
            self._target_ids.append(
                torch.full((pad_len,), self.target_pad, dtype=torch.int64)
            )

            packed_record = self._pack_record(
                record,
                self._input_ids,
                self._target_ids,
                self._dataset_ids,
                self._sample_ids,
            )

            self._input_ids = []
            self._target_ids = []
            self._dataset_ids = []
            self._sample_ids = []

            yield packed_record
