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


def collate_offsets(offsets: list[torch.Tensor]) -> torch.Tensor:
    max_len = max([len(offset) for offset in offsets])
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

    def _split_long_sequence(self, record):
        input_ids = record[self.input_key]
        target_ids = record[self.target_key]
        splitted = False
        while len(input_ids) > self.length:
            packed_input_ids = [input_ids[: self.length]]
            packed_target_ids = [target_ids[: self.length]]
            packed_dataset_ids = [record._meta_["dataset_id"]]
            packed_sample_ids = [record._meta_["sample_id"]]

            yield (
                (
                    packed_input_ids,
                    packed_target_ids,
                    packed_dataset_ids,
                    packed_sample_ids,
                    self.length,
                ),
                True,
            )

            input_ids = input_ids[self.length :]
            target_ids = target_ids[self.length :]
            splitted = True

        if not splitted or (splitted and self.use_rest_of_long_sequence):
            yield (
                (
                    [input_ids],
                    [target_ids],
                    [record._meta_["dataset_id"]],
                    [record._meta_["sample_id"]],
                    len(input_ids),
                ),
                False,
            )

    def __call__(self, iterator):
        packed_input_ids = []
        packed_len = 0
        packed_target_ids = []
        packed_dataset_ids = []
        packed_sample_ids = []

        for record in iterator:
            current_len = packed_len + len(record[self.input_key])

            dtype = record[self.input_key].dtype

            if current_len <= self.length:
                packed_input_ids.append(record[self.input_key])
                packed_len += len(record[self.input_key])
                packed_target_ids.append(record[self.target_key])
                packed_dataset_ids.append(record._meta_["dataset_id"])
                packed_sample_ids.append(record._meta_["sample_id"])

                continue

            if packed_len > 0:
                pad = self.length - packed_len
                packed_input_ids.append(torch.full((pad,), self.input_pad, dtype=dtype))
                packed_target_ids.append(
                    torch.full((pad,), self.target_pad, dtype=dtype)
                )

                yield self._pack_record(
                    record,
                    packed_input_ids,
                    packed_target_ids,
                    packed_dataset_ids,
                    packed_sample_ids,
                )

            for (
                packed_input_ids,
                packed_target_ids,
                packed_dataset_ids,
                packed_sample_ids,
                packed_len,
            ), is_packed in self._split_long_sequence(record):
                if is_packed:
                    yield self._pack_record(
                        record,
                        packed_input_ids,
                        packed_target_ids,
                        packed_dataset_ids,
                        packed_sample_ids,
                    )
