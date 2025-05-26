import itertools
from dataclasses import dataclass
from typing import Callable

import orjson
from slickconf import Instance
import torch

from halite.data.record import Record


class ReadRawText:
    def __init__(self, key="text"):
        self.key = key

    def __call__(self, iterator):
        for record in iterator:
            features = record.data.decode("utf-8")
            del record.data
            record[self.key] = features

            yield record


class ParseFeatures:
    def __call__(self, iterator):
        for record in iterator:
            features = orjson.loads(record.data)
            del record.data
            record.update(features)

            yield record


class SelectFeatures:
    def __init__(self, keys=("text",)):
        self.keys = keys

    def __call__(self, iterator):
        for record in iterator:
            for key in self.keys:
                record[key] = record.data[key]

            del record.data

            yield record


class Map:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, iterator):
        for record in iterator:
            mapped = self.fn(record)

            for key, val in mapped.items():
                record[key] = val

            yield record


class ApplyTemplate:
    def __init__(self, key, template):
        self.key = key

        from halite.projects.common.template import get_render_fn

        self.template = get_render_fn(template)

    def __call__(self, iterator):
        for record in iterator:
            record[self.key] = self.template(record)

            yield record


class Tokenize:
    def __init__(self, tokenizer, keys=("text",), output_keys=None, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.keys = keys
        self.output_keys = output_keys

        if self.output_keys is not None:
            assert len(self.output_keys) == len(self.keys)

        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, iterator):
        for features in iterator:
            for i, key in enumerate(self.keys):
                target_key = self.output_keys[i] if self.output_keys else key

                features[target_key] = self.tokenizer.encode(
                    features[key], **self.tokenizer_kwargs
                )

            yield features


@dataclass
class SequencePackingState:
    start_token: int


class SequencePacking:
    def __init__(self, length, key="text"):
        self.length = length
        self.key = key

        self._start = 0
        self._packed_ids = []
        self._packed_dataset_ids = []
        self._packed_sample_ids = []
        self._tokens = []

    def load_state_dict(self, state_dict):
        self._tokens = state_dict["tokens"]
        self._packed_dataset_ids = state_dict["packed_dataset_ids"]
        self._packed_sample_ids = state_dict["packed_sample_ids"]

    def state_dict(self):
        return {
            "tokens": self._tokens.copy(),
            "packed_dataset_ids": self._packed_dataset_ids.copy(),
            "packed_sample_ids": self._packed_sample_ids.copy(),
        }

    def __call__(self, iterator):
        for input_record in iterator:
            self._tokens.extend(input_record[self.key])

            self._packed_dataset_ids.append(input_record["_meta_"]["dataset_id"])
            self._packed_sample_ids.append(input_record["_meta_"]["sample_id"])

            while len(self._tokens) >= self.length:
                packed_ids = self._tokens[: self.length]
                self._tokens = self._tokens[self.length :]

                packed_record = {**input_record, self.key: packed_ids}
                packed_record["_meta_"].update(
                    {
                        "dataset_ids": self._packed_dataset_ids,
                        "sample_ids": self._packed_sample_ids,
                    }
                )
                packed_record = Record(packed_record)

                self._packed_dataset_ids = self._packed_dataset_ids[-1:]
                self._packed_sample_ids = self._packed_sample_ids[-1:]

                yield packed_record

            if len(self._tokens) == 0:
                self._packed_dataset_ids = []
                self._packed_sample_ids = []


class AutoregressiveSample:
    def __init__(self, key="text"):
        self.key = key

    def __call__(self, iterator):
        for record in iterator:
            record["input"] = record[self.key][:-1]
            record["target"] = record[self.key][1:]

            yield record


def collate_list(batch):
    return batch


class Collator:
    def __init__(
        self,
        keys=("text",),
        skip_except_keys=False,
        collate_fns: dict[str, Callable] = None,
    ):
        self.keys = keys
        self.skip_except_keys = skip_except_keys

        self.collate_fns = {}
        if collate_fns is not None:
            self.collate_fns = collate_fns

    def __call__(self, batch):
        collated = Record()
        collated._meta_["batch_keys"] = self.keys

        if hasattr(batch[0], "_tokenized_keys_"):
            collated._meta_["tokenized_keys"] = batch[0]._tokenized_keys_

        for key in self.keys:
            if key in self.collate_fns:
                collated[key] = self.collate_fns[key]([record[key] for record in batch])

            else:
                collated[key] = torch.stack(
                    [torch.as_tensor(record[key]) for record in batch], 0
                )

        if self.skip_except_keys:
            return collated

        collated._meta_["samples_meta"] = []

        for record in batch:
            for key, val in record.items():
                if key in self.keys or key in ("_batch_keys", "_tokenized_keys_"):
                    continue

                if key == "_meta_":
                    collated._meta_["samples_meta"].append(val)

                    continue

                if key not in collated:
                    collated[key] = []

                collated[key].append(val)

        return collated
