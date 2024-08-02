import orjson
import torch

from halite.data.record import Record


class ReadRawText:
    def __init__(self, key="text"):
        self.key = key
    
    def __call__(self, iterator):
        for record in iterator:
            features = record.data.decode('utf-8')
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


class Tokenize:
    def __init__(self, tokenizer, keys=("text",), **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.keys = keys
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, iterator):
        for features in iterator:
            for key in self.keys:
                features[key] = self.tokenizer.encode(
                    features[key], **self.tokenizer_kwargs
                )

            yield features


class SequencePacking:
    def __init__(self, length, key="text"):
        self.length = length
        self.key = key

    def __call__(self, iterator):
        packed_ids = []
        packed_dataset_ids = []
        packed_sample_ids = []

        for input_record in iterator:
            start = 0
            while start < len(input_record[self.key]):
                rem_data = input_record[self.key][start:]

                if len(packed_ids) + len(rem_data) < self.length:
                    packed_ids.extend(rem_data)  # use rest of example, move-on
                    packed_dataset_ids.append(input_record._meta_["dataset_id"])
                    packed_sample_ids.append(input_record._meta_["sample_id"])

                    break

                else:
                    take = self.length - len(packed_ids)
                    packed_ids.extend(rem_data[:take])
                    packed_dataset_ids.append(input_record._meta_["dataset_id"])
                    packed_sample_ids.append(input_record._meta_["sample_id"])

                    packed_record = {**input_record, self.key: packed_ids}
                    packed_record["_meta_"].update(
                        {
                            "dataset_ids": packed_dataset_ids,
                            "sample_ids": packed_sample_ids,
                        }
                    )
                    packed_record = Record(packed_record)

                    yield packed_record

                    start += take
                    packed_ids = []
                    packed_dataset_ids = []
                    packed_sample_ids = []

                    # Drop remainder for simplicity.
                    # We lose the rest of the example on restore.


class AutoregressiveSample:
    def __init__(self, key='text'):
        self.key = key
        
    def __call__(self, iterator):
        for record in iterator:
            record['input'] = record[self.key][:-1]
            record['target'] = record[self.key][1:]
            
            yield record

class Collator:
    def __init__(self, keys=("text",)):
        self.keys = keys

    def __call__(self, batch):
        collated = Record()
        collated._meta_ = []

        for key in self.keys:
            collated[key] = torch.stack(
                [torch.as_tensor(record[key]) for record in batch], 0
            )

        for record in batch:
            for key, val in record.items():
                if key in self.keys:
                    continue

                if key not in collated:
                    collated[key] = []

                collated[key].append(val)

        return collated
