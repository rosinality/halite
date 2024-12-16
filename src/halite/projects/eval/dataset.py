def hf_dataset(path, name, split="test"):
    from datasets import load_dataset

    return load_dataset(path, name, split=split, trust_remote_code=True)


def first_n(dataset, n):
    def sample():
        return dataset[:n]

    return sample
