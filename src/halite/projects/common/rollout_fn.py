from typing import Any

import torch


class Detokenize:
    def __init__(
        self,
        tokenizer: Any,
        keys=("output_ids",),
        output_keys=("output_texts",),
        **tokenizer_kwargs,
    ):
        self.tokenizer = tokenizer
        self.keys = keys
        self.output_keys = output_keys
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            if key not in data:
                continue

            target_key = self.output_keys[i] if self.output_keys else key

            detokenized = self.tokenizer.decode(data[key], **self.tokenizer_kwargs)

            data[target_key] = detokenized

        return data


class Compose:
    def __init__(self, *postprocessors):
        self.postprocessors = postprocessors

    def __call__(self, rewards_dict, data, types):
        for postprocessor in self.postprocessors:
            rewards, rewards_dict = postprocessor(rewards_dict, data, types)

        return rewards, rewards_dict


class WeightedSum:
    def __init__(self, output_key, /, **weights):
        self.output_key = output_key
        self.weights = weights

    def __call__(self, rewards_dict, data, types):
        rewards = 0

        for key, weight in self.weights.items():
            rewards += rewards_dict[key] * weight

        rewards_dict = {**rewards_dict, self.output_key: rewards}

        return rewards, rewards_dict


class ToTokenReward:
    def __init__(self, sample_key, reward_key, output_key):
        self.sample_key = sample_key
        self.reward_key = reward_key
        self.output_key = output_key

    def __call__(self, rewards_dict, data, types):
        samples = [row[self.sample_key] for row in data]
        max_length = max(len(sample) for sample in samples)
        rewards = rewards_dict[self.reward_key]
        last_index = torch.tensor([len(sample) - 1 for sample in samples])

        token_rewards = rewards.new_zeros(len(samples), max_length)
        token_rewards[torch.arange(len(samples)), last_index] = rewards

        rewards_dict = {**rewards_dict, self.output_key: token_rewards}

        return token_rewards, rewards_dict
