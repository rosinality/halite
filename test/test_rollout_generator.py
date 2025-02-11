import argparse

from slickconf import load_config, instantiate
import torch.distributed.checkpoint as dcp
import torch

from halite.data.record import Record
from halite.transformers.infer import InferenceEngine, ModelConfig
from halite.data.tokenizers.llama3 import Llama3Tokenizer

from halite.projects.common.rollout import (
    RolloutGenerator,
    Handler,
    Detokenize,
    RewardRegistry,
)


class LengthPenalty:
    def __init__(self, eos_text):
        self.eos_text = eos_text

    def __call__(self, data):
        rewards = []

        for row in data:
            row_rewards = []

            for sample in row:
                try:
                    start = sample.index(self.eos_text)
                    row_rewards.append(1 / (start + 1))

                except ValueError:
                    row_rewards.append(-1)

            rewards.append(row_rewards)

        return torch.tensor(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--tokenizer", type=str)
    args = parser.parse_args()

    conf = load_config(args.conf)

    with torch.device("meta"):
        model = instantiate(instantiate(conf.model_infer)(conf.model))

    model.to_empty(device="cuda")

    state_dict = {"model": model.state_dict()}
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=args.checkpoint,
    )

    tokenizer = Llama3Tokenizer(args.tokenizer)

    inference_engine = InferenceEngine(
        model.to(device="cuda", dtype=torch.bfloat16),
        tokenizer,
        ModelConfig(
            n_heads=conf.model_conf.n_heads,
            n_key_value_heads=conf.model_conf.n_key_value_heads,
            head_dim=conf.model_conf.head_dim,
            n_layers=conf.model_conf.n_layers,
            context_len=conf.model_conf.context_len,
        ),
    )

    length_penalty = Handler(
        "length_penalty",
        LengthPenalty("<|end_of_text|>"),
        args=("output_texts",),
        targets="*",
    )

    rollout_generator = RolloutGenerator(
        inference_engine,
        RewardRegistry(length_penalty),
        preprocessors=[Detokenize(tokenizer)],
    )

    rollout_generator.initialize()

    question1 = "\int_{-\infty}^{\infty} e^{-x^2} \,dx = "
    question2 = "125 + 235 = "

    rollout = rollout_generator.generate(
        [
            [
                question1,
                {"max_new_tokens": 1024, "n": 64},
            ],
            [
                question2,
                {"max_new_tokens": 1024, "n": 64},
            ],
        ],
        types=["math", "arithmetic"],
        batch=Record({"input_text": [question1, question2]}),
    )

    print(rollout)
