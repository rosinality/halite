import argparse

from slickconf import load_config, instantiate
import torch.distributed.checkpoint as dcp
import torch

from halite.transformers.infer import InferenceEngine, ModelConfig

from halite.projects.common.rollout import (
    RolloutGenerator,
    Handler,
    Request,
    RewardRegistry,
)
from halite.projects.common.rollout_fn import Compose, Detokenize, ToTokenReward


class LengthPenalty:
    def __init__(self, eos_text):
        self.eos_text = eos_text

    def __call__(self, data):
        rewards = []

        for sample in data:
            try:
                start = sample.index(self.eos_text)
                row_rewards = 1 / (start + 1)

            except ValueError:
                row_rewards = -1

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

    tokenizer = instantiate(conf.tokenizer, args.tokenizer)

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
        LengthPenalty("\\boxed"),
        args=("output_texts",),
        targets="*",
    )

    rollout_generator = RolloutGenerator(
        inference_engine,
        RewardRegistry(
            length_penalty,
            postprocess=Compose(
                ToTokenReward("output_ids", "length_penalty", "token_rewards")
            ),
        ),
        preprocessors=[Detokenize(tokenizer)],
    )

    rollout_generator.initialize()

    question1 = "\int_{-\infty}^{\infty} e^{-x^2} \,dx = "
    question2 = "125 + 235 = "

    rollout = rollout_generator.generate(
        [
            Request(
                question1,
                "math",
                {"max_new_tokens": 512, "n": 4},
                {"input_text": question1},
            ),
            Request(
                question2,
                "arithmetic",
                {"max_new_tokens": 512, "n": 4},
                {"input_text": question2},
            ),
        ],
    )

    print(rollout)
