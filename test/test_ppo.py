import uuid

import torch
from torch.nn import functional as F

from halite.projects.common.rollout import (
    RolloutGenerator,
    Handler,
    Rollout,
    RewardRegistry,
)
from halite.projects.common.rollout_fn import Compose, ToTokenReward
from halite.projects.ppo.trainer import PPOTrainer, compute_grpo_advantage
from halite.projects.ppo.variants import PPOActorLoss
from halite.transformers.infer.engine.engine import InferenceResult


class InferenceEngineMock:
    def initialize(self):
        pass

    def infer_batch(self, requests):
        return [
            InferenceResult(
                id=requests[0].id,
                input_ids=[1, 2, 3, 4, 5],
                output_ids=[[6, 7, 8], [6, 7, 8, 9, 10], [1, 2]],
            ),
            InferenceResult(
                id=requests[1].id,
                input_ids=[6, 7, 8],
                output_ids=[[1, 2, 3, 4, 5], [6, 7], [8, 9, 10]],
            ),
        ]


class RewardMock:
    def __call__(self, data):
        rewards = []

        for i in range(len(data)):
            rewards.append(i + 1)

        return torch.tensor(rewards, dtype=torch.float32)


def log_probs_from_logits(logits, labels, ignore_index, temperature=1.0):
    if isinstance(temperature, torch.Tensor) or temperature != 1.0:
        logits = logits / temperature.reshape(-1, 1, 1)

    return -F.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index
    )


def entropy_from_logits(logits, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature.reshape(-1, 1, 1)

    return -torch.sum(torch.softmax(logits, -1) * torch.log_softmax(logits, -1), -1)


class ActorMock:
    def __call__(self, data):
        response_len = data.response_ids.shape[-1]

        logits = (
            torch.arange(data.input_ids.numel() * 11, dtype=torch.float32).reshape(
                *data.input_ids.shape, 11
            )
            / 100
        )

        logits = logits.detach()
        logits.requires_grad = True
        self.logits = logits
        logits = logits[:, response_len - 1 : -1]

        return log_probs_from_logits(
            logits, data.response_ids, -1, data.temperatures
        ), entropy_from_logits(
            logits,
            data.temperatures,
        )


if __name__ == "__main__":
    inference_engine = InferenceEngineMock()

    reward_handler = Handler(
        "mock",
        RewardMock(),
        args=("output_ids",),
        targets="*",
    )

    rollout_generator = RolloutGenerator(
        inference_engine,
        RewardRegistry(
            reward_handler,
            postprocess=Compose(ToTokenReward("output_ids", "mock", "token_rewards")),
        ),
    )

    rollout_generator.initialize()

    question1 = "\int_{-\infty}^{\infty} e^{-x^2} \,dx = "
    question2 = "125 + 235 = "

    rollout = rollout_generator.generate(
        [
            Rollout(
                id=uuid.uuid4().hex,
                input_ids=[1, 2, 3, 4, 5],
                type="math",
                sampling_params={"max_new_tokens": 512, "n": 4},
                state={"input_text": question1},
            ),
            Rollout(
                id=uuid.uuid4().hex,
                input_ids=[6, 7, 8],
                type="arithmetic",
                sampling_params={"max_new_tokens": 512, "n": 4},
                state={"input_text": question2},
            ),
        ],
    )

    actor = ActorMock()
    trainer = PPOTrainer(
        actor,
        compute_grpo_advantage,
        PPOActorLoss(
            clip_low=0.2,
            clip_high=0.2,
            pg_loss_agg="token-sum",
            pg_loss_max_tokens=4096,
        ),
    )

    rollout = trainer.compute_advantage(rollout)
    pg_loss = trainer.compute_actor_loss(rollout)

    rewards_target = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2],
            [0, 3, 0, 0, 0],
            [0, 0, 0, 0, 4],
            [0, 5, 0, 0, 0],
            [0, 0, 6, 0, 0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        rollout.rewards,
        rewards_target,
    ), f"{rollout.rewards=}, {rewards_target=}"

    input_ids_target = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 1, 2, 0, 0, 0],
            [0, 0, 6, 7, 8, 1, 2, 3, 4, 5],
            [0, 0, 6, 7, 8, 6, 7, 0, 0, 0],
            [0, 0, 6, 7, 8, 8, 9, 10, 0, 0],
        ]
    )

    assert torch.all(
        rollout.batch.input_ids == input_ids_target,
    ), f"{rollout.batch.input_ids=}, {input_ids_target=}"

    advantages_target = torch.tensor(
        [
            [-1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        rollout.advantages,
        advantages_target,
    ), f"{rollout.advantages=}, {advantages_target=}"

    pg_loss.pg_loss.backward()

    logits = (
        torch.arange(rollout.batch.input_ids.numel() * 11, dtype=torch.float32).reshape(
            *rollout.batch.input_ids.shape, 11
        )
        / 100
    )

    logits = logits.detach()
    logits.requires_grad = True

    logits_slice = logits[:, rollout.batch.response_ids.shape[-1] - 1 : -1]

    log_probs = log_probs_from_logits(
        logits_slice, rollout.batch.response_ids, -1, rollout.batch.temperatures
    )

    pg_loss_test = -rollout.advantages * (log_probs - log_probs.detach())
    batch_size = pg_loss_test.shape[0]
    response_mask = rollout.batch.attention_mask[
        :, -rollout.batch.response_ids.shape[-1] :
    ]
    pg_loss_test = torch.sum(pg_loss_test * response_mask) / (4096 * batch_size)
    pg_loss_test.backward()

    assert torch.allclose(actor.logits.grad, logits.grad)
