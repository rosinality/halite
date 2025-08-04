import asyncio
import uuid

from monarch.actor import Actor, current_rank, current_size, endpoint, proc_mesh
import torch
from torch.nn import functional as F

from halite.projects.common.rollout import (
    RolloutGenerator,
    Handler,
    Rollout,
    RewardRegistry,
)
from halite.projects.common.rollout_fn import Compose, ToTokenReward
from halite.projects.common.rollout_monarch import (
    EnvironmentWorker,
    GeneratorWorker,
    RolloutWorker,
    FinishedGeneration,
    get_peer_ranks,
    ReplayBuffer,
    BidirectionalQueue,
)
from halite.projects.ppo.trainer import PPOTrainer, compute_grpo_advantage
from halite.projects.ppo.variants import PPOActorLoss


class Request:
    def __init__(self, id, input_ids, sampling_params):
        self.id = id
        self.input_ids = input_ids
        self.sampling_params = sampling_params


class SchedulerMock:
    def __init__(self):
        self.counter = 0
        self.requests = []
        self.req_counter = [0, 0]

        self.outputs = [
            [[6, 7, 8], [6, 7, 8, 9, 10], [1, 2]],
            [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10]],
        ]

    def build_batch_request(self, requests):
        from halite.transformers.infer.engine.batch import SamplingParams

        return [
            Request(req.id, req.input_ids, SamplingParams(**req.sampling_params))
            for req in requests
        ]

    def infer(self, requests):
        if len(requests) > 0 and requests[0].sampling_params.max_new_tokens == 0:
            return False, []

        for request in requests:
            self.requests.append(request)

        if len(self.requests) > self.counter:
            req = self.requests[self.counter]

            if req.input_ids == [1, 2, 3, 4, 5]:
                output_ids = self.outputs[0][self.req_counter[0] % 3]
                self.req_counter[0] += 1

            else:
                output_ids = self.outputs[1][self.req_counter[1] % 3]
                self.req_counter[1] += 1

            result = [(req.id, req.input_ids, output_ids)]

            self.counter += 1

        else:
            result = []

        finished = False
        if self.counter >= 6:
            finished = True

        return finished, result


class ModelRunnerMock:
    def __init__(self):
        self.model = None


class InferenceEngineMock:
    def __init__(self):
        self.scheduler = SchedulerMock()
        self.model_runner = ModelRunnerMock()


class RewardMock:
    def __init__(self):
        self.counter = 0

    def __call__(self, data):
        rewards = []

        for _ in range(len(data)):
            rewards.append(self.counter + 1)
            self.counter += 1

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


class BasicReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

    @endpoint
    async def sample(self):
        batch = []
        sample_count = 0

        while True:
            async with self.put_condition:
                await self.put_condition.wait_for(
                    lambda: self.latest_index < len(self.buffer)
                )

            sample = self.buffer[self.latest_index]

            if isinstance(sample, FinishedGeneration):
                return sample

            self.latest_index += 1

            batch.extend(sample)

            sample_count += 1

            if sample_count == self.batch_size:
                self.buffer = self.buffer[self.latest_index :]
                self.latest_index = 0

                break

        return batch


class Trainer(Actor):
    def __init__(self, replay_buffers, generator):
        rank = current_rank()

        self.replay_buffers = replay_buffers.slice(**rank)
        self.generator = generator

        self.rank = rank["gpus"]

        self.generator_ranks = get_peer_ranks(
            self.rank, current_size()["gpus"], generator.size()
        )

        self.actor = ActorMock()
        self.trainer = PPOTrainer(
            self.actor,
            compute_grpo_advantage,
            PPOActorLoss(
                clip_low=0.2,
                clip_high=0.2,
                pg_loss_agg="token-sum",
                pg_loss_max_tokens=4096,
            ),
        )

        self.batch_size = 6

    @endpoint
    async def run(self):
        step = 0

        finished = False

        while True:
            if finished:
                break

            step += 1

            rollouts = await self.replay_buffers.sample.call_one()

            if isinstance(rollouts, FinishedGeneration):
                break

            rollouts = self.trainer.compute_advantage(rollouts)
            pg_loss = self.trainer.compute_actor_loss(rollouts)

            pg_loss.pg_loss.backward()

            self.check_rollout(rollouts)

        for rank in self.generator_ranks:
            generator = self.generator.slice(gpus=rank)
            await generator.finished_training.call_one()

    def check_rollout(self, rollouts):
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
            rollouts.rewards,
            rewards_target,
        ), f"{rollouts.rewards=}, {rewards_target=}"

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
            rollouts.batch.input_ids == input_ids_target,
        ), f"{rollouts.batch.input_ids=}, {input_ids_target=}"

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
            rollouts.advantages,
            advantages_target,
        ), f"{rollouts.advantages=}, {advantages_target=}"

        logits = (
            torch.arange(
                rollouts.batch.input_ids.numel() * 11, dtype=torch.float32
            ).reshape(*rollouts.batch.input_ids.shape, 11)
            / 100
        )

        logits = logits.detach()
        logits.requires_grad = True

        logits_slice = logits[:, rollouts.batch.response_ids.shape[-1] - 1 : -1]

        log_probs = log_probs_from_logits(
            logits_slice, rollouts.batch.response_ids, -1, rollouts.batch.temperatures
        )

        pg_loss_test = -rollouts.advantages * (log_probs - log_probs.detach())
        batch_size = pg_loss_test.shape[0]
        response_mask = rollouts.batch.attention_mask[
            :, -rollouts.batch.response_ids.shape[-1] :
        ]
        pg_loss_test = torch.sum(pg_loss_test * response_mask) / (4096 * batch_size)
        pg_loss_test.backward()

        assert torch.allclose(self.actor.logits.grad, logits.grad)


class Generator(Actor):
    def __init__(self, trajectory_queue):
        inference_engine = InferenceEngineMock()

        self.generator = GeneratorWorker(
            inference_engine=inference_engine,
            trajectory_queue=trajectory_queue,
        )

    @endpoint
    async def update_state_dict(self, state_dict_buffers):
        await self.generator.update_state_dict(state_dict_buffers)

    @endpoint
    async def finished_training(self):
        await self.generator.finished_training()

    @endpoint
    async def run(self):
        await self.generator.run()

        await self.generator.wait_for_finish()


def dataloader():
    for _ in range(10):
        yield True


def request_builder(_):
    return [
        Rollout(
            id=uuid.uuid4().hex,
            input_ids=[1, 2, 3, 4, 5],
            type="math",
            sampling_params={"max_new_tokens": 512, "n": 3},
            state={"input_text": None},
        ),
        Rollout(
            id=uuid.uuid4().hex,
            input_ids=[6, 7, 8],
            type="arithmetic",
            sampling_params={"max_new_tokens": 512, "n": 3},
            state={"input_text": None},
        ),
    ]


class RolloutManager(Actor):
    def __init__(
        self, trajectory_queue, environment_queue, replay_buffers, n_generators
    ):
        self.rollout = RolloutWorker(
            dataloader=dataloader(),
            request_builder=request_builder,
            trajectory_queue=trajectory_queue,
            environment_queue=environment_queue,
            replay_buffers=replay_buffers,
            batch_size=2,
            max_iter=1,
            n_generators=n_generators,
        )

        self.trajectory_queue = trajectory_queue

    @endpoint
    async def run(self):
        await self.rollout.run()

        await self.trajectory_queue.put_input.call_one(FinishedGeneration())


class Environment(Actor):
    def __init__(self, environment_queue):
        reward_handler = Handler(
            "mock",
            RewardMock(),
            args=("output_ids",),
            targets="*",
        )

        rollout_generator = RolloutGenerator(
            None,
            RewardRegistry(
                reward_handler,
                postprocess=Compose(
                    ToTokenReward("output_ids", "mock", "token_rewards")
                ),
            ),
        )

        self.environment = EnvironmentWorker(
            rollout_generator,
            environment_queue,
        )

    @endpoint
    async def run(self):
        await self.environment.run()


async def main():
    generator_mesh = await proc_mesh(gpus=1)
    trainer_mesh = await proc_mesh(gpus=1)
    rollout_mesh = await proc_mesh(gpus=1)

    batch_size = 2
    replay_buffers = await trainer_mesh.spawn(
        "replay_buffers", BasicReplayBuffer, batch_size
    )
    trajectory_queue = await generator_mesh.spawn(
        "trajectory_queue", BidirectionalQueue
    )
    generator = await generator_mesh.spawn(
        "generator",
        Generator,
        trajectory_queue,
    )
    environment_queue = await rollout_mesh.spawn(
        "environment_queue", BidirectionalQueue
    )
    rollout_manager = await rollout_mesh.spawn(
        "rollout",
        RolloutManager,
        trajectory_queue,
        environment_queue,
        replay_buffers,
        1,
    )
    environment = await generator_mesh.spawn(
        "environment", Environment, environment_queue
    )
    trainer = await trainer_mesh.spawn("trainer", Trainer, replay_buffers, generator)

    await asyncio.gather(
        generator.run.call(),
        environment.run.call(),
        trainer.run.call(),
        rollout_manager.run.call(),
    )

    await asyncio.sleep(0.1)

    await asyncio.gather(
        generator_mesh.stop(),
        trainer_mesh.stop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
