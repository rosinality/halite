import argparse
import asyncio
import uuid

from slickconf import load_config, instantiate
from monarch.actor import Actor, current_rank, current_size, endpoint, proc_mesh
import torch
from torch import distributed as dist
import torch.distributed.checkpoint as dcp

from halite.projects.common.rollout import (
    RolloutGenerator,
    Rollout,
    Handler,
    RewardRegistry,
)
from halite.projects.common.rollout_fn import Compose, Detokenize, ToTokenReward
from halite.projects.common.rollout_monarch import (
    EnvironmentWorker,
    GeneratorWorker,
    FinishedGeneration,
    get_peer_ranks,
    BidirectionalQueue,
    ReplayBuffer,
    RolloutWorker,
)
from halite.transformers.infer import InferenceEngine, ModelConfig


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

    @endpoint
    async def run(self):
        step = 0

        while True:
            step += 1

            sample = await self.replay_buffers.sample.call_one()

            if isinstance(sample, FinishedGeneration):
                break

            print(self.rank, [rollout.get_field("input_text") for rollout in sample])

        for rank in self.generator_ranks:
            generator = self.generator.slice(gpus=rank)
            await generator.finished_training.call_one()


class Generator(Actor):
    def __init__(self, conf, checkpoint, tokenizer, trajectory_queue):
        self.rank = current_rank()

        torch.cuda.set_device(self.rank["gpus"])

        with torch.device("meta"):
            model = instantiate(instantiate(conf.model_infer)(conf.model))

        model.to_empty(device="cuda")

        state_dict = {"model": model.state_dict()}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint,
        )

        tokenizer = instantiate(conf.tokenizer, tokenizer)

        inference_engine = InferenceEngine(
            model.to(device="cuda", dtype=torch.bfloat16),
            tokenizer,
            ModelConfig(
                n_heads=conf.model_conf.n_heads,
                n_key_value_heads=conf.model_conf.n_key_value_heads,
                head_dim=conf.model_conf.head_dim,
                n_layers=conf.model_conf.n_layers,
                context_len=conf.model_conf.context_len,
                gpu_id=self.rank["gpus"],
            ),
        )

        self.generator = GeneratorWorker(
            inference_engine=inference_engine,
            trajectory_queue=trajectory_queue,
        )

        self.training_condition = asyncio.Condition()
        self.training_finished = False

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
    for i in range(16):
        yield True


class RequestBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        rollouts = []

        for i in range(16):
            question = f"{i + 1} + {i + 1} = "

            rollouts.append(
                Rollout(
                    id=uuid.uuid4().hex,
                    input_ids=self.tokenizer.encode(question),
                    type="math",
                    sampling_params={"max_new_tokens": 512, "n": 4},
                    state={"input_text": question},
                )
            )

        return rollouts


class RolloutManager(Actor):
    def __init__(
        self,
        conf,
        tokenizer,
        trajectory_queue,
        environment_queue,
        replay_buffers,
        n_generators,
    ):
        tokenizer = instantiate(conf.tokenizer, tokenizer)

        self.rollout = RolloutWorker(
            dataloader=dataloader(),
            request_builder=RequestBuilder(tokenizer),
            trajectory_queue=trajectory_queue,
            environment_queue=environment_queue,
            replay_buffers=replay_buffers,
            batch_size=16,
            max_iter=1,
            n_generators=n_generators,
        )

        self.trajectory_queue = trajectory_queue

    @endpoint
    async def run(self):
        await self.rollout.run()


class Environment(Actor):
    def __init__(self, conf, tokenizer, environment_queue):
        tokenizer = instantiate(conf.tokenizer, tokenizer)

        length_penalty = Handler(
            "length_penalty",
            LengthPenalty("\\boxed"),
            args=("output_texts",),
            targets="*",
        )

        rollout_generator = RolloutGenerator(
            None,
            RewardRegistry(
                length_penalty,
                postprocess=Compose(
                    ToTokenReward("output_ids", "length_penalty", "token_rewards")
                ),
            ),
            preprocessors=[Detokenize(tokenizer)],
        )

        self.environment = EnvironmentWorker(
            rollout_generator,
            environment_queue,
        )

    @endpoint
    async def run(self):
        await self.environment.run()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--tokenizer", type=str)
    args = parser.parse_args()

    conf = load_config(args.conf)

    generator_mesh = await proc_mesh(gpus=4)
    trainer_mesh = await proc_mesh(gpus=4)
    rollout_mesh = await proc_mesh(gpus=1)

    replay_buffers = await trainer_mesh.spawn("replay_buffers", BasicReplayBuffer, 1)
    trajectory_queue = await rollout_mesh.spawn("trajectory_queue", BidirectionalQueue)
    environment_queue = await rollout_mesh.spawn(
        "environment_queue", BidirectionalQueue
    )
    generator = await generator_mesh.spawn(
        "generator",
        Generator,
        conf,
        args.checkpoint,
        args.tokenizer,
        trajectory_queue,
    )
    rollout_manager = await rollout_mesh.spawn(
        "rollout",
        RolloutManager,
        conf,
        args.tokenizer,
        trajectory_queue,
        environment_queue,
        replay_buffers,
        4,
    )
    environment = await generator_mesh.spawn(
        "environment", Environment, conf, args.tokenizer, environment_queue
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
