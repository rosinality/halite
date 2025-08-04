import asyncio
import gc
import os

from monarch.actor import Actor, current_rank, current_size, endpoint, proc_mesh
from monarch.rdma import RDMABuffer
from slickconf import instantiate, load_arg_config
import torch
from torch import nn

try:
    import wandb

except ImportError:
    wandb = None

from halite.distributed import (
    all_reduce_mean,
    find_free_port,
    find_local_ip,
    load_checkpoint,
)
from torch.distributed.tensor import DTensor
from halite.data.dataloader import DataLoader
from halite.data.dataset import build_dataset_from_spec, WeightedIterableDataset
from halite.logging import get_logger
from halite.parallel import ParallelDims
from halite.projects.ppo.config import PPOConfig
from halite.optim import group_parameters
from halite.transformers.infer import InferenceEngine, ModelConfig
from halite.transformers.infer.types import ServerConfig
from halite.utils import get_torch_dtype
from halite.projects.common.rollout_monarch import (
    EnvironmentWorker,
    GeneratorWorker,
    FinishedGeneration,
    get_peer_ranks,
    ReplayBuffer,
    BidirectionalQueue,
    RolloutWorker,
)


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def find_multiple_free_ports(n_ports):
    ports = set()

    while len(ports) < n_ports:
        ports.add(find_free_port())

    return list(ports)


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
                async with self.put_condition:
                    self.buffer = self.buffer[self.latest_index :]
                    self.latest_index = 0

                break

        return batch


def detach(input):
    if isinstance(input, torch.Tensor):
        return input.detach()

    return input


def build_rollout_report(conf, rollouts):
    report_texts = []

    n_samples = 0
    for i, rollout in enumerate(rollouts.rollouts):
        reward = rollout.rewards_dict[conf.reward_key].to("cpu")

        if i % conf.show_every_nth_sample != 0:
            continue

        input_text = rollout.get_field(conf.input_key)
        output_text = rollout.get_field(conf.output_key)

        report_text = f"# {i}\n[input]\n{input_text}\n\n[output]\n{output_text}\n\n[reward]\n{reward}"

        if conf.additional_keys is not None:
            for key in conf.additional_keys:
                report_text += f"\n\n[{key}]\n{rollout.get_field(key)}"

        report_texts.append(report_text)

        n_samples += 1

        if n_samples >= conf.show_n_samples:
            break

    return "\n\n".join(report_texts)


def build_model(conf, mesh, pdims, device, type="model", logger=None):
    if logger is not None:
        logger.info(f"building the {type}")

    with torch.device("meta"):
        model = instantiate(conf.model)

    model = model.to(dtype=get_torch_dtype(conf.model_conf.dtype))

    if conf.wrapper is not None:
        if logger is not None:
            logger.info("applying wrapper")

        model = instantiate(conf.wrapper)(model=model, mesh=mesh, parallel_dims=pdims)

    if conf.parallelize is not None:
        logger.info("applying parallelize")
        model = instantiate(conf.parallelize)(
            model=model, mesh=mesh, parallel_dims=pdims
        )

    if conf.wrapper is not None or conf.parallelize is not None:
        if logger is not None:
            logger.info(str(model))

    model.to_empty(device=device)

    if conf.checkpoint_path is not None:
        load_checkpoint(conf.checkpoint_path, model_parts=model)

    return model


class Trainer(Actor):
    def __init__(self, conf, replay_buffers, generator):
        self.conf = conf

        self.rank = current_rank()["gpus"]
        self.world_size = current_size()["gpus"]

        self.replay_buffers = replay_buffers
        self.generator = generator

    @endpoint
    async def initialize(self, local_ranks):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        device = torch.device("cuda")

        self.device_id = local_ranks[self.rank]

        pdims = ParallelDims(
            dp_replicate=self.conf.training.data_parallel_replicate,
            dp_shard=self.conf.training.data_parallel_shard,
            tp=self.conf.training.tensor_parallel,
            pp=self.conf.training.pipeline_parallel,
            local_rank=self.device_id,
            world_size=self.world_size,
        )
        pdims.initialize()
        mesh = pdims.build_mesh("cuda")

        self.parallel_dims = pdims

        self.logger = get_logger(mesh)

        self.replay_buffers = self.replay_buffers.slice(**current_rank())

        self.generator_ranks = get_peer_ranks(
            self.rank, self.world_size, self.generator.size()
        )

        if pdims.is_primary and wandb is not None:
            wandb.init(project="halite-ppo")

        actor = build_model(
            self.conf.ppo.actor, mesh, pdims, device, logger=self.logger
        )
        ref = None
        if self.conf.ppo.ref is not None:
            ref = build_model(
                self.conf.ppo.ref, mesh, pdims, device, logger=self.logger
            )

        critic = None
        if self.conf.ppo.critic is not None:
            critic = build_model(
                self.conf.ppo.critic, mesh, pdims, device, logger=self.logger
            )

        actor_train = actor
        if self.conf.ppo.actor_wrapper is not None:
            actor_train = instantiate(self.conf.ppo.actor_wrapper)(actor)

        self.trainer = instantiate(self.conf.ppo.trainer)(
            actor=actor_train,
            ref=ref,
            critic=critic,
            device=device,
        )

        self.optimizer = instantiate(self.conf.training.optimizer)(
            group_parameters(actor, weight_decay=self.conf.training.weight_decay)
        )
        self.scheduler = instantiate(self.conf.training.scheduler)(
            optimizer=self.optimizer,
        )

    @torch.no_grad()
    def _gather_state_dict(self):
        with torch.no_grad():
            state_dict = self.trainer.actor.state_dict()
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

            resharded = {}

            for k, tensor in state_dict.items():
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()

                resharded[k] = tensor

        return resharded

    def prepare_rdma_buffer(self):
        self._rdma_buffer = {
            k: RDMABuffer(v.view(torch.uint8).flatten())
            for k, v in self._gather_state_dict().items()
        }

    async def update_state_dict(self):
        state_dict = self._gather_state_dict()

        for k, tensor in state_dict.items():
            await self._rdma_buffer[k].write(tensor.view(torch.uint8).flatten())

    def show_gpu_memory(self, title):
        if self.parallel_dims.is_primary:
            free_memory, _ = torch.cuda.mem_get_info(self.device_id)
            free_memory = free_memory / (1 << 30)

            print(f"{title}: {free_memory:.2f}GB")

    @endpoint
    async def run(self):
        global_step = 0

        finished = False

        self.prepare_rdma_buffer()

        for gen_rank in self.generator_ranks:
            generator = self.generator.slice(gpus=gen_rank)

            await generator.register_rdma_buffer.call_one(self._rdma_buffer)

        while True:
            if finished:
                break

            rollouts = await self.replay_buffers.sample.call_one()

            if isinstance(rollouts, FinishedGeneration):
                break

            rollouts = self.trainer.compute_advantage(rollouts)
            rollout_batches = [rollouts]

            if self.conf.training.ppo_minibatch_size is not None:
                rollout_batches = rollouts.split(self.conf.training.ppo_minibatch_size)

            metrics = []
            for _ in range(self.conf.training.ppo_n_epochs):
                self.optimizer.zero_grad(set_to_none=True)

                for rollout_batch in rollout_batches:
                    rollout_microbatches = [rollout_batch]

                    if self.conf.training.ppo_microbatch_size is not None:
                        rollout_microbatches = rollout_batch.split(
                            self.conf.training.ppo_microbatch_size
                        )

                    n_microbatches = len(rollout_microbatches)

                    for rollout_microbatch in rollout_microbatches:
                        actor_loss = self.trainer.compute_actor_loss(rollout_microbatch)

                        (actor_loss.loss / n_microbatches).backward()

                        loss_dict = {
                            k: detach(v) for k, v in actor_loss.metric_dict().items()
                        }
                        loss_dict["sample/response_len/mean"] = (
                            rollout_microbatch.batch.response_len.float().mean()
                        )

                        metrics.append(loss_dict)

                grad_norm = None
                if self.conf.training.clip_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.trainer.actor.parameters(),
                        self.conf.training.clip_grad_norm,
                        foreach=True,
                    )

                self.scheduler.step()
                self.optimizer.step()

            await self.update_state_dict()

            for gen_rank in self.generator_ranks:
                generator = self.generator.slice(gpus=gen_rank)

                await generator.update_state_dict.call_one()

            if global_step == 0:
                self.logger.info(
                    f"ppo epochs: {self.conf.training.ppo_n_epochs}, minibatches: {len(rollout_batches)}, microbatches: {n_microbatches}"
                )

            if (
                self.conf.ppo.report is not None
                and global_step % self.conf.ppo.report.log_step == 0
            ):
                self.logger.info(build_rollout_report(self.conf.ppo.report, rollouts))

            if (
                global_step
                % min(self.conf.output.log_step, self.conf.output.wandb_log_step)
                == 0
            ):
                metrics_mean = {}
                for metric in metrics:
                    for k, v in metric.items():
                        if k not in metrics_mean:
                            metrics_mean[k] = []

                        metrics_mean[k].append(torch.as_tensor(v, dtype=torch.float32))

                metrics_mean = {
                    k: torch.stack(v).mean() for k, v in metrics_mean.items()
                }

                metrics_mean = {
                    k: all_reduce_mean(
                        v,
                        self.parallel_dims.mesh.get_group("dp"),
                        self.parallel_dims.dp,
                    ).item()
                    for k, v in metrics_mean.items()
                }

                lr = self.optimizer.param_groups[0]["lr"]

                loss_txt = "; ".join([f"{k}: {v:.5f}" for k, v in metrics_mean.items()])

                if global_step % self.conf.output.log_step == 0:
                    if grad_norm is None:
                        self.logger.info(
                            f"step: {global_step}; {loss_txt}; lr: {lr:.7f}"
                        )

                    else:
                        self.logger.info(
                            f"step: {global_step}; {loss_txt}; grad norm: {grad_norm.item():.3f}; lr: {lr:.7f}"
                        )

                if (
                    self.parallel_dims.is_primary
                    and wandb is not None
                    and global_step % self.conf.output.wandb_log_step == 0
                ):
                    report = {"actor/lr": lr, **metrics_mean}

                    if grad_norm is not None:
                        report["actor/grad_norm"] = grad_norm

                    wandb.log(report, step=global_step)

            global_step += 1

        for rank in self.generator_ranks:
            generator = self.generator.slice(gpus=rank)
            await generator.finished_training.call_one()

    @endpoint
    async def finalize(self):
        torch.distributed.destroy_process_group()


class Generator(Actor):
    def __init__(self, conf, trajectory_queue):
        self.conf = conf

        self.rank = current_rank()["gpus"]
        self.world_size = current_size()["gpus"]

        self.trajectory_queue = trajectory_queue

    @endpoint
    async def initialize(self, local_ranks):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        device = torch.device("cuda")

        pdims = ParallelDims(
            dp_replicate=self.conf.training.data_parallel_replicate,
            dp_shard=self.conf.training.data_parallel_shard,
            tp=self.conf.training.tensor_parallel,
            pp=self.conf.training.pipeline_parallel,
            local_rank=local_ranks[self.rank],
            world_size=self.world_size,
        )
        pdims.initialize()
        mesh = pdims.build_mesh("cuda")

        with torch.device("meta"):
            actor_infer = instantiate(
                instantiate(self.conf.ppo.actor.model_infer)(self.conf.ppo.actor.model)
            )

        if self.conf.ppo.actor.parallelize_infer is not None:
            actor_infer = instantiate(self.conf.ppo.actor.parallelize_infer)(
                model=actor_infer, mesh=mesh, parallel_dims=pdims
            )

        actor_infer = actor_infer.to(dtype=get_torch_dtype(self.conf.ppo.infer_dtype))
        actor_infer.to_empty(device=device)

        load_checkpoint(self.conf.ppo.actor.checkpoint_path, model_parts=actor_infer)

        tokenizer = instantiate(self.conf.ppo.actor.tokenizer)

        inference_engine = InferenceEngine(
            actor_infer.to(device=device),
            tokenizer,
            ModelConfig(
                n_heads=self.conf.ppo.actor.model_conf.n_heads,
                n_key_value_heads=self.conf.ppo.actor.model_conf.n_key_value_heads,
                head_dim=self.conf.ppo.actor.model_conf.head_dim,
                n_layers=self.conf.ppo.actor.model_conf.n_layers,
                context_len=self.conf.ppo.actor.model_conf.context_len,
                memory_fraction_static=self.conf.ppo.inference.memory_fraction,
                kv_cache_dtype=get_torch_dtype(self.conf.ppo.infer_dtype),
                gpu_id=self.rank,
                distributed=True,
            ),
            ServerConfig(
                use_cudagraph=True, cudagraph_additonal_batch_size=(1, 2, 4, 8)
            ),
        )

        self.generator = GeneratorWorker(
            inference_engine=inference_engine,
            trajectory_queue=self.trajectory_queue,
        )

    @endpoint
    async def register_rdma_buffer(self, rdma_buffer):
        await self.generator.register_rdma_buffer(rdma_buffer)

    @endpoint
    async def update_state_dict(self):
        await self.generator.update_state_dict()

    @endpoint
    async def finished_training(self):
        await self.generator.finished_training()

    @endpoint
    async def run(self):
        await self.generator.run()

        await self.generator.wait_for_finish()

    @endpoint
    async def finalize(self):
        torch.distributed.destroy_process_group()


class RolloutManager(Actor):
    def __init__(
        self, conf, trajectory_queue, environment_queue, replay_buffers, n_generators
    ):
        rank = current_rank()
        self.rank = rank["gpus"]
        self.world_size = current_size()["gpus"]

        self.logger = get_logger()

        tokenizer = instantiate(conf.ppo.actor.tokenizer)
        request_builder = instantiate(conf.ppo.request_builder, tokenizer=tokenizer)

        train_source, train_ratios, train_names = build_dataset_from_spec(
            conf.data.train, split="train", split_ratio=conf.data.train_ratio
        )

        preprocess_ops = []
        if conf.data.preprocess is not None:
            for op in conf.data.preprocess:
                preprocess_ops.append(instantiate(op))

        collate_fn = None
        if conf.data.collate_fn is not None:
            collate_fn = instantiate(conf.data.collate_fn)

        train_dset = WeightedIterableDataset(
            train_source,
            train_ratios,
            train_names,
            num_replicas=self.world_size,
            rank=self.rank,
            operations=preprocess_ops,
        )

        train_loader = DataLoader(
            train_dset,
            batch_size=conf.training.train_batch_size,
            collate_fn=collate_fn,
            num_workers=2,
            rank=self.rank,
            drop_last=True,
        )

        self.rollout = RolloutWorker(
            dataloader=train_loader,
            request_builder=request_builder,
            trajectory_queue=trajectory_queue,
            environment_queue=environment_queue,
            replay_buffers=replay_buffers,
            batch_size=conf.training.train_batch_size,
            max_iter=conf.training.max_iter,
            n_generators=n_generators,
        )

        self.trajectory_queue = trajectory_queue

    @endpoint
    async def run(self):
        await self.rollout.run()


class Environment(Actor):
    def __init__(self, conf, environment_queue):
        rollout_generator = instantiate(conf.ppo.rollout_generator)(None)

        self.environment = EnvironmentWorker(
            rollout_generator,
            environment_queue,
        )

    @endpoint
    async def run(self):
        await self.environment.run()


async def main():
    conf = load_arg_config(PPOConfig)

    local_ip = find_local_ip()
    free_ports = find_multiple_free_ports(2)

    generator_mesh = await proc_mesh(
        gpus=conf.monarch.generator_mesh_size,
        env={"MASTER_ADDR": local_ip, "MASTER_PORT": str(free_ports[0])},
    )
    trainer_mesh = await proc_mesh(
        gpus=conf.monarch.trainer_mesh_size,
        env={"MASTER_ADDR": local_ip, "MASTER_PORT": str(free_ports[1])},
    )
    rollout_mesh = await proc_mesh(gpus=1)

    replay_buffers = await trainer_mesh.spawn(
        "replay_buffers",
        BasicReplayBuffer,
        conf.training.train_batch_size // conf.monarch.trainer_mesh_size,
    )
    trajectory_queue = await rollout_mesh.spawn("trajectory_queue", BidirectionalQueue)
    environment_queue = await rollout_mesh.spawn(
        "environment_queue", BidirectionalQueue
    )
    generator = await generator_mesh.spawn(
        "generator",
        Generator,
        conf,
        trajectory_queue,
    )
    rollout_manager = await rollout_mesh.spawn(
        "rollout",
        RolloutManager,
        conf,
        trajectory_queue,
        environment_queue,
        replay_buffers,
        conf.monarch.generator_mesh_size,
    )
    environment = await generator_mesh.spawn(
        "environment",
        Environment,
        conf,
        environment_queue,
    )
    trainer = await trainer_mesh.spawn(
        "trainer", Trainer, conf, replay_buffers, generator
    )

    local_ranks = [
        rank % 8
        for rank in range(
            conf.monarch.generator_mesh_size + conf.monarch.trainer_mesh_size
        )
    ]

    await asyncio.gather(
        generator.initialize.call(local_ranks[: conf.monarch.generator_mesh_size]),
        trainer.initialize.call(local_ranks[conf.monarch.generator_mesh_size :]),
    )

    await asyncio.gather(
        generator.run.call(),
        environment.run.call(),
        trainer.run.call(),
        rollout_manager.run.call(),
    )

    await asyncio.gather(generator.finalize.call(), trainer.finalize.call())

    await asyncio.sleep(0.1)

    await asyncio.gather(
        generator_mesh.stop(),
        trainer_mesh.stop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
