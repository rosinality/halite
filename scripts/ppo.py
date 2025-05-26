import gc

import torch
from torch import nn
from slickconf import instantiate, load_arg_config, summarize

try:
    import wandb

except ImportError:
    wandb = None

from halite.distributed import (
    all_reduce_mean,
    load_checkpoint,
    CheckpointManager,
)
from halite.data.dataloader import DataLoader, DataManager
from halite.data.dataset import build_dataset_from_spec, WeightedIterableDataset
from halite.logging import get_logger
from halite.parallel import ParallelDims
from halite.projects.ppo.config import PPOConfig
from halite.optim import group_parameters
from halite.transformers.infer import InferenceEngine, ModelConfig
from halite.utils import get_torch_dtype


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def build_rollout_report(conf, rollout):
    report_texts = []

    n_samples = 0
    for i, (sample, reward) in enumerate(
        zip(rollout.samples, rollout.rewards_dict[conf.reward_key].to("cpu").unbind())
    ):
        if i % conf.show_every_nth_sample != 0:
            continue

        input_text = sample[conf.input_key]
        output_text = sample[conf.output_key]

        report_text = f"# {i}\n[input]\n{input_text}\n\n[output]\n{output_text}\n\n[reward]\n{reward}"

        if conf.additional_keys is not None:
            for key in conf.additional_keys:
                report_text += f"\n\n[{key}]\n{sample[key]}"

        report_texts.append(report_text)

        n_samples += 1

        if n_samples >= conf.show_n_samples:
            break

    return "\n\n".join(report_texts)


def train(
    conf,
    trainer,
    request_builder,
    rollout_generator,
    optimizer,
    scheduler,
    train_loader,
    eval_loader,
    checkpoint_manager,
    device,
    epoch,
    global_step,
    parallel_dims,
    logger,
):
    loader = iter(DataManager(train_loader, parallel_dims.mesh))

    train_iter = len(train_loader)

    step = 0
    while True:
        try:
            batch = next(loader)

        except StopIteration:
            break

        batch = batch.to(device)

        rollout_generator.initialize()
        rollout_generator.load_state_dict(trainer.actor.state_dict())

        requests = request_builder(batch)
        rollout = rollout_generator.generate(requests)

        rollout_generator.cleanup()

        clean_memory()

        rollout = trainer.compute_advantage(rollout)
        rollout_batches = [rollout]

        metrics = []
        for ppo_epoch in range(conf.training.ppo_n_epochs):
            optimizer.zero_grad()

            if conf.training.ppo_minibatch_size is not None:
                rollout_batches = rollout.split(conf.training.ppo_minibatch_size)

            for rollout_batch in rollout_batches:
                actor_loss = trainer.compute_actor_loss(rollout_batch)

                actor_loss.pg_loss.backward()

            grad_norm = None
            if conf.training.clip_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    trainer.actor.parameters(),
                    conf.training.clip_grad_norm,
                    foreach=True,
                )

            scheduler.step()
            optimizer.step()

            loss_dict = actor_loss._asdict()
            loss_dict["actor/mean_response_len"] = (
                rollout.batch.response_len.float().mean()
            )

            metrics.append(loss_dict)

        if global_step % conf.output.log_step == 0:
            if conf.ppo.report is not None:
                logger.info(build_rollout_report(conf.ppo.report, rollout))

            metrics_mean = {}
            for metric in metrics:
                for k, v in metric.items():
                    if k not in metrics_mean:
                        metrics_mean[k] = []

                    metrics_mean[k].append(v)

            metrics_mean = {k: torch.stack(v).mean() for k, v in metrics_mean.items()}

            metrics_mean = {
                k: all_reduce_mean(
                    v, parallel_dims.mesh.get_group("dp"), parallel_dims.dp
                ).item()
                for k, v in metrics_mean.items()
            }

            lr = optimizer.param_groups[0]["lr"]

            loss_txt = "; ".join([f"{k}: {v:.5f}" for k, v in metrics_mean.items()])

            if grad_norm is None:
                logger.info(
                    f"epoch {epoch}; {step}/{train_iter}; global {global_step}; {loss_txt}; lr: {lr:.7f}"
                )

            else:
                logger.info(
                    f"epoch {epoch}; {step}/{train_iter}; global {global_step}; {loss_txt}; grad norm: {grad_norm.item():.3f}; lr: {lr:.7f}"
                )

            if parallel_dims.is_primary and wandb is not None:
                report = {"actor/lr": lr}
                report.update({f"actor/{k}": v for k, v in metrics_mean.items()})

                if grad_norm is not None:
                    report["actor/grad_norm"] = grad_norm

                wandb.log(report, step=global_step)

        global_step += 1
        step += 1

    return step


def logging_loss(loss_dict, global_step, parallel_dims, logger):
    loss_txt = "; ".join([f"{k}: {v:.5f}" for k, v in loss_dict.items()])

    logger.info(f"{loss_txt} at step {global_step}")

    if parallel_dims.is_primary and wandb is not None:
        wandb.log({"eval/" + k: v for k, v in loss_dict.items()}, step=global_step)


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


def main():
    conf = load_arg_config(PPOConfig)

    pdims = ParallelDims(
        dp_replicate=conf.training.data_parallel_replicate,
        dp_shard=conf.training.data_parallel_shard,
        tp=conf.training.tensor_parallel,
        pp=conf.training.pipeline_parallel,
    )
    mesh = pdims.build_mesh("cuda")
    logger = get_logger(mesh)

    logger.info(summarize(conf))
    logger.info(
        f"dp replicate: {pdims.dp_replicate}, dp shard: {pdims.dp_shard}, tp: {pdims.tp} pp: {pdims.pp}"
    )

    device = torch.device("cuda")

    if pdims.is_primary and wandb is not None:
        wandb.init(project="halite-ppo")

    torch.distributed.barrier()

    actor = build_model(conf.ppo.actor, mesh, pdims, device, logger=logger)
    ref = None
    if conf.ppo.ref is not None:
        ref = build_model(conf.ppo.ref, mesh, pdims, device, logger=logger)

    critic = None
    if conf.ppo.critic is not None:
        critic = build_model(conf.ppo.critic, mesh, pdims, device, logger=logger)

    actor_train = actor
    if conf.ppo.actor_wrapper is not None:
        actor_train = instantiate(conf.ppo.actor_wrapper)(actor)

    trainer = instantiate(conf.ppo.trainer)(
        actor=actor_train,
        ref=ref,
        critic=critic,
        device=device,
    )

    with torch.device("meta"):
        actor_infer = instantiate(
            instantiate(conf.ppo.actor.model_infer)(conf.ppo.actor.model)
        )

    if conf.ppo.actor.parallelize_infer is not None:
        actor_infer = instantiate(conf.ppo.actor.parallelize_infer)(
            model=actor_infer, mesh=mesh, parallel_dims=pdims
        )

    actor_infer = actor_infer.to(dtype=get_torch_dtype(conf.ppo.infer_dtype))
    actor_infer.to_empty(device=device)

    tokenizer = instantiate(conf.ppo.actor.tokenizer)

    inference_engine = InferenceEngine(
        actor_infer.to(device=device),
        tokenizer,
        ModelConfig(
            n_heads=conf.ppo.actor.model_conf.n_heads,
            n_key_value_heads=conf.ppo.actor.model_conf.n_key_value_heads,
            head_dim=conf.ppo.actor.model_conf.head_dim,
            n_layers=conf.ppo.actor.model_conf.n_layers,
            context_len=conf.ppo.actor.model_conf.context_len,
            memory_fraction_static=0.6,
            kv_cache_dtype=get_torch_dtype(conf.ppo.infer_dtype),
        ),
    )

    request_builder = instantiate(conf.ppo.request_builder)
    rollout_generator = instantiate(conf.ppo.rollout_generator)(inference_engine)

    train_source, train_ratios, train_names = build_dataset_from_spec(
        conf.data.train, split="train", split_ratio=conf.data.train_ratio
    )
    eval_source, eval_ratios, eval_names = build_dataset_from_spec(
        conf.data.eval, split="eval", split_ratio=conf.data.eval_ratio
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
        num_replicas=pdims.dp,
        rank=mesh.get_local_rank("dp"),
        operations=preprocess_ops,
    )

    eval_dset = WeightedIterableDataset(
        eval_source,
        eval_ratios,
        eval_names,
        num_replicas=pdims.dp,
        rank=mesh.get_local_rank("dp"),
        operations=preprocess_ops,
    )

    logger.info(f"train_dset\n{train_dset.summary()}")
    logger.info(f"eval_dset\n{eval_dset.summary()}")

    train_loader = DataLoader(
        train_dset,
        batch_size=conf.training.train_batch_size // pdims.dp,
        collate_fn=collate_fn,
        num_workers=4,
        rank=mesh.get_local_rank("dp"),
    )
    eval_loader = DataLoader(
        eval_dset,
        batch_size=conf.training.eval_batch_size // pdims.dp,
        collate_fn=collate_fn,
        num_workers=4,
        rank=mesh.get_local_rank("dp"),
    )

    optimizer = instantiate(conf.training.optimizer)(
        group_parameters(actor, weight_decay=conf.training.weight_decay)
    )
    scheduler = instantiate(conf.training.scheduler)(
        optimizer=optimizer,
    )

    checkpoint_manager = CheckpointManager(
        actor,
        optimizer,
        scheduler,
        train_loader,
        model_config=conf.ppo.actor,
        directory=conf.output.output_dir,
        interval=conf.output.save_step,
    )

    global_step = 0

    for epoch in range(conf.training.n_epochs):
        logger.info(f"epoch {epoch}")

        global_step += train(
            conf,
            trainer,
            request_builder,
            rollout_generator,
            optimizer,
            scheduler,
            train_loader,
            eval_loader,
            checkpoint_manager,
            device,
            epoch,
            global_step,
            pdims,
            logger,
        )


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
