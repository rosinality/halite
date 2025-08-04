import torch
from torch import nn
from torch import distributed as dist
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
from halite.projects.lm.config import LMConfig
from halite.optim import group_parameters


def train(
    conf,
    model,
    criterion,
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
    is_train = model.training
    model.train()

    if conf.training.model_initializer is not None:
        initializer = instantiate(conf.training.model_initializer)
        initializer(model)

    postprocess = None
    if conf.training.postprocess is not None:
        postprocess = instantiate(conf.training.postprocess)

    train_iter = len(train_loader)

    loader = iter(DataManager(train_loader, parallel_dims.mesh.get_group("dp")))

    train_step_fn = instantiate(conf.training.train_step_fn)

    step = 0
    while True:
        try:
            batch = next(loader)

        except StopIteration:
            break

        optimizer.zero_grad()
        batch = batch.to(device)

        loss, loss_dict = train_step_fn(batch, model, criterion)

        loss.backward()

        grad_norm = None
        if conf.training.clip_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), conf.training.clip_grad_norm, foreach=True
            )

        scheduler.step()
        optimizer.step()

        if postprocess is not None:
            postprocess(model)

        if global_step % conf.output.log_step == 0:
            lr = optimizer.param_groups[0]["lr"]

            loss_txt = "; ".join([f"{k}: {v:.5f}" for k, v in loss_dict.items()])

            if grad_norm is None:
                logger.info(
                    f"epoch {epoch}; {step}/{train_iter}; global {global_step}; loss: {loss.item():.5f}; {loss_txt}; lr: {lr:.7f}"
                )

            else:
                logger.info(
                    f"epoch {epoch}; {step}/{train_iter}; global {global_step}; loss: {loss.item():.5f}; {loss_txt}; grad norm: {grad_norm.item():.3f}; lr: {lr:.7f}"
                )

            if parallel_dims.is_primary and wandb is not None:
                report = {"train/loss": loss, "train/lr": lr}
                report.update({f"train/{k}": v for k, v in loss_dict.items()})

                if grad_norm is not None:
                    report["train/grad_norm"] = grad_norm

                wandb.log(report, step=global_step)

        global_step += 1
        step += 1

        if step % conf.training.eval_step == 0:
            eval_loss = evaluate(
                conf, model, criterion, eval_loader, device, parallel_dims, logger
            )
            logging_loss(eval_loss, global_step, parallel_dims, logger)

        if conf.training.max_iter is not None and global_step >= conf.training.max_iter:
            break

        checkpoint_manager.save(global_step)

    model.train(is_train)

    return step


@torch.no_grad()
def evaluate(conf, model, criterion, eval_loader, device, parallel_dims, logger):
    is_train = model.training
    model.eval()

    length = len(eval_loader)

    step = 0

    data_manager = DataManager(eval_loader, parallel_dims.mesh)
    loader = iter(data_manager)
    losses_sum = {}
    total_targets = 0
    use_micro_average = False

    if conf.training.eval_step_fn is None:
        eval_step_fn = instantiate(conf.training.train_step_fn)

    else:
        eval_step_fn = instantiate(conf.training.eval_step_fn)

    while True:
        try:
            batch = next(loader)

        except StopIteration:
            break

        step += 1

        batch = batch.to(device)
        loss, loss_dict = eval_step_fn(batch, model, criterion)

        if parallel_dims.is_primary and step % conf.output.log_step == 0:
            logger.info(f"evaluating [{step}/{length}]; loss: {loss.item():.5f}")

        n_targets = loss_dict.get("n_targets", 1)

        if n_targets.item() == 1:
            torch.save(batch, f"batch_{step}.pt")

        total_targets += n_targets
        if "n_targets" in loss_dict:
            use_micro_average = True

        for key, val in loss_dict.items():
            if key == "n_targets":
                continue

            val = val.float()

            if key not in losses_sum:
                losses_sum[key] = val * n_targets

            else:
                losses_sum[key] += val * n_targets

    if use_micro_average:
        total_targets = torch.as_tensor(total_targets, device=device)
        dist.all_reduce(total_targets, group=parallel_dims.mesh.get_group("dp"))
        total_targets = total_targets.item()
        losses_sum = {k: v / total_targets for k, v in losses_sum.items()}
        total_targets = 1

    else:
        losses_sum = {k: v / step for k, v in losses_sum.items()}
        total_targets = parallel_dims.dp

    try:
        for key, val in losses_sum.items():
            val = all_reduce_mean(
                val,
                parallel_dims.mesh.get_group("dp"),
                total_targets,
            )

            losses_sum[key] = val

    except:
        pass

    for key, val in losses_sum.items():
        if isinstance(val, torch.Tensor):
            losses_sum[key] = val.item()

    model.train(is_train)

    return losses_sum


def logging_loss(loss_dict, global_step, parallel_dims, logger):
    loss_txt = "; ".join([f"{k}: {v:.5f}" for k, v in loss_dict.items()])

    logger.info(f"{loss_txt} at step {global_step}")

    if parallel_dims.is_primary and wandb is not None:
        wandb.log({"eval/" + k: v for k, v in loss_dict.items()}, step=global_step)


def main():
    conf = load_arg_config(LMConfig)

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
        wandb.require("core")
        wandb.init(project="halite-lm")

    torch.distributed.barrier()

    logger.info("building model")

    with torch.device("meta"):
        model = instantiate(conf.model.model)

    if conf.model.wrapper is not None:
        logger.info("applying wrapper")
        model = instantiate(conf.model.wrapper)(
            model=model, mesh=mesh, parallel_dims=pdims
        )

    if conf.model.parallelize is not None:
        logger.info("applying parallelize")
        model = instantiate(conf.model.parallelize)(
            model=model, mesh=mesh, parallel_dims=pdims
        )

    if conf.model.wrapper is not None or conf.model.parallelize is not None:
        logger.info(str(model))

    model.to_empty(device=device)

    if conf.model.checkpoint_path is not None:
        load_checkpoint(conf.model.checkpoint_path, model_parts=model)

    criterion = instantiate(conf.training.criterion)
    optimizer = instantiate(conf.training.optimizer)(
        group_parameters(model, weight_decay=conf.training.weight_decay)
    )
    scheduler = instantiate(conf.training.scheduler)(
        optimizer=optimizer,
    )

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

    logger.info("evaluating baseline")
    eval_loss = evaluate(conf, model, criterion, eval_loader, device, pdims, logger)
    logging_loss(eval_loss, 0, pdims, logger)

    logger.info("start training")

    checkpoint_manager = CheckpointManager(
        model,
        optimizer,
        scheduler,
        train_loader,
        model_config=conf.model,
        directory=conf.output.output_dir,
        interval=conf.output.save_step,
    )

    global_step = 0

    for epoch in range(conf.training.n_epochs):
        logger.info(f"epoch {epoch}")

        global_step += train(
            conf,
            model,
            criterion,
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

    logger.info("final evaluation score")
    eval_loss = evaluate(conf, model, criterion, eval_loader, device, pdims, logger)
    logging_loss(eval_loss, global_step, pdims, logger)

    checkpoint_manager.save(global_step)


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
