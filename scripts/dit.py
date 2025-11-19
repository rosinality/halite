import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid
from slickconf import instantiate, load_arg_config, summarize

try:
    import wandb

except ImportError:
    wandb = None

from halite.distributed import all_reduce_mean
from halite.logging import get_logger
from halite.parallel import ParallelDims
from halite.projects.dit.config import DiTConfig
from halite.optim import group_parameters


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for module in ema_model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()

    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].detach().mul_(decay).add_(param.detach(), alpha=1 - decay)


def train(
    conf,
    model,
    model_ema,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    device,
    epoch,
    global_step,
    parallel_dims,
    logger,
):
    is_train = model.training
    model.train()

    train_iter = len(train_loader)

    dp_size = parallel_dims.mesh["dp"].size()
    dp_group = parallel_dims.mesh["dp"].get_group()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True).to(torch.float32).div_(255)
        images = images * 2 - 1
        labels = labels.to(device, non_blocking=True).to(torch.long)

        loss = criterion.loss(model, images, labels)

        optimizer.zero_grad()
        loss.backward()
        scheduler.step()
        optimizer.step()

        update_ema(model_ema, model, conf.training.ema)

        if step % conf.output.log_step == 0:
            lr = optimizer.param_groups[0]["lr"]

            loss_val = all_reduce_mean(
                loss,
                dp_group,
                dp_size,
            )

            logger.info(
                f"epoch {epoch}; {step}/{train_iter}; global {global_step}; loss: {loss_val:.5f}; lr: {lr:.7f}"
            )

            if parallel_dims.is_primary and wandb is not None:
                report = {"train/loss": loss_val, "train/lr": lr}

                wandb.log(report, step=global_step)

        if global_step % conf.output.sampling_step == 0:
            eval_batch_size = conf.training.eval_batch_size
            sample_size = eval_batch_size // dp_size
            local_rank = parallel_dims.mesh["dp"].get_local_rank()
            labels = torch.arange(
                local_rank * sample_size, (local_rank + 1) * sample_size, device=device
            )
            samples = criterion.generate(model_ema, labels, seed=conf.training.seed)
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            samples_batch = samples.new_zeros(eval_batch_size, *samples.shape[1:])
            dist.barrier()
            dist.all_gather_into_tensor(samples_batch, samples, group=dp_group)

            n_row = round(eval_batch_size**0.5)
            samples_batch = make_grid(
                samples_batch, nrow=n_row, normalize=True, value_range=(0, 1)
            )
            samples_batch = (samples_batch * 255).to("cpu", torch.uint8)

            if parallel_dims.is_primary and wandb is not None:
                wandb.log({"samples": wandb.Image(samples_batch)}, step=global_step)

        global_step += 1

    model.train(is_train)

    return step


def apply_compile(model, config):
    config = {} if config is None else config

    if "fullgraph" not in config:
        config["fullgraph"] = True

    for i, block in model.blocks.named_children():
        block = torch.compile(block, **config)
        model.blocks.register_module(i, block)


def build_model(model, wrapper, parallel_dims, device):
    with torch.device("meta"):
        model = instantiate(model)

    if wrapper is not None:
        model = instantiate(wrapper)(
            model=model, mesh=parallel_dims.mesh, parallel_dims=parallel_dims
        )

    model.to_empty(device=device)

    return model


def main():
    conf = load_arg_config(DiTConfig)

    pdims = ParallelDims(
        dp_replicate=conf.training.data_parallel_replicate,
        dp_shard=conf.training.data_parallel_shard,
        tp=conf.training.tensor_parallel,
        pp=conf.training.pipeline_parallel,
    )
    mesh = pdims.build_mesh("cuda")
    logger = get_logger(mesh)

    seed = conf.training.seed + mesh.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(summarize(conf))
    logger.info(
        f"dp replicate: {pdims.dp_replicate}, dp shard: {pdims.dp_shard}, tp: {pdims.tp} pp: {pdims.pp}"
    )

    device = torch.device("cuda")

    if pdims.is_primary and wandb is not None:
        project = conf.output.project
        project = "halite-dit" if project is None else project

        hparams = conf.hparams
        name = conf.output.name

        if (
            hparams is not None
            and conf.output.name is not None
            and not isinstance(conf.output.name, str)
        ):
            name = instantiate(conf.output.name)(hparams)

        wandb.init(project=project, name=name, config=hparams)

    torch.distributed.barrier()

    logger.info("building model")

    model = build_model(conf.model.model, conf.model.wrapper, pdims, device)
    model.init_weights(device)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[mesh["dp"].get_local_rank()])
    logger.info(str(model))

    model_ema = build_model(conf.model.model, conf.model.wrapper, pdims, device)
    model_ema.eval()
    model_ema.load_state_dict(model.state_dict())

    if conf.training.gradient_checkpointing:
        logger.info("use gradient checkpointing")
        model.gradient_checkpointing_enable(1)

    criterion = instantiate(conf.training.criterion)
    optimizer = instantiate(conf.training.optimizer)(
        group_parameters(model, weight_decay=conf.training.weight_decay)
    )

    train_dset = instantiate(conf.data.train)

    sampler = DistributedSampler(
        train_dset,
        num_replicas=mesh["dp"].size(),
        rank=mesh["dp"].get_local_rank(),
        shuffle=True,
    )

    logger.info(f"train batch size: {conf.training.train_batch_size}")

    train_loader = DataLoader(
        train_dset,
        sampler=sampler,
        batch_size=conf.training.train_batch_size // pdims.dp,
        num_workers=8,
        drop_last=True,
    )

    iter_per_epoch = len(train_loader)

    scheduler = instantiate(conf.training.scheduler)(
        optimizer=optimizer,
        warmup=conf.training.scheduler.warmup * iter_per_epoch,
        n_iter=conf.training.n_epochs * iter_per_epoch,
    )

    logger.info("start training")

    global_step = 0

    for epoch in range(conf.training.n_epochs):
        logger.info(f"epoch {epoch}")

        sampler.set_epoch(epoch)

        global_step += train(
            conf,
            model,
            model_ema,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            device,
            epoch,
            global_step,
            pdims,
            logger,
        )

    # dcp.save(
    #    get_model_state_dict(model),
    #    checkpoint_id=os.path.join(conf.output.output_dir, f"step-{global_step}"),
    # )


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
