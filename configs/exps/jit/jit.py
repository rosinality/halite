from functools import partial

from slickconf import call, field
from torch import optim
from torchvision import transforms

from halite.optim import lr_scheduler
from halite.transformers.parallelize import parallelize
from halite.projects.dit.dataset import CenterCrop, ImageFolder
from halite.projects.dit.diffusion import Diffusion, EquilibriumMatchingJiT

from ...models.jit import jit

image_size = 256
batch_size = 128 * 8
lr = 5e-5 * batch_size / 256

conf = field()

conf.model = field(
    wrapper=partial(
        parallelize,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        tensor_parallel_config={"enable_async_tp": True},
        activation_checkpointing=False,
        activation_checkpointing_config={"mode": "full", "selective": "op"},
        compile=True,
        reshard_after_forward=True,
    )
)

conf.data = field(
    train=ImageFolder(
        "/mnt/ddn/ilsvrc2012",
        transform=transforms.Compose(
            [
                CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
            ]
        ),
    )
)

conf.training = field(
    train_batch_size=batch_size,
    eval_batch_size=64,
    n_epochs=600,
    gradient_checkpointing=False,
    optimizer=partial(optim.AdamW, lr=lr, betas=(0.9, 0.95), weight_decay=0),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=lr,
        initial_multiplier=1e-6,
        final_multiplier=1,
        warmup=5,
        decay=("linear", "cos"),
    ),
    criterion=Diffusion(
        input_shape=(3, image_size, image_size),
        n_labels=1000,
        guidance_interval=(0.1, 1.0),
        guidance_scale=2.9,
        noise_scale=1.0,
        p_mean=-0.8,
        p_std=0.8,
    ),
    ema=0.9996,
)

conf.output = field(log_step=10, output_dir="/mnt/ddn/jit", sampling_step=1000)


def jit_b_16():
    conf.model.model = call[jit](
        image_size=256,
        patch_size=16,
        n_labels=1000,
        dim=768,
        patch_dim=128,
        n_heads=12,
        head_dim=768 // 12,
        n_layers=12,
        intermediate_size=int(768 * 4 * 2 / 3),
        in_context_len=32,
        in_context_start=4,
        qk_norm=True,
    )

    return conf


def eqm_jit_b_16():
    conf = jit_b_16()
    conf.training.criterion = EquilibriumMatchingJiT(
        input_shape=(3, image_size, image_size),
        n_labels=1000,
        guidance_interval=(0.1, 1.0),
        guidance_scale=2.9,
        noise_scale=1.0,
        p_mean=-0.8,
        p_std=0.8,
    )

    return conf
