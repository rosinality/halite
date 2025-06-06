from functools import partial

from slickconf import call, field
from torch import optim
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    FullyShardShampooConfig,
    PrecisionConfig,
)

from halite.data import preprocess
from halite.nn.loss import CrossEntropyLoss
from halite.optim import lr_scheduler
from halite.transformers.parallelize import parallelize
from halite.data.tokenizers.sentencepiece import SentencePieceTokenizer

from ..data.dclm_samples import conf as data_conf
from ..models.scaling.base import transformer

conf = field()

lr = 1e-3
weight_decay = 0.1
n_vocab = 32000
tokenizer_path = "/mnt/naplm/seonghyeon/llama2-tokenizer.model"
max_length = 2048

tokenizer = SentencePieceTokenizer(tokenizer_path)

conf.model = field(
    model=call[transformer](
        vocab_size=n_vocab,
        dim=1024,
        n_heads=8,
        n_layers=23,
        intermediate_size=call[int](1024 * 3.5),
        max_position_embeddings=max_length,
        softcap=50.0,
        post_norm=True,
        qkv_split=True,
        gated_ff_split=True,
        attention_processor="torch",
    ),
    wrapper=partial(
        parallelize,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        tensor_parallel_config={"enable_async_tp": True},
        activation_checkpointing=True,
        activation_checkpointing_config={"mode": "full", "selective": "op"},
        compile=True,
    ),
)

conf.data = field(
    train=data_conf,
    train_ratio=0.9998,
    eval=data_conf,
    eval_ratio=0.0002,
    preprocess=[
        preprocess.ReadRawText(),
        preprocess.Tokenize(tokenizer, bos=True, eos=True),
        preprocess.SequencePacking(max_length + 1),
        preprocess.AutoregressiveSample(),
    ],
    collate_fn=preprocess.Collator(keys=("input", "target")),
)

conf.training = field(
    train_batch_size=320,
    eval_batch_size=320,
    max_iter=50000,
    gradient_checkpointing=False,
    optimizer=partial(
        DistributedShampoo,
        lr=lr,
        betas=(0.9, 0.95),
        epsilon=1e-12,
        max_preconditioner_dim=8192,
        precondition_frequency=10,
        use_decoupled_weight_decay=True,
        inv_root_override=2,
        distributed_config=FullyShardShampooConfig(),
        grafting_config=AdamGraftingConfig(
            beta2=0.95,
            epsilon=1e-08,
        ),
    ),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=lr,
        initial_multiplier=1e-6,
        warmup=5000,
        decay=("linear", "cos"),
    ),
    criterion=CrossEntropyLoss(z_loss=1e-4, fast=True),
    weight_decay=weight_decay,
    clip_grad_norm=1.0,
    n_epochs=1,
)

conf.output = field(log_step=10, output_dir="/mnt/ddn/seonghyeon")
