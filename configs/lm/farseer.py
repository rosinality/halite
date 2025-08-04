from functools import partial

from slickconf import call, field
from torch import nn, optim

from halite.data import preprocess
from halite.nn.loss import CrossEntropyLoss
from halite.optim import lr_scheduler
from halite.transformers.embedding import TextEmbedding
from halite.transformers.parallelize import parallelize
from halite.data.tokenizers.sentencepiece import SentencePieceTokenizer
from halite.projects.common.logging import hparam_to_name

from ..data.nemotron_cc_high import conf as data_high
from ..data.nemotron_cc_medium_high import conf as data_medium_high
from ..data.nemotron_cc_medium import conf as data_medium
from ..data.nemotron_cc_medium_low import conf as data_medium_low
from ..data.nemotron_cc_low import conf as data_low
from ..models.transformer import transformer, use_fused_linear_cross_entropy

conf = field()

conf.hparams = field(
    lr=2 ** (-0.5 * 14),
    dim=1024,
    n_layers=16,
    intermediate_size=2752,
    weight_decay=0.1,
    n_vocab=32000,
    batch_size=16,
    tokenizer_path="/mnt/naplm/users/seonghyeon/llama2-tokenizer.model",
    max_length=2048,
    max_tokens=8192 * (10**6),
    model_size="202m",
    data_setting="uniform",
    data_ratios={
        "high": 0.2,
        "medium-high": 0.2,
        "medium": 0.2,
        "medium-low": 0.2,
        "low": 0.2,
    },
)

tokenizer = SentencePieceTokenizer(conf.hparams.ref.tokenizer_path)

conf.model = field(
    model=call[transformer](
        vocab_size=conf.hparams.n_vocab,
        dim=conf.hparams.dim,
        n_heads=8,
        head_dim=conf.hparams.dim // 8,
        n_layers=conf.hparams.n_layers,
        intermediate_size=conf.hparams.intermediate_size,
        context_len=conf.hparams.max_length,
        post_norm=False,
        attention_processor="torch",
        embedding=TextEmbedding(
            nn.Embedding(conf.hparams.n_vocab, conf.hparams.dim),
            0,
            embed_init=partial(
                nn.init.normal_,
                std=conf.hparams.dim**-1.5,
            ),
            multiplier=conf.hparams.dim**0.5,
        ),
    ),
    wrapper=partial(
        parallelize,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        tensor_parallel_config={"enable_async_tp": True},
        activation_checkpointing=True,
        activation_checkpointing_config={"mode": "full", "selective": "op"},
        compile=True,
        force_fsdp=True,
    ),
)

train_data_conf = field(
    root=data_high.root,
    ratios=conf.hparams.ref.data_ratios,
    shards={
        "high": data_high.shards["high"],
        "medium-high": data_medium_high.shards["medium-high"],
        "medium": data_medium.shards["medium"],
        "medium-low": data_medium_low.shards["medium-low"],
        "low": data_low.shards["low"],
    },
)

eval_data_conf = field(
    root=data_high.root,
    ratios={
        "high": 0.2,
        "medium-high": 0.2,
        "medium": 0.2,
        "medium-low": 0.2,
        "low": 0.2,
    },
    shards={
        "high": data_high.shards["high"],
        "medium-high": data_medium_high.shards["medium-high"],
        "medium": data_medium.shards["medium"],
        "medium-low": data_medium_low.shards["medium-low"],
        "low": data_low.shards["low"],
    },
)


conf.data = field(
    train=train_data_conf,
    train_ratio=0.9998,
    eval=eval_data_conf,
    eval_ratio=0.0002,
    preprocess=[
        preprocess.ReadRawText(),
        preprocess.Tokenize(tokenizer, bos=True, eos=True),
        preprocess.SequencePacking(conf.hparams.ref.max_length + 1),
        preprocess.AutoregressiveSample(),
    ],
    collate_fn=preprocess.Collator(keys=("input", "target")),
)

conf.training = field(
    train_batch_size=conf.hparams.ref.batch_size,
    eval_batch_size=256,
    max_iter=conf.hparams.ref.max_tokens
    // conf.hparams.ref.max_length
    // conf.hparams.ref.batch_size,
    gradient_checkpointing=False,
    optimizer=partial(optim.AdamW, lr=conf.hparams.ref.lr, betas=(0.9, 0.95)),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=conf.hparams.ref.lr,
        initial_multiplier=1e-6,
        final_multiplier=1e-6 / conf.hparams.ref.lr,
        warmup=2000,
        decay=("linear", "cos"),
    ),
    criterion=CrossEntropyLoss(z_loss=1e-4, fast=True),
    weight_decay=conf.hparams.ref.weight_decay,
    clip_grad_norm=1.0,
    n_epochs=1,
)

conf.output = field(
    log_step=10,
    output_dir="/mnt/ddn/seonghyeon",
    name=partial(
        hparam_to_name,
        format="{model_size}@lr-{lr:.5f}_batch-{batch_size}_max_tokens-{max_tokens:.3e}_data-{data_setting}",
    ),
)


def scale_202m():
    conf.model.model = use_fused_linear_cross_entropy(conf.model.model, z_loss=1e-4)
    conf.training.calc_loss_in_model = True

    return conf
