from functools import partial

from slickconf import field
from torch import optim

from halite.data import preprocess
from halite.nn.loss import CrossEntropyLoss
from halite.optim import lr_scheduler
from halite.transformers.parallelize import parallelize
from halite.transformers.flex_attention import FlexAttentionProcessor
from halite.transformers.flex_attention.causal import CausalMask
from halite.transformers.flex_attention.batch_document_mask import BatchDocumentMask
from halite.projects.common.config import load_model
from halite.projects.common.template import simple_format
from halite.projects.common.train_fn import basic_train_step
from halite.projects.sft.preprocess import (
    SFTSample,
    SFTSequencePacking,
    collate_offsets,
)

from ..data.hendrycks_math import conf as data_conf
from ..models.transformer import use_flex_attention

model_checkpoint = "/mnt/naplm/seonghyeon/llama/halite/llama3.2-3b"
prompt_template = "Problem: {0}\n\nSolution:"
response_template = " {0}"
lr = 3e-5
max_iter = 1000

conf = field()

conf.data = field(
    train=data_conf.train,
    eval=data_conf.test,
    preprocess=[
        preprocess.ParseFeatures(),
        SFTSample(
            prompt_key="problem",
            response_key="solution",
            prompt_map_fn=simple_format(prompt_template),
            response_map_fn=simple_format(response_template),
            prompt_tokenizer_kwargs={"bos": True, "eos": False},
            response_tokenizer_kwargs={"bos": False, "eos": True},
        ),
        SFTSequencePacking(
            use_position_ids=True,
            use_document_offsets=True,
            use_rest_of_long_sequence=False,
        ),
    ],
    collate_fn=preprocess.Collator(
        keys=("input", "target", "position_ids", "document_offsets"),
        collate_fns={"document_offsets": partial(collate_offsets, max_len=128)},
    ),
)


conf.training = field(
    train_batch_size=8,
    eval_batch_size=8,
    max_iter=max_iter,
    train_step_fn=partial(basic_train_step),
    optimizer=partial(optim.AdamW, lr=lr, betas=(0.9, 0.95)),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=lr,
        n_iter=max_iter,
        initial_multiplier=1e-6,
        warmup=500,
        decay=("linear", "cos"),
    ),
    criterion=CrossEntropyLoss(z_loss=1e-4, fast=True, micro_average=True),
    weight_decay=0.1,
    clip_grad_norm=1.0,
    n_epochs=5,
)

conf.output = field(log_step=10, save_step=100)


def llama3_2_3b():
    conf.model = load_model(model_checkpoint)
    conf.model.model = use_flex_attention(conf.model.model)
    conf.model.model.flex_attention_processor = FlexAttentionProcessor(
        block_mask=BatchDocumentMask(CausalMask())
    )
    conf.model.parallelize = partial(
        parallelize,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        tensor_parallel_config={"enable_async_tp": True},
        activation_checkpointing=True,
        activation_checkpointing_config={"mode": "full", "selective": "op"},
        compile=False,
    )
    conf.data.preprocess[1].tokenizer = conf.model.tokenizer
    conf.data.preprocess[2].length = 1024

    return conf
