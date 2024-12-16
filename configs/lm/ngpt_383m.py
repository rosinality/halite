from functools import partial

from slickconf import call, field
from torch import optim

from halite.data import preprocess
from halite.nn.loss import CrossEntropyLoss
from halite.optim import lr_scheduler
from halite.transformers.parallelize import parallelize
from halite.transformers.optimizer import ngpt_normalize
from halite.data.tokenizers.sentencepiece import SentencePieceTokenizer

from ..data.dclm_samples import conf as data_conf
from ..models.ngpt import ngpt

conf = field()

lr = 1e-3
weight_decay = 0
n_vocab = 32000
tokenizer_path = "/mnt/naplm/seonghyeon/llama2-tokenizer.model"
max_length = 2048

tokenizer = SentencePieceTokenizer(tokenizer_path)

conf.model = field(
    model=call[ngpt](
        vocab_size=n_vocab,
        dim=1024,
        n_heads=8,
        n_layers=23,
        intermediate_size=call[int](1024 * 3.5),
        max_position_embeddings=max_length,
        softcap=50.0,
        attention_processor="torch",
    ),
    wrapper=partial(
        parallelize,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        tensor_parallel_config={"enable_async_tp": True},
        activation_checkpointing=True,
        activation_checkpointing_config={"mode": "full", "selective": "op"},
        compile=False,
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
    optimizer=partial(optim.AdamW, lr=lr, betas=(0.9, 0.95)),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=lr,
        initial_multiplier=0,
        warmup=1,
        decay=("linear", "cos"),
    ),
    criterion=CrossEntropyLoss(z_loss=1e-4, fast=True),
    model_initializer=partial(ngpt_normalize),
    postprocess=partial(ngpt_normalize),
    weight_decay=weight_decay,
    clip_grad_norm=1.0,
    n_epochs=1,
)

conf.output = field(log_step=10, output_dir="/mnt/ddn/seonghyeon")
