from functools import partial

from slickconf import call, field
from torch import optim

from halite.data import preprocess
from halite.nn.loss import CrossEntropyLoss
from halite.optim import lr_scheduler
from halite.parallel.fsdp import apply_fsdp
from halite.data.tokenizers.sentencepiece import SentencePieceTokenizer

from ...data.dclm_samples import conf as data_conf
from ...models.scaling.base import transformer

conf = field()

lr = 0.01
weight_decay = 1e-4
n_vocab = 32000
tokenizer_path = "/mnt/naplm/seonghyeon/llama2-tokenizer.model"
max_length = 2048

tokenizer = SentencePieceTokenizer(tokenizer_path)

conf.model = field(
    model=call[transformer](
        vocab_size=n_vocab,
        dim=1408,
        n_head=11,
        n_layer=7,
        intermediate_size=call[int](1408 * 3.5),
        max_position_embeddings=max_length,
        softcap=50.0,
        post_norm=True,
        attention_processor="torch",
    ),
    wrapper=partial(apply_fsdp, param_dtype="bfloat16", reduce_dtype="float32"),
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
    max_iter=15259,
    gradient_checkpointing=True,
    optimizer=partial(optim.AdamW, lr=lr, betas=(0.9, 0.95)),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=lr,
        initial_multiplier=1e-6,
        warmup=5000,
        decay=("linear", "cos"),
    ),
    criterion=CrossEntropyLoss(z_loss=1e-4, fast=True),
    weight_decay=weight_decay / lr,
    clip_grad_norm=1.0,
    n_epochs=1,
)

conf.output = field(log_step=10, output_dir="/mnt/ddn/seonghyeon")
