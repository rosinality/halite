from functools import partial

from slickconf import field

from halite.data import preprocess
from halite.nn.loss import CrossEntropyLoss
from halite.optim import lr_scheduler
from halite.transformers.parallelize import parallelize
from halite.projects.common.config import load_model
from halite.projects.common.template import simple_format
from halite.projects.sft.preprocess import (
    SFTSample,
    SFTSequencePacking,
    collate_offsets,
)

from ..data.hendrycks_math import conf as data_conf

model_checkpoint = "/mnt/naplm/seonghyeon/llama/halite/llama3.2-3b"
prompt_template = "Problem: {0}\n\nSolution:"
response_template = " {0}"

conf = field()

conf.model = load_model(model_checkpoint)
conf.parallelize = partial(
    parallelize,
    param_dtype="bfloat16",
    reduce_dtype="float32",
    tensor_parallel_config={"enable_async_tp": True},
    activation_checkpointing=True,
    activation_checkpointing_config={"mode": "full", "selective": "op"},
    compile=True,
)

conf.data = field(
    train=data_conf.train,
    eval=data_conf.test,
    preprocess=[
        preprocess.ParseFeatures(),
        SFTSample(
            prompt_key="problem",
            response_key="solution",
            tokenizer=conf.model.tokenizer,
            prompt_map_fn=simple_format(prompt_template),
            response_map_fn=simple_format(response_template),
            prompt_tokenizer_kwargs={"bos": True, "eos": False},
            response_tokenizer_kwargs={"bos": False, "eos": True},
        ),
        SFTSequencePacking(
            length=conf.model.model_conf.context_len,
            use_position_ids=True,
            use_document_offsets=True,
            use_rest_of_long_sequence=False,
        ),
    ],
    collate_fn=preprocess.Collator(
        keys=("input", "target", "position_ids", "document_offsets"),
        collate_fns={"document_offsets": collate_offsets},
    ),
)


conf.training = field(
    train_batch_size=8,
    eval_batch_size=8,
)
