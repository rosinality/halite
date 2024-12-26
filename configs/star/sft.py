from functools import partial

from slickconf import field, get_instance_attr, tag

from halite.data import preprocess
from halite.projects.common.config import get_tokenizer, load_model
from halite.projects.common.template import simple_format
from halite.projects.sft.preprocess import SFTSample, SFTSequencePacking

from ..data.hendrycks_math import conf as data_conf

model_checkpoint = "/mnt/naplm/seonghyeon/llama/halite/llama3.2-3b"
prompt_template = "Problem: {0}\n\nSolution:"
response_template = " {0}"

conf = field()

conf.model = load_model(model_checkpoint)

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
        SFTSequencePacking(length=conf.model.model_conf.context_len),
    ],
    collate_fn=preprocess.Collator(keys=("input", "target")),
)


conf.training = field(
    train_batch_size=8,
    eval_batch_size=8,
)
