from functools import partial

from slickconf import call, field, external
from slickconf import call, field, external
from torch import optim

from halite.data import preprocess
from halite.optim import lr_scheduler
from halite.projects.common.config import load_model
from halite.projects.common.rollout import (
    Handler,
    RolloutGenerator,
    RewardRegistry,
    RequestBuilder,
)
from halite.projects.common.rollout_fn import Compose, Detokenize, ToTokenReward
from halite.projects.common.train_fn import basic_train_step
from halite.projects.ppo.model import UnpaddedModel
from halite.projects.ppo.trainer import PPOTrainer, compute_grpo_advantage
from halite.projects.ppo.variants import PPOAdaptiveEntropyActorLoss

from ..data.hendrycks_math import conf as data_conf
from ..models.transformer import use_flash_attention
from .rewards import MathVerify


lr = 1e-6
max_iter = 1000


conf = field()

template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {{ problem }}
Assistant: <think>"""

conf.data = field(
    train=data_conf.train,
    eval=data_conf.test,
    preprocess=[
        preprocess.ParseFeatures(),
        preprocess.ApplyTemplate(key="prompt_text", template=template),
    ],
    collate_fn=preprocess.Collator(
        keys=("prompt_text", "solution"),
        collate_fns={
            "prompt_text": partial(preprocess.collate_list),
            "solution": partial(preprocess.collate_list),
        },
    ),
)


conf.training = field(
    train_batch_size=64,
    eval_batch_size=8,
    max_iter=max_iter,
    ppo_minibatch_size=128,
    ppo_microbatch_size=8,
    ppo_n_epochs=1,
    train_step_fn=partial(basic_train_step),
    optimizer=partial(optim.AdamW, lr=lr, betas=(0.9, 0.95)),
    scheduler=partial(
        lr_scheduler.cycle_scheduler,
        lr=lr,
        n_iter=max_iter,
        initial_multiplier=1e-6,
        final_multiplier=1.0,
        warmup=0,
        decay=("linear", "cos"),
    ),
    weight_decay=0.1,
    clip_grad_norm=1.0,
    n_epochs=30,
)

conf.output = field(log_step=10, save_step=100, wandb_log_step=1)
conf.output = field(log_step=10, save_step=100, wandb_log_step=1)


def qwen3_0_6b_grpo():
    input_key = "input_text"
    output_key = "output_texts"
    reward_key = "correctness"

    conf.ppo = field()
    conf.ppo.actor = load_model("/mnt/naplm/seonghyeon/qwen3/halite/qwen3-0.6b")
    conf.ppo.actor.model = use_flash_attention(conf.ppo.actor.model)
    conf.ppo.actor.parallelize.compile_config = {"fullgraph": False}
    conf.ppo.actor_wrapper = partial(UnpaddedModel)
    conf.ppo.trainer = partial(
        PPOTrainer,
        advantage_fn=partial(compute_grpo_advantage, std_normalize=False),
        actor_loss=PPOAdaptiveEntropyActorLoss(
            clip_low=0.2,
            clip_high=0.2,
            pg_loss_agg="token-sum",
            pg_loss_max_tokens=4096,
            init_entropy_coef=0.0,
            min_entropy_coef=0.0,
            max_entropy_coef=0.005,
            entropy_coef_update_size=0.0001,
            target_entropy=0.2,
        ),
        log_probs_batch_size=16,
    )
    conf.ppo.rollout_generator = partial(
        RolloutGenerator,
        reward_registry=RewardRegistry(
            Handler(
                "correctness",
                external[MathVerify](),
                args=(output_key, "solution"),
            ),
            postprocess=Compose(
                ToTokenReward("output_ids", reward_key, "token_rewards")
            ),
        ),
        preprocessors=[Detokenize(conf.ppo.actor.tokenizer)],
    )
    conf.ppo.request_builder = RequestBuilder(
        "prompt_text",
        sampling_params={"max_new_tokens": 2048, "n": 16, "stop": "</answer>"},
        type="math",
        meta_maps={"input_text": "prompt_text", "solution": "solution"},
    )
    conf.ppo.report = field(
        input_key=input_key,
        output_key=output_key,
        reward_key=reward_key,
        additional_keys=["solution"],
        show_every_nth_sample=8,
    )

    return conf


def qwen3_8b_grpo():
    conf = call[qwen3_0_6b_grpo]()
    conf.ppo.actor = load_model("/mnt/naplm/seonghyeon/qwen3/halite/qwen3-8b")

    return conf
