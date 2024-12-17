# halite

Halite is an acceleration framework for pre-training, post-training, inference and evaluation of large language models built from scratch with PyTorch.

This is my on-going project, but I'm desined this framework with below things in mind.

- **Post-Training**: Halite starts from my earlier work for accelerating post-training of LLMs especially RLHF and PPO. Halite supports easier way to implement various and sophisticated alignment techniques.
- **Transformers**: Halite supports design and modification of novel transformer architectures with composable components. All of components are not tied to specific architecture, and you can compose it just in your config, without any framework-level code changes, thanks to [slickconf](https://github.com/rosinality/slickconf). Of course, it supports convert checkpoints from another framework in declarative way.
- **Parallelism**: Halite designed to support multi-dimensional parallelism, not only plain FSDP, in a performant and flexible way without hassles.
- **Inference**: Most post-training method requires to sample from the model, a lot. It is crucial to sample efficiently for post-training frameworks to be practical. Halite internalizes inference engine inspired from [SGLang](https://github.com/sgl-project/sglang) that allows switch training or inference mode of the model without any additional cost or checkpointing.
- **Evaluation**: There are great frameworks for evaluating LLMs, like [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). But if you have a framework that allows fast inference, then it could be conveinent to have a unified framework that also supports evaluation.
- **Pre-Training**: It would be safe to use verified frameworks for experiments like pre-training which requires a lot of compute costs. But if you have a framework that allows flexible configurations, various architectures, efficient parallelization, and evaluation, then it would be useful to have a support for pre-training, especially for small-scale explorative experiments. Actually pre-training is just one kind of possible experiments that can be implemented with Halite, like many post-training methods.

## Overall Structure

```
configs/            root directory for config files
src/halite          root directory for halite library
    data/           dataset loading and preprocessing related tools
    projects/       root directory for experiment and method related codes, like PPO, evaluation, etc
    transformers/   composable components for building transformer architectures
        infer/      inference engine for models composed using components above
scripts/            root directory for experiment and utility scripts
```

## Configuration

The aspect which Halite is most different from other frameworks is its configuration system. Many would find it is unfamiliar.

SlickConf, which is configuration system used in Halite is inspired by another configuration system, [Hydra](https://hydra.cc/), [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html), [Fiddle](https://github.com/google/fiddle). It allows you to use python code to define your configuration, and set python classes or functions in the config. But, importantly, it converts these classes or functions into a dictionary without python dependencies, and validates the config with pydantic.

For example, Llama 3 architecture is defined as follows in the [config file](https://github.com/rosinality/halite/blob/main/configs/models/llama/llama3_2_3b.py):

```python
from halite.transformers.position import Llama3RoPE, apply_rotary_emb

from ..transformer import transformer

conf = field()

dim = 3072
n_heads = 24
head_dim = dim // n_heads
context_len = 8192
use_complex_rope = True
qkv_split = True

transformer_config = field(
    vocab_size=128256,
    dim=dim,
    n_heads=n_heads,
    head_dim=head_dim,
    n_layers=28,
    n_key_value_heads=8,
    intermediate_size=8192,
    rms_norm_epsilon=1e-5,
    context_len=context_len,
    pos_embed=Llama3RoPE(
        head_dim,
        context_len,
        use_scaled_rope=True,
        use_complex=use_complex_rope,
    ),
    pos_embed_apply_fn=partial(apply_rotary_emb, use_complex=use_complex_rope),
    qkv_split=qkv_split,
    gated_ff_split=qkv_split,
)

conf.model = call[transformer](**transformer_config)
```

As you can use python classes and functions, you can compose your model without any framework-level code changes, just in your config. (For example, in above example you can change position embedding in your config.) Actually transformer itself is configured in the [config](https://github.com/rosinality/halite/blob/main/configs/models/transformer.py), composed of components defined in [transformers directory](https://github.com/rosinality/halite/tree/main/src/halite/transformers).

This allows you to extend the framework easily. For example, if you want to use a new optimizer, you can just assign it to configuration, like [this](https://github.com/rosinality/halite/blob/main/configs/lm/scale_383m_shampoo.py):

```python
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    FullyShardShampooConfig,
    PrecisionConfig,
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
```

In above example I used `DistributedShampoo` optimizer from [Optimizers](https://github.com/facebookresearch/optimizers) directly. You don't need any code changes to the Halite framework itself. You don't need to add configuration fields, `if` conditions, and so on. It is just a function assignment and composition.

You may feel it is too complex, unlike simple YAML-based configuration systems. But Halite is tightly coupled with this style of configuration, and it would be hard to use without it. (For example, transformers are consists with individual components, and it is hard to compose them to work without this style of configuration.)
