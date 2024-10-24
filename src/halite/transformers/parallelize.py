from collections import defaultdict

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.tensor.parallel import parallelize_module

from halite.parallel.fsdp import apply_fsdp


def parallelize(
    model,
    mesh,
    parallel_dims,
    param_dtype,
    reduce_dtype,
    tensor_parallel_config=None,
    activation_checkpointing=False,
    activation_checkpointing_config=None,
    compile=False,
):
    if parallel_dims.tp_enabled:
        apply_tp(model, mesh["tp"], tensor_parallel_config)

    if activation_checkpointing:
        apply_activation_checkpointing(model, activation_checkpointing_config)

    if compile:
        apply_compile(model)

    if parallel_dims.dp_mesh_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh = mesh["dp_replicate", "dp_shard"]

        else:
            dp_mesh = mesh["dp"]

        apply_fsdp(model, dp_mesh, param_dtype, reduce_dtype)

    return model


SELECTIVE_SAVE_LIST = {
    torch.ops.aten.mm.default,
    # torch.ops.aten._scaled_dot_product_efficient_attention.default,
    # torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def activation_checkpointing(module, config):
    if config["mode"] == "full":
        return checkpoint_wrapper(module, preserve_rng_state=False)

    assert (
        config["mode"] == "selective"
    ), "Activation checkpointing mode should be full or selective"

    selective_mode = config.get("selective", "1")
    use_selective_op = selective_mode == "op"

    if not use_selective_op:
        use_selective_layer = int(selective_mode)

    if use_selective_op:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def make_custom_policy(meta):
            def custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"

                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1

                to_save = func in SELECTIVE_SAVE_LIST and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )

                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)

            return create_selective_checkpoint_contexts(make_custom_policy(meta))

        return checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )


def apply_activation_checkpointing(model, config):
    for i, block in model.blocks.named_children():
        block = activation_checkpointing(block, config)
        model.blocks.register_module(i, block)


def apply_tp(model, tp_mesh, config):
    config = {} if config is None else config

    enable_async_tp = config.pop("enable_async_tp", False)
    plans = get_parallelize_plan(model, config)

    parallelize_module(model, tp_mesh, plans)

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)


def apply_compile(model):
    for i, block in model.blocks.named_children():
        block = torch.compile(block, fullgraph=True)
        model.blocks.register_module(i, block)


def get_parallelize_plan(model, config):
    model_plans = {}

    for path, module in model.named_modules():
        if not hasattr(module, "parallelize_plan"):
            continue

        module_plan = module.parallelize_plan(**config)
        plan_dict = {}
        for key, plan in module_plan.items():
            if key == ".":
                key = path

            else:
                if len(path.strip()) > 0:
                    key = path + "." + key

            plan_dict[key] = plan

        model_plans.update(plan_dict)

    return model_plans
