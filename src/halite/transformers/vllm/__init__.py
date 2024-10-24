from copy import deepcopy
from unittest.mock import NonCallableMagicMock

from meshfn import instantiate
from meshfn.nn import get_meta
from meshfn.nn.load import init_empty_weights
from meshfn.transformers import load_model


def share_parameters(model, vllm_model):
    for n, p in vllm_model.named_parameters():
        if "." in n:
            module_key, weight_key = n.rsplit(".", 1)
            module = model.get_submodule(module_key)
            vllm_module = vllm_model.get_submodule(module_key)

        else:
            module = model
            vllm_module = vllm_model
            weight_key = p

        vllm_module.register_parameter(weight_key, getattr(module, weight_key))

    return vllm_model


def instantiate_vllm(conf, parallel_context, device, dtype):
    try:
        # if conf calls meshfn.transformers.load_model
        vllm_model = instantiate(
            conf,
            parallel_context=parallel_context,
            device=device,
            dtype=dtype,
            load_checkpoint=False,
            vllm=True,
        )

    except (TypeError, AttributeError) as e:
        # else, conf calls model constructing function directly
        vllm_model = instantiate(
            conf,
            parallel_context=parallel_context,
            device=device,
            dtype=dtype,
            vllm=True,
        )

    return vllm_model


def build_vllm_model(
    model,
    tokenizer,
    model_path_or_conf=None,
    parallel_context=None,
    device=None,
    dtype=None,
    logger=None,
    **vllm_kwargs,
):
    vllm_model = None

    if isinstance(model_path_or_conf, str):
        with init_empty_weights(include_buffers=False):
            vllm_model = load_model(
                model_path_or_conf,
                parallel_context=parallel_context,
                device=device,
                dtype=dtype,
                load_checkpoint=False,
                vllm=True,
            )

    if hasattr(model_path_or_conf, "keys") and hasattr(model_path_or_conf, "values"):
        with init_empty_weights(include_buffers=False):
            vllm_model = instantiate_vllm(
                model_path_or_conf, parallel_context, device, dtype
            )

    if vllm_model is None:
        arch = get_meta(model, "arch")
        arch = deepcopy(arch)

        try:
            arch.pop("[meta]")

        except KeyError:
            pass

        if arch is not None:
            with init_empty_weights(include_buffers=False):
                vllm_model = instantiate_vllm(arch, parallel_context, device, dtype)

        else:
            raise ValueError(
                "model should be path to the checkpoint, "
                "or instantiatable config, "
                "or model instance that has arch meta data"
            )

    vllm_model = share_parameters(model, vllm_model)

    from meshfn.transformers.builder.vllm.engine import vLLM

    return vLLM(
        vllm_model,
        tokenizer,
        parallel_context=parallel_context,
        logger=logger,
        **vllm_kwargs,
    )
