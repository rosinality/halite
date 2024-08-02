from dataclasses import dataclass
import re
from typing import Any, Callable, TypeAlias

import torch
from pydantic import BaseModel
from slickconf import instantiate


def convert_state_dict(state_dict, mapping, reverse=False):
    new_sd = {}

    for orig_key, tensor in state_dict.items():
        for pattern, convert in mapping.items():
            for attr_key, _ in convert["placement"].items():
                if reverse:
                    find_pattern, count = get_find_pattern(convert["key"], attr_key)

                else:
                    find_pattern, count = get_find_pattern(pattern, attr_key)

                found = re.findall(f"({find_pattern})", orig_key)

                if len(found) == 0:
                    continue

                found_str = found[0]

                if isinstance(found_str, tuple):
                    found_str = found_str[0]

                if reverse:
                    convert_pattern = get_replace_pattern(pattern, count)

                else:
                    convert_pattern = get_replace_pattern(convert["key"], count)

                target_key = re.sub(find_pattern, convert_pattern, found_str)

                new_sd[target_key + f".{attr_key}"] = tensor

    return new_sd


def convert_to_hf_state_dict(state_dict, mapping, config):
    new_sd = {}

    for orig_key, tensor in state_dict.items():
        for pattern, convert in mapping.items():
            for attr_key, attr_val in convert["map"].items():
                find_pattern, count = get_find_pattern(convert["key"], attr_key)
                found = re.findall(f"({find_pattern})", orig_key)

                if len(found) == 0:
                    continue

                found_str = found[0]

                if isinstance(found_str, tuple):
                    found_str = found_str[0]

                convert_pattern = get_replace_pattern(pattern, count)
                target_key = re.sub(find_pattern, convert_pattern, found_str)

                if isinstance(attr_val, tuple):
                    attr_val, process = attr_val

                    new_sd[target_key + f".{attr_val}"] = process(config, tensor)

                else:
                    new_sd[target_key + f".{attr_val}"] = tensor

    return new_sd


def get_find_pattern(pattern: str, attr=None, ensure_ending=True):
    count = pattern.count("#")
    find_pattern = pattern.replace("#", r"([0-9]+)")

    if attr is not None:
        find_pattern += f".{attr}"

    if ensure_ending:
        find_pattern += "$"

    return find_pattern, count


def get_replace_pattern(pattern: str, count: int):
    for i in range(count):
        pattern = pattern.replace("#", f"\\{i + 1}", 1)

    return pattern


def get_matched_keys(keys, pattern: str, replace_pattern=None):
    find_pattern, count = get_find_pattern(pattern)

    found_keys = {}

    for key in keys:
        found = re.findall(f"({find_pattern})", key)

        if len(found) == 0:
            continue

        found_str = found[0]

        if isinstance(found_str, tuple):
            found_str = found_str[0]

        target_key = found_str

        if replace_pattern is not None:
            replace_pattern = get_replace_pattern(replace_pattern, count)
            target_key = re.sub(find_pattern, replace_pattern, found_str)

        found_keys[found_str] = target_key

    found_keys = sorted(found_keys.items(), key=lambda x: x[0])

    if replace_pattern is not None:
        return tuple(zip(*found_keys))

    return tuple([key[0] for key in found_keys])


Checkpoint: TypeAlias = dict[str, Any]


@dataclass
class Policy:
    weight_maps: dict[str, dict[str, Any]]
    to_halite_preprocess: Callable[[Checkpoint, dict], Checkpoint] | None = None
    to_halite_postprocess: Callable[[Checkpoint, dict], Checkpoint] | None = None
    from_halite_preprocess: Callable[[Checkpoint, dict], Checkpoint] | None = None
    from_halite_postprocess: Callable[[Checkpoint, dict], Checkpoint] | None = None

    @classmethod
    def from_pydantic(cls, config: BaseModel):
        def has_processor(processor_name):
            if hasattr(config, processor_name):
                return instantiate(getattr(config, processor_name))

            return None

        return cls(
            weight_maps=config.weight_maps,
            to_halite_preprocess=has_processor("to_halite_preprocess"),
            to_halite_postprocess=has_processor("to_halite_postprocess"),
            from_halite_preprocess=has_processor("from_halite_preprocess"),
            from_halite_postprocess=has_processor("from_halite_postprocess"),
        )


def extract_placement(weight_maps, mode="to_halite"):
    placements = {}

    for key, maps in weight_maps.items():
        if "placement" not in maps:
            continue

        if mode == "to_halite":
            placements[maps["key"]] = maps["placement"]

        elif mode == "from_halite":
            placements[key] = maps["placement"]

    return placements


def unshard_tensor(state_dicts, key, placement):
    if isinstance(placement, str) and placement == "replicate":
        for sd in state_dicts:
            if key in sd:
                return sd[key]

    if len(state_dicts) == 1:
        return state_dicts[0][key]

    assert (
        len(placement) == 2
    ), "placement should be replicate or shard with sharding dims"

    shard_dim = placement[1]

    return torch.cat([sd[key] for sd in state_dicts], shard_dim)


def convert_to_halite(
    state_dicts: Checkpoint | list[Checkpoint],
    model_config: dict[str, Any],
    policy: Policy,
):
    if not isinstance(state_dicts, list):
        state_dicts = [state_dicts]

    state_dicts = [convert_state_dict(sd, policy.weight_maps) for sd in state_dicts]

    if policy.to_halite_preprocess is not None:
        state_dicts = [
            policy.to_halite_preprocess(model_config, state_dict)
            for state_dict in state_dicts
        ]

    new_state_dict = {}

    placements = extract_placement(policy.weight_maps, mode="to_halite")

    merged_states = {}
    for state_dict in state_dicts:
        for key, tensor in state_dict.items():
            merged_states[key] = tensor

    for key, tensor in merged_states.items():
        for pattern, placement in placements.items():
            for target, place in placement.items():
                find_pattern, _ = get_find_pattern(pattern, target)
                found = re.findall(f"({find_pattern})", key)

                if len(found) == 0:
                    continue

                new_state_dict[key] = unshard_tensor(state_dicts, key, place)

    if policy.to_halite_postprocess is not None:
        new_state_dict = policy.to_halite_postprocess(model_config, new_state_dict)

    return new_state_dict


def convert_from_halite(
    state_dict: Checkpoint, model_config: dict[str, Any], policy: Policy
):
    pass


def convert_checkpoint(
    state_dicts: Checkpoint | list[Checkpoint],
    model_config: dict[str, Any],
    policy: BaseModel | Policy,
    mode: str = "to_halite",
):
    if not isinstance(policy, Policy):
        policy = Policy.from_pydantic(policy)

    if mode == "to_halite":
        return convert_to_halite(state_dicts, model_config, policy)

    elif mode == "from_halite":
        return convert_from_halite(state_dicts, model_config, policy)

    else:
        raise ValueError(
            f"unsupported mode: {mode}, supported modes: to_halite, from_halite"
        )
