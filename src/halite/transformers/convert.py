import re


def convert_state_dict(state_dict, mapping, config, reverse=False):
    new_sd = {}

    for orig_key, tensor in state_dict.items():
        for pattern, convert in mapping.items():
            for attr_key, attr_val in convert["map"].items():
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

                if isinstance(attr_val, tuple):
                    attr_val, process = attr_val

                    new_sd[target_key + f".{attr_val}"] = process(config, tensor)

                else:
                    new_sd[target_key + f".{attr_val}"] = tensor

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


def get_find_pattern(pattern: str, attr=None):
    count = pattern.count("#")
    find_pattern = pattern.replace("#", r"([0-9]+)")

    if attr is not None:
        find_pattern += f".{attr}$"

    else:
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
