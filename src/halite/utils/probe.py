from collections import defaultdict
from enum import Enum
import json
import functools
from typing import Any
from pathlib import Path
import uuid

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils.module_tracker import ModuleTracker

PROBING_ENABLED = False


QUANTILES = [
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999,
    0.999999,
    0.9999999,
]


def tensor_to_list(input):
    if not isinstance(input, torch.Tensor):
        return input

    return input.tolist()


@functools.cache
def get_quantiles(device: torch.device, dtype) -> torch.Tensor:
    return torch.tensor(QUANTILES, device=device, dtype=dtype)


def get_stats(x_: torch.Tensor, remove_inf=False) -> dict[str, Any]:
    if x_.dtype not in [torch.float, torch.double, torch.float16, torch.bfloat16]:
        return {}
    x = x_.flatten()
    if remove_inf:
        x = x[x.abs() < float("inf")]
    if x.dtype is not torch.double:
        x = x.float()
    xabs = x.abs()
    quantiles = get_quantiles(x.device, x.dtype)
    mean = x.mean()
    std = x.std()
    return {
        "shape": tuple(x_.shape),
        "mean": mean,
        "std": std,
        "skew": (((x - mean) / std) ** 3).double().mean(),
        "kurtosis": (((x - mean) / std) ** 4).double().mean(),
        "abs.mean": xabs.mean(),
        "max": x.max(),
        "min": x.min(),
        # Note: `quantile` takes at most 2**24 elements, see
        # https://github.com/pytorch/pytorch/issues/64947
        "quantiles": torch.quantile(x[: 2**24], quantiles),
    }


class LinearBwType(Enum):
    DW = 1
    DX = 2
    UNKNOWN = 3


class Probe(TorchDispatchMode):
    def __init__(self, output_file: str | None = None, verbose=False):
        self.output_file = Path(output_file) if output_file is not None else None
        self.output_tensors_tmpdir = None
        self.mod_tracker = ModuleTracker()
        self.store: dict[str, dict[str, Any]] = {}
        self.linear_data: dict[str, tuple[Any, Any, Any, Any, Any]] = {}
        self.uid_to_path: dict[str, str] = {}
        self.count_per_path: dict[str, int] = defaultdict(int)
        self.metadata = {}
        self.enabled = False
        self.verbose = verbose

    def setup_logging(self):
        if self.output_file is not None:
            self.output_file.parent.mkdir(exist_ok=True)
            self.output_tensors_tmpdir = (
                self.output_file.parent / f"tensors-{str(uuid.uuid4())[:8]}"
            )
            self.output_tensors_tmpdir.mkdir(exist_ok=True)

    def __enter__(self):
        global PROBING_ENABLED
        assert not self.enabled, "entered probe mode twice"

        self.mod_tracker.__enter__()
        super().__enter__()
        self.enabled = True
        PROBING_ENABLED = True

        self.setup_logging()

        return self

    def __exit__(self, *args):
        global PROBING_ENABLED
        assert self.enabled, "exiting probe mode without entering it"

        super().__exit__(*args)
        self.mod_tracker.__exit__(*args)

        self.flush_and_clear()
        PROBING_ENABLED = False
        self.enabled = False

    def flush_and_clear(self):
        if self.output_file is not None:
            dump_data = tree_map(tensor_to_list, self.store)

            with self.output_file.open("a") as f:
                json.dump(
                    {
                        "data": dump_data,
                        "meta": self.metadata,
                        "quantiles": QUANTILES,
                    },
                    f,
                )
                f.write("\n")

        if self.output_tensors_tmpdir is not None:
            dump_dir = self.output_tensors_tmpdir.parent / "tensors"
            dump_dir.mkdir(exist_ok=True)
            dirname = ""

            if "it" in self.metadata:
                dirname = f"it{int(self.metadata['it']):010}"

            if dirname == "" or (dump_dir / dirname).exists():
                n_files = len(list(dump_dir.glob(f"{dirname}v*")))
                dirname = f"{dirname}v{n_files}"

            dump_dir = dump_dir / dirname
            self.output_tensors_tmpdir.rename(dump_dir)
            self.output_tensors_tmpdir = None

        self.store.clear()
        self.count_per_path.clear()
        self.uid_to_path.clear()

    def log_tensor(self, name, tensor, **kwargs):
        self.store[name] = get_stats(tensor, **kwargs)

        if self.output_tensors_tmpdir is not None:
            name_sane = name.replace("::", "__").replace("/", "")
            torch.save(tensor, self.output_tensors_tmpdir / f"{name_sane}.pt")

    def find_bw_path_and_type(self, path, out, args):
        def is_path_correct_dw(path):
            in_shape, w_shape, out_shape, input_sm, weight_sm = self.linear_data[path]

            return out.shape == (w_shape[1], w_shape[0]) and torch.allclose(
                input_sm, args[1][:4, :4]
            )

        def is_path_correct_dx(path):
            in_shape, w_shape, out_shape, input_sm, weight_sm = self.linear_data[path]

            return out.shape == in_shape and torch.allclose(weight_sm, args[1][:4, :4])

        if path in self.linear_data:
            if is_path_correct_dw(path):
                return path, LinearBwType.DW

            if is_path_correct_dx(path):
                return path, LinearBwType.DX

        for candidate_path in self.mod_tracker.parents:
            if candidate_path not in self.linear_data:
                continue

            if is_path_correct_dw(candidate_path):
                return candidate_path, LinearBwType.DW

            if is_path_correct_dx(candidate_path):
                return candidate_path, LinearBwType.DX

        return path, LinearBwType.UNKNOWN

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        path = None

        for p in self.mod_tracker.parents:
            if p == "Global":
                continue

            if path is None or len(p) > len(path):
                path = p

        if path is None:
            path = "Global"

        path = path.replace("._checkpoint_wrapped_module", "")
        out = func(*args, **kwargs)

        if func._overloadpacket in [torch.ops.aten.addmm, torch.ops.aten.mm]:
            if not self.mod_tracker.is_bw:
                if func._overloadpacket == torch.ops.aten.addmm:
                    bias, input, weight = args[:3]

                else:
                    input, weight = args[:2]

                self.log_tensor(f"{path}::in", input)
                self.log_tensor(f"{path}::weight", weight)
                self.log_tensor(f"{path}::out", out)
                self.linear_data[path] = (
                    input.shape,
                    weight.shape,
                    out.shape,
                    input[:4, :4].clone(),
                    weight[:4, :4].clone(),
                )

            elif func._overload_packet == torch.ops.aten.mm:
                new_path, bwtype = self.find_bw_path_and_type(path, out, args)
                if new_path != path:
                    if self.verbose:
                        print(f"probe: fixing path `{path}` -> `{new_path}`")

                    path = new_path

                if bwtype == LinearBwType.DW:
                    self.log_tensor(f"{path}::w.g", out)

                elif bwtype == LinearBwType.DX:
                    self.log_tensor(f"{path}::in.g", out)
                    self.log_tensor(f"{path}::out.g", args[0])

        return out
