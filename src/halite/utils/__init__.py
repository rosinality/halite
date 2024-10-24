import torch
from packaging import version


def is_torch_2() -> bool:
    """Check the PyTorch version and compared it with version 2.0.0.

    Returns:
        bool: Return True if current version is greater than or equal to 2.0.0 else False
    """
    return version.parse(torch.__version__) >= version.parse("2.0.0")


def is_torch_2_1() -> bool:
    """Check the PyTorch version and compared it with version 2.0.0.

    Returns:
        bool: Return True if current version is greater than or equal to 2.0.0 else False
    """
    return version.parse(torch.__version__) >= version.parse("2.1.0")


def get_torch_dtype(dtype):
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype in ["float32", "torch.float32", "fp32"]:
        return torch.float32

    if dtype in ["float16", "torch.float16", "half", "fp16"]:
        return torch.float16

    if dtype in ["bfloat16", "bfloat", "torch.bfloat16", "bf16"]:
        return torch.bfloat16

    raise ValueError(f"passed dtype {dtype} is not an appropriate dtype")
