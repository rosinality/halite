from typing import Any, Callable, Dict, Optional


class Policy:
    arch: Callable
    weight_mappings = Dict[str, Dict[str, Any]]
    prefixes: Dict[str, Dict[str, Optional[str]]] = None
    pre_unshard_process: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    post_shard_process: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
