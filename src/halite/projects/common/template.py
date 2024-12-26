import collections
from typing import Callable

from jinja2 import Environment, StrictUndefined


class Template:
    def __init__(self, template):
        self.template = template

    def __call__(self, *args, **kwargs):
        return self.template.render(*args, **kwargs)


def get_render_fn(
    template: str,
    trim_blocks: bool = True,
    lstrip_blocks: bool = True,
    keep_trailing_newline: bool = True,
    filters: dict[str, Callable] | None = None,
) -> Callable[[str], str]:
    env = Environment(
        trim_blocks=trim_blocks,
        lstrip_blocks=lstrip_blocks,
        keep_trailing_newline=keep_trailing_newline,
        undefined=StrictUndefined,
    )

    if filters is not None:
        for filter_name, filter in filters.items():
            env.filters[filter_name] = filter

    return Template(env.from_string(template))


class SimpleFormat:
    def __init__(self, template: str):
        self.template = template

    def __call__(self, *args, **kwargs):
        return self.template.format(*args, **kwargs)


def simple_format(template: str) -> Callable[[str], str]:
    return SimpleFormat(template=template)
