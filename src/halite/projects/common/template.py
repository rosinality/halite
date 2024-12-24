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
) -> Callable[[str], str]:
    env = Environment(
        trim_blocks=trim_blocks,
        lstrip_blocks=lstrip_blocks,
        keep_trailing_newline=keep_trailing_newline,
        undefined=StrictUndefined,
    )

    return Template(env.from_string(template))
