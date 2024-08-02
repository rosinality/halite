from slickconf import field, call, tag

from .base import transformer


conf = field(model=call[transformer](tag("n_vocab"), 96, 4, 3, call[int](96 * 3.5), 2048, rms_norm_epsilon=1e-6, post_norm=True))
