from halite.nn.utils import get_model_dtype


class ModelMixin:
    @property
    def dtype(self):
        return get_model_dtype(self)
