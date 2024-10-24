from slickconf import load_config, instantiate
import torch

conf = load_config('scale_1.py')

print(conf)

model = instantiate(conf.model, _tags_={'n_vocab': 128})
model.init_weights(device='cpu')

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
model(x)
