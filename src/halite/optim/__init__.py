from halite.optim import lr_scheduler


def group_parameters(model, weight_decay):
    decay_params = []
    no_decay_params = []

    for p in model.parameters():
        if p.ndim < 2:
            no_decay_params.append(p)

        else:
            decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return optim_groups
