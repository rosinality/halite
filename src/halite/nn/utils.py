def get_model_dtype(model):
    dtype = None

    for p in model.parameters():
        dtype = p.dtype

        if p.is_floating_point():
            return dtype

    return dtype