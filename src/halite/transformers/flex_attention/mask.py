class ComposeMask:
    def __init__(self, *masks):
        self.masks = masks
        self.mask_inputs = [mask.inputs for mask in self.masks]

        inputs = []
        for mask in self.masks:
            inputs.extend(mask.inputs)

        self.inputs = tuple(set(inputs))

    def __call__(self, **kwargs):
        masks = []

        for mask_fn, inputs in zip(self.masks, self.mask_inputs):
            mask_inputs = {input: kwargs[input] for input in inputs}
            masks.append(mask_fn(**mask_inputs))

        def mask_mod(b, h, q_idx, kv_idx):
            result_mask = None

            for mask in masks:
                if result_mask is None:
                    result_mask = mask(b, h, q_idx, kv_idx)

                else:
                    result_mask = result_mask & mask(b, h, q_idx, kv_idx)

            return result_mask

        return mask_mod
