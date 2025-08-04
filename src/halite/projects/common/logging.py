def hparam_to_name(hparams: dict, format: str) -> str:
    return format.format(**hparams)
