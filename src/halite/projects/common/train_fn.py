from collections.abc import Mapping


def get_fields(record, fields):
    if isinstance(record, Mapping):
        if isinstance(fields, str):
            return {fields: record[fields]}

        return {field: record[field] for field in fields}

    else:
        if isinstance(fields, str):
            return {fields: getattr(record, fields)}

        return {field: getattr(record, field) for field in fields}


def basic_train_step(data, model, criterion):
    model_input = {}

    if "input" in data:
        model_input["input_ids"] = data["input"]

    if "position_ids" in data:
        model_input["position_ids"] = data["position_ids"]

    if "document_offsets" in data:
        model_input["document_offsets"] = data["document_offsets"]

    out = model(**model_input)
    loss, loss_dict = criterion(out.logits, data["target"])

    return loss, loss_dict
