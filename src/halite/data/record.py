from collections.abc import Sequence


class Record(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "_meta_" not in self:
            self["_meta_"] = {}

        self.__dict__ = self

    def to(self, *args, **kwargs):
        res = {}

        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)

            elif isinstance(v, list):
                res_list = []

                for e in v:
                    if hasattr(e, "to"):
                        res_list.append(e.to(*args, **kwargs))

                    else:
                        res_list.append(e)

                v = res_list

            res[k] = v

        return self.__class__(**res)

    def slice(self, slice):
        new_record = Record()

        for k, v in self.items():
            if isinstance(v, Sequence):
                new_record[k] = v[slice]

            else:
                new_record[k] = v

        return new_record

    def unbind(self, keys=None):
        field_length = -1

        for key in keys:
            val = self[key]

            if isinstance(val, Sequence):
                if field_length != -1 and field_length != len(val):
                    raise ValueError("all fields must have the same length")

                field_length = len(val)

        records = []
        for i in range(field_length):
            record = Record()

            for key in keys:
                val = self[key]

                if isinstance(val, Sequence):
                    record[key] = val[i]

                else:
                    record[key] = val

            records.append(record)

        return records
