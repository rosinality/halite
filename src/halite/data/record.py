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
