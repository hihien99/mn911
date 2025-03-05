class tuple_inst:
    def __init__(self, elem_type: type = str, delimiter: str = ','):
        self.elem_type = elem_type
        self.delimiter = delimiter

    def __call__(self, vs):
        if isinstance(vs, str):
            if vs.startswith('[') and vs.endswith(']'):
                vs = vs[1:-1]
            elif vs.startswith('(') and vs.endswith(')'):
                vs = vs[1:-1]
            elif vs.startswith('{') and vs.endswith('}'):
                vs = vs[1:-1]
            return tuple(self.elem_type(v) for v in vs.replace(' ', '').split(self.delimiter))
        else:
            return tuple(self.elem_type(v) for v in vs)
