class Value():
    def __init__(self, data, parents = (), label = ""):
        self.data = data
        self.parents = parents
        self.grad = 0.0
        self.backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"LABEL: {self.label} | DATA: {self.data} | GRAD: {self.grad}"

    def __add__(self, val):
        if not val:
            print("val is null, can't perform ADD operation")
            return

        val = val if isinstance(val, Value) else Value(val)
        out = Value(self.data + val.data, (self, val))

        def _backward():
            self.grad += 1 * out.grad
            val.grad += 1 * out.grad

            self.backward()
            val.backward()

        out.backward = _backward
        return out
    
    def __radd__(self, val):
        return val.__add__(self.data)

    def __sub__(self, val):
        if not val:
            print("val is null, can't perform SUB operation")
            return

        val = val if isinstance(val, Value) else Value(val)
        out = Value(self.data - val.data, (self, val))

        def _backward():
            self.grad += 1 * out.grad
            val.grad += -1 * out.grad

            self.backward()
            val.backward()

        out.backward = _backward
        return out

    def __rsub__(self, val):
        return val.__sub__(self.data)

    def __mul__(self, val):
        if not val:
            print("val is null, can't perform MUL operation")
            return

        val = val if isinstance(val, Value) else Value(val)
        out = Value(self.data * val.data, (self, val))

        def _backward():
            local_grad_self = val.data
            local_grad_val = self.data
            self.grad += local_grad_self * out.grad
            val.grad += local_grad_val * out.grad

            self.backward()
            val.backward()

        out.backward = _backward
        return out

    def __rmul__(self, val):
        return val.__mul__(self.data)