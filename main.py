import random as rand

class AdvancedGate():
    def __init__(self, op):
        self.op = op

    def compute(self, inputs, weights = []):
        res = Value(0)
        if self.op == "dot":
            res = Value(0)
            plus = ElementaryGate("+")
            mul = ElementaryGate("*")
            for i, w in zip(inputs, weights):
                res = plus.compute([res, mul.compute([i,w])])
        elif self.op == "random":
            res = Value(0)
            plus = ElementaryGate("+")
            sub = ElementaryGate("-")
            mul = ElementaryGate("*")
            ops = [plus, sub, mul]
            for _ in range(5):
                choice = rand.choice(ops)
                res += choice.compute(inputs)
        
        return res

class ElementaryGate():
    def __init__(self, op):
        self.op = op

    def compute(self, inputs):
        res = Value(0)
        if self.op == "+":
            for i in inputs:
                res += i
            return res
        elif self.op == "-":
            for i in inputs:
                res -= i
            return res
        elif self.op == "*":
            res = Value(1)
            for i in inputs:
                res *= i
        
        return res

class Value():
    def __init__(self, data, parents = ()):
        self.data = data
        self.parents = parents
        self.grad = 0.0
        self.backward = lambda: None

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

def print_grads(grads):
    for g in grads:
        print(f"{g} grad=", g.grad)

def main():
    a = Value(10)
    b = Value(20)
    w1 = Value(15)
    w2 = Value(5)
    plus = ElementaryGate("+")
    mul = ElementaryGate("*")
    dot = AdvancedGate("dot")
    random = AdvancedGate("random")
    y = plus.compute((a,b,69))
    z = mul.compute((y, plus.compute((b,b,420))))
    t = dot.compute((z,b), (w1, w2))
    r = random.compute((t, a))

    # backprop
    r.grad = 1
    r.backward()
    print_grads((a,b,y,z,w1,w2,t))

if __name__ == "__main__":
    main()