import random as rand
from value import Value

# NOT USED, DID THIS JUST TO EXPERIMENT A BIT
class AdvancedGate():
    def __init__(self, op):
        self.op = op

    def compute(self, inputs, weights = []):
        res = Value(0)
        if self.op == "dot":
            plus = ElementaryGate("+")
            mul = ElementaryGate("*")
            for i, w in zip(inputs, weights):
                res = plus.compute([res, mul.compute([i,w])])
        elif self.op == "random":
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