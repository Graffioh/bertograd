import random

def relu(x):
    return max(0, x)

class Neuron():
    def __init__(self, num_inputs, activation=None):
        self.weights = [random.random() for _ in range(num_inputs)]
        self.bias = random.random() 
        self.activation = activation if activation else relu
    
    def fire(self, inputs):
        dot_res = 0
        for i, w in zip(inputs, self.weights):
            dot_res += i*w
        
        return self.activation(dot_res + self.bias)

    