from neuron import Neuron

class Layer():
    def __init__(self, size, num_inputs, activation=None):
        self.neurons = [Neuron(num_inputs, activation) for _ in range(size)]
    
    def fire(self, inputs):
        return [n.fire(inputs) for n in self.neurons]
    
        
