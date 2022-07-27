import numpy as np

class Neuron(object):
   def __init__(self, num_inputs, activation_fn):
     super().__init__()
     # Randomly initializing the weight vector and bias value:
     self.W = np.random.rand(num_inputs)
     self.b = np.random.rand(1)
     self.activation_fn = activation_fn
   def forward(self, x):
     """Forward the input signal through the neuron."""
     z = np.dot(x, self.W) + self.b
     return self.activation_function(z)
     
np.random.seed(42)
# Random input column array of 3 values (shape = `(1, 3)`)
x = np.random.rand(3).reshape(1, 3)

step_fn = lambda y: 0 if y <= 0 else 1
perceptron = Neuron(num_inputs=x.size, activation_fn=step_fn)
out = perceptron.forward(x)
