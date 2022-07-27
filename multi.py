import numpy as np
class FullyConnectedLayer(object):
 """A simple fully-connected NN layer.
 Args:
 num_inputs (int): The input vector size/number of input values.
 layer_size (int): The output vector size/number of neurons.
 activation_fn (callable): The activation function for this layer.
 Attributes:
 W (ndarray): The weight values for each input.
 b (ndarray): The bias value, added to the weighted sum.
 size (int): The layer size/number of neurons.
 activation_fn (callable): The neurons' activation function.
 """
   def __init__(self, num_inputs, layer_size, activation_fn):
     super().__init__()
     # Randomly initializing the parameters (using a normal distribution this time):
     self.W = np.random.standard_normal((num_inputs, layer_size))
     self.b = np.random.standard_normal(layer_size)
     self.size = layer_size
     self.activation_fn = activation_fn
    
   def forward(self, x):
     """Forward the input signal through the layer."""
     z = np.dot(x, self.W) + self.b
     return self.activation_fn(z)
   
  np.random.seed(42)
  # Random input column-vectors of 2 values (shape = `(1, 2)`):
  x1 = np.random.uniform(-1, 1, 2).reshape(1, 2)
  x2 = np.random.uniform(-1, 1, 2).reshape(1, 2)
  
  relu_fn = lambda y: np.maximum(y, 0) 
  # Defining our activation function
  layer = FullyConnectedLayer(2, 3, relu_fn)
  
  x12 = np.concatenate((x1, x2)) 
  # stack of input vectors, of shape `(2, 2)`
  out12 = layer.forward(x12)

