from Activation import Activation
import numpy as np

# As I mentioned in Activations, this is where we'll our tanH function (activation function) and its derivative.
# Lucky for us, numpy has a built in tanH, so we just need to take a very simple derivative
class Tanh(Activation):

    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_derivative = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_derivative)
