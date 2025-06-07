from Layer import Layer
import numpy as np

# Our activation layer exists to apply a function over our input values (Y = f(x)) as a 
# way of getting the program to make stronger values have a larger impact. 
# You can think of this as just regular -1 to 1 correlation where 0 is none and the data's correlation increases as we approach either -1 or +1 
#                                                                                                                   ^ (strong - or + correlation)
# For this network, I'll use the hyperbolic tangent function (which we'll create in a seperate file) because it's both easy to implement and 
# easy to visualize (from an impact perspective). 
# AssemblyAI has a great video on youtube showing other examples called "Activation Functions in Neural Networks Explained"
class Activation(Layer):

    def __init__(self, function, function_derivative):
        self.function = function
        self.function_derivative = function_derivative

    def forward(self, input):
        self.input = input
        return self.function(self.input)
    
    def backward(self, output_gradient, lr):
        return np.multiply(output_gradient, self.function_derivative(self.input))
