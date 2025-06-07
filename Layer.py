import numpy as np
# The function of a layer in a neural network is to take a series of inputs and match each one of them to a series of outputs once per.
# It will then take these inputs and run them through a series of calculations or a function.

# Thus, for our layers, we need them to work forward, taking in an input and giving an output (forward propagation)
# and we need them to work backward, taking in an output gradient and returning and input gradient (backward propagation)
# specifically, this last step needs to update the paramters that affect the output (weight and bias) which is the process of "learning"
#     ^ this whole line will be discussed further as we create the dense layer      ^ w and b from 'y = wx + b'


# lets start by defining a base layer class which will serve as the template for all our other layers
# and lets define three overview functions of what we need according to the parenthesis above

#           X --> |       | --> Y
#                 | Layer |
#       dE/dX <-- |       | <-- dE/dY
#       
#         (A diagram of any layer)
class Layer:

    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_gradient, lr):
        pass 
