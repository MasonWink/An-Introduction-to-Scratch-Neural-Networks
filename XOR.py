from Dense import Dense
from hyperbolicTangent import Tanh
from meanSquare import meanSquaredError, derivativeMeanSquared
import numpy as np
import matplotlib.pyplot as plt

# A XOR table is that which includes two values of 0 or 1 in which the input of the same value at one time returns a 0,
# while a unique set of inputs returns a 1

# Let's start by defining our possible inputs and the values they would output (reshaping them to fit our algorithm).
X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Now, we can imagine our neural network as two inputs (x1 and x2 from the table)
# of which, their combinations will be three ([0,0], [0, 1], [1, 1]).
# Finally, whichever combination we have will decide what our one output value will be (y1)
# This order of operations leads us down a path of defined input and output sizes
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]
epochs = 10000
lr = .05

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        predict = x
        for layer in network:
            predict = layer.forward(predict)
    
        error = error + meanSquaredError(y, predict)
        gradient = derivativeMeanSquared(y, predict)
        for layer in reversed(network):
            gradient = layer.backward(gradient, lr)
        
    error /= len(X)
    print("%d/%d, error: %f" % (e + 1, epochs, error)) 


