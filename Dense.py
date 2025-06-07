from Layer import Layer
import numpy as np
# Our dense layer will do forward propagation based on the input value
# and it will do backward propagation based on dE/dY which will be determined later from a mean squared error function
#                                              ^                                          ^ this is where dE (dError) comes from
# For all of this to go smoothly, we should note what we have, what we need to calculate, and what for.
# we have an input (X) and dE/dY (output_gradient), we need dE/dW and dE/dB (parameter adjustments), and this is all to get dE/dX (input_ gradient)
#    
#           X --> |       | --> Y
#                 | Layer |
#       dE/dX <-- |       | <-- dE/dY
#       
#      (reminder of our layer diagram)

class Dense(Layer):
    # all of the mathematical "proofs", from here on out, can be found in the math section of this repo

    # To understand why we set these parameters the way we do (with random values according to Y's size and X's), 
    # lets think about what weight and bias stem from.      
    # W and b are parameters in y = wx + b. If we imagined, then, Y as a giant matrix , we would have : Y = [y1 = wX + wX +...+ b]
    #                                                              ^ (because we have a               :     [y2 = wX + wX +...+ b]
    #                                                                giant amount of points)          :     [y3 = wX + wX +...+ b] and so on...
    # 
    # A couple of notes here then: (1) there are multiple wXs, remember, because each input needs to map to the output at least once,
    #                                                                     ^ (you'll see this described through i and j in my math notes)
    # 
    # so each of the little ys will contain every x value multiplied by weight but only one bias
    #                              (2) you'll notice that, because of this, we now have a giant amount of Ws, Xs, and Bs.

    # This is why we use output and input size, because, the amount of Ws, if W were it's own individual matrix, would have as many rows as
    # there are little y values (output_size), and as many columns as there are x values (input_size). 
    # And B, that only appears once per row, will have as many rows as little y values, but only one column.
    
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    # Our forward is just going to be y = mx + b.
    def forward(self, input):
        self.input = input
        return np.dot(self.weight, self.input) + self.bias
    
    # From what was mentioned in the comments above, during backward propagation, we want to update adjust our parameters accordingly using dE/dY,
    # in order to return an input gradient (dE/dX).
    # But what is accordingly? 
    # Adjusting the parameters accordingly means abusing the definition of a derivative (finding the rate of change at a point) to
    # shift the weight and bias to do the opposite of this rate, eventually leaving us with the local minimum of the function and, subsequently,
    # the lowest possible error at that point. 
    # To visualize this, think of a ball at the top of a hill where we want to roll it down to our friend at the bottom (the lowest possible error). 
    # We obviously wouldn't try to roll the ball up the hill (up the slope), but down it.
    # So, we roll the ball down the hill (go down our slop, go negative to our derivative) and the bottom is our 
    # local minimum (the place where our slope zeros out, the place where error is minimized)
    # Therefore, we need the partial derivatives of weight and bias to be subtracted from our intial weight and bias (moving down the slope), 
    # but we also need to set how fast we're gonna roll this ball down the hill (learning rate) which we'll do by multiplying our 
    # partial derivatives by our set learning rate before we do the subtraction. 
    # If our learning rate is too high, we risk our ball rolling past our friend. 
    # If our learning rate is too low, we risk our ball stopping midway down the hill (not finding the optimal value before our epochs run out).
    
    def backward(self, output_gradient, lr):
        # Once again, the math section will show you how these equations were found (as well as why we don't create a new bias gradient).

        # We need to calculate dE/dX first; otherwise, we're putting our input gradient one step ahead.
        weight_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weight.T, output_gradient) #dE/dW 

        # Adjusting our parameters
        self.weight = self.weight - (weight_gradient * lr)
        self.bias = self.bias - (output_gradient * lr)
        return input_gradient
