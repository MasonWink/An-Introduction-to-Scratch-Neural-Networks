import numpy as np

# This is the final part of network before we actually solve the problem
# We need to implement a mean squared error function and it's derivative in order to 
# have the dE/dY that serves as the basis for all our other equations and updates

def meanSquaredError(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

def derivativeMeanSquared(y_actual, y_predicted):
    n = np.size(y_actual)
    return (2/n) * (y_predicted - y_actual)
