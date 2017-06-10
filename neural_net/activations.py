import numpy as np


def sigmoid(x=None,deriv=False):
    ''' Sigmoid function (used for hidden neurons)
    '''
    if deriv == True:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))


def purelin(x=None,deriv=False):
    ''' Linear (usually used in output neurons)
    '''
    if deriv == True:
        return 1 
    return x 


# Vectorize the functions to accept vectors and return vectors
sigmoid = np.vectorize(sigmoid)
purelin = np.vectorize(purelin)
