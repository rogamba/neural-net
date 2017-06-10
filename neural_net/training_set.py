import numpy as np


def regfunc(x1,x2):
    ''' z = x1^2 + 2x1x2 + 2x2^2 + 1
    '''
    return x1**2 + 2*x1*x2 + 2*x2**2 + 1



def gen_training_set():
    ''' Returns a key value tuple of input:output
        ( [[x1],[x2]] , z )
    '''
    training_domain=[]
    for x1 in np.arange(-2,2,0.2):
        for x2 in np.arange(-2,2,0.2):
            training_domain.append( (x1,x2) )
    training_set = [ ( np.matrix([x1,x2]).transpose() , round(regfunc(x1,x2),1) ) for (x1,x2) in training_domain ]
    return training_set