import numpy as np
from neural_net import NeuralNet
from activations import sigmoid, purelin


# Parameters you can mess around with...

INPUTS                          =2              # Number of input neurons
HIDDEN_LAYERS                   =[3,3,2]        # List of neurons for every hidden layer
OUTPUTS                         =1              # Number of output neurons
LEARNING_RATE                   =0.05            # Learning rate
ACTIVATION_HIDDEN               ="sigmoid"      # Activation function for neurons in the hidden layer               
ACTIVATION_OUTPUT               ="purelin"      # Activation function for neurons in the output layer
TRAINING_STRATEGY               ="batch"        # Type of training (incremental | batch for backpropagation
TRAINING_ITERATIONS             =50             # Number of max iterations through the training set
ERROR_THRESHOLD                 =0.08           # Error threshold to stop the training and not overfitting            


# Weights by layer (randomly assigned)
Wi = [np.matrix( np.random.randn(HIDDEN_LAYERS[0], INPUTS) )]
Wh = [ np.matrix( np.random.randn(HIDDEN_LAYERS[i],HIDDEN_LAYERS[i-1]) )   for i in range(1,len(HIDDEN_LAYERS))  ]
Wo = [np.matrix( np.random.randn(OUTPUTS,HIDDEN_LAYERS[-1]) )]
 
# Bias by layer (randomly assigned)
bh = [ np.matrix( np.random.randn(HIDDEN_LAYERS[i],1) )   for i in range(0,len(HIDDEN_LAYERS))  ]
bo = [np.matrix( np.random.randn(OUTPUTS,1) )]


# Initial values by layer
layers = len(HIDDEN_LAYERS) + 1
W = Wi+Wh+Wo                                            # Lista de matrices de pesos
b = bh+bo                                               # Lista de vectores de bias
f = [ eval(ACTIVATION_HIDDEN) for i in range(0,layers-1)]+[eval(ACTIVATION_OUTPUT)]   # Lista de funciones por capa


print("Backpropagation Example")
print("Initial values:")
print("W:")
print(W)
print("b:")
print(b)
print("Functions:")
print(f)


if __name__=='__main__':
    net = NeuralNet(layers=layers,W=W,b=b,f=f,strategy=TRAINING_STRATEGY,eta=LEARNING_RATE,iterations=TRAINING_ITERATIONS,acceptable_error=ERROR_THRESHOLD)
    net.train()
    net.test(p=np.matrix([1,1]).transpose())

