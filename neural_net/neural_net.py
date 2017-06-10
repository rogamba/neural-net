import numpy as np
import sys
import random
from activations import sigmoid, purelin
from training_set import gen_training_set, regfunc

''' Ejemplo de la creación de una red neuronal entrenada con 
    retropropagación dada una función de pruebas
    Enter an initial net configuration
'''

class NeuralNet:

    layers=None                 # Number of layers
    trainig_set=[]              # Training set
    training_batch = []         # Batch of net params to adjust weights and bias
    W=[]                        # List of weights
    b=[]
    f=[]
    a=[]
    n=[]
    s=[]
    p=[]
    t=[]
    mse=100

    def __init__(self,layers=2,W=W,b=b,f=f,strategy="incremental",eta=0.5,iterations=10,acceptable_error=0.5):
        self.layers = layers        # int of number of layers
        self.W = W                  # list of weight matrices by layer
        self.b = b                  # list of bias vectors by layer
        self.f = f                  # list of functions by layer
        self.F = f                  # list of matrices of function derivatives valuated
        self.eta = eta              # Learning Rate
        self.strategy=strategy      # Training strategy (bathch or incremental)
        self.acceptable_error=acceptable_error
        self.iterations=iterations
        self.training_set = gen_training_set()
        self.training_batch = [None]*len(self.training_set)
        print(self.training_set)
        pass    


    def feedforward(self):
        self.a = []
        self.n = []
        # Iterate layers
        for i in range(0,self.layers):
            # a^m-1 = p if i = 0
            act = self.pk if len(self.a) <= 0 else self.a[i-1]
            # Get input vector of layer
            self.n.append( np.dot(self.W[i],act) + self.b[i] )
            # Apply activation to input 
            self.a.append( self.f[i](self.n[-1]) )


    def differential_matrices(self,):
        self.F = []
        for m in range(0,self.layers):
            # get vector input in m layer
            n_prime = self.f[m](self.n[m],deriv=True)   
            # Construimos matriz diagonal del vector
            Fm = np.diag(n_prime.A1)
            # hacemos el append a la lista
            self.F.append(Fm)



    def sensitivities(self):
        # Generate F'(n)
        self.differential_matrices()
        # Get last one
        self.s = [ None for i in range(0,self.layers) ]
        # sM = -2 * FM * (t-a)
        self.s[-1] = np.dot( -2*self.F[-1] , (self.tk - self.a[-1]) )
        # Propagate sensitivity backwards
        for m in range(self.layers-2,-1,-1):
            self.s[m] = np.dot( np.dot( self.F[m] , self.W[m+1].transpose()) , self.s[m+1] )


    def save_batch(self):
        #print("Saving batch: "+str(q))
        self.training_batch.append({
            "W" : self.W,
            "s" : self.s,
            "a" : self.a,
            "b" : self.b,
            "t" : self.tk,
            "p" : self.pk
        })


    def adjust_batch_weights(self):
        # Adjust weight layer by layer 
        for m in range(self.layers-1,-1,-1):
            sum_w = 0
            sum_b = 0
            for batch in self.training_batch:
                sum_w = sum_w + np.dot( batch['s'][m] , batch['a'][m-1].transpose() )
                sum_b = sum_b + batch['s'][m]
            # Adjusting the weights and bias
            self.W[m] = self.W[m] - (self.eta/len(self.training_batch)) * sum_w
            self.b[m] = self.b[m] - (self.eta/len(self.training_batch)) * sum_b


    def mean_square_error(self):
        sum_error=0
        for batch in self.training_batch:
            error = (batch["t"]-batch["a"][-1])
            sum_error = sum_error + np.dot(error.transpose(),error)
        self.mse = (1/len(self.training_batch)) * sum_error
        print("MSE: "+ str(self.mse))



    def adjust_weights(self):
        ''' Adjust weights for incremental training
        '''
        for m in range(self.layers-1,-1,-1):
            self.W[m] = self.W[m] - self.eta * np.dot( self.s[m] , self.a[m-1].transpose() )
            self.b[m] = self.b[m] - self.eta * self.s[m]



    def train(self):
        # Hacemos el pass
        for i in range(0,self.iterations):
            # Iteramos por los puntos del training set {p:input, t:target_output}
            self.training_batch=[]
            if self.strategy == 'incremental':
                random.shuffle(self.training_set)
            for p,t in self.training_set:
                
                # inputs and outputs
                self.tk = t
                self.pk = p
       
                # Hacemos el forward pass
                self.feedforward()

                # Test error
                self.error = self.tk - self.a[-1]  

                # Hacemos el backpropagation
                self.sensitivities()
                
                # Strategy
                if self.strategy == 'batch':
                    # If incremental trainig adjust weights
                    self.save_batch()
                else:
                    self.adjust_weights()

            print("ERROR: "+str(self.error))

            if self.strategy == 'batch':
                # Mean square error
                self.mean_square_error()
                # Ajustamos los pesos y bias de todo el training set
                self.adjust_batch_weights()
                
        

    def test(self,p=None):
        print("################# Test")
        self.pk = p
        self.feedforward()
        print(p)
        target = regfunc( np.array(p)[0],np.array(p)[1] )
        print("Expected value: ", target)
        print("Output value: ",self.a[-1])


