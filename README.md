# Neural Network Simple Module

Simple python package to better understand how neural networks and backpropagation work


## Network Structure

Una red neuronal es una representación matemática del funcionamiento básico del cerebro. Éste está compuesto de neuronas que a su vez reciben y entregan señales entre ellas. Al recibir una señal suficientemente grande, la neurona se activa y entrega un output a la siguiente neurona, por otro lado, si la señal no es suficientemente fuerte la neurona no se activa o su señal de salida será muy baja. 

## Backpropagation

The end goal of backpropagation is to adjust weights so that the cost function gets minimized. So every iterarion adjust a little the weights and bias in the direction of the steepest descent of the cost function.
Wm(k+1) = Wm(k) - alpha * delta_Wm
bm(k+1) = bm(k) - alpha * delta_bm

In term of the sensitivities
Wm(k+1) = Wm(k) - alpha * Sm * (a_m1)^T
bm(k+1) = bm(k) - bm(k) * alpha * Sm
Where:
Sm = dF/dn = [ dF/dnm_1, dF/dnm_2, ..., dF/dnm_sm ]

Computando la variación de los inputs de la capa m+1 con respecto de la capa m 
La matriz jacobiana quedaría: (asumiendo una red 2:3:2)

dn2/dn1 = [[ dn2_1/dn1_1 , dn2_1/dn1_2, dn2_1/dn1_3],
            [ dn2_2/dn1_1 , dn2_2/dn1_2, dn2_2/dn1_3]]

analizando dn2_1/dn1_1...

n2_1 (inputs de la neurona 1 de la segunda capa) = ( w2_11*a1_1 + w2_21*a1_2 + w2_31*a1_3 ) + b2_1
w2_11 : peso segunda capa de neurona 1 (capa1) a neurona 1 (capa2)
a1_1  : activación (output) de la neurona 1 capa 1 ( a = f(n) )

Por lo tanto la parcial con respecto del input de la neurona 1 de la capa anterior es:

dn2_1/dn1_1 = w2_11 * d/dn1_1(a1_1) = w2_11 * f1'(n1_1)

Y por lo tanto la matriz jacobiana puede traducirse en forma matricial en:

dn2/dn1 = W2 * F1'(n1) 

donde:

F1'(n1) =  [[f1'(n1)     0        ],
            [0           f1'(n2)],]

Por lo tanto la sensibilidad:
Sm = Fm'(nm) * (Wm+1)^T * Sm+1 


Para obtener la última sensibilidad (de donde empezar el backpropagation...)
Sm = -2*Fm'(nm) * (t-a) 

Ajstando pesos con la sensibilidad
W(k+1) = W(k) + alpha * S^m (a^m-1)^T
b(k+1) = b(k) + alpha * S^m 

Batch training
Iteramos por todos los puntos del set de entrenamiento e incrementamos los pesos conforme al promedio del error

## How to run?

Network configurations can be adjusted in the run.py file, params that can be configured include:
- Number of inputs
- Number of output neurons
- Hidden layers
- Number of neurons per layer
- Activation function in the hidden layers
- Activation function in the output layer

Instantiate the NeuralNet class with the parameters you chose:
```python
net = NeuralNet(layers=layers,W=weights,b=bias,f=functions,strategy=training_strategy)
```
Then execute the method to start training the function and test the resulting net
```python
net.train()
net.test(p=np.matrix([1,1]).transpose())
```