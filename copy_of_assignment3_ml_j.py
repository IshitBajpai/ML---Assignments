# -*- coding: utf-8 -*-
"""Copy of Assignment3_Ml_j.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b8F87fLlUnVbBe9YGcJE8zaZnNUSruJY
"""

from keras.datasets import mnist
import numpy as np 
from matplotlib import pyplot as plt
from numpy.random import shuffle
import pickle
import os
import warnings

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
train_X = train_X.reshape((train_X.shape[0],28*28))
test_X = test_X.reshape((test_X.shape[0],28*28))

def Sigmoid(x):
    s= 1/(1+np.exp(-x))
    d=s*(1-s)
    return s

def Sigmoid_grad(t):
    return t*(1-t)

def Relu(x):
	return np.maximum(0.0, x)
 
def Relu_grad(t):
  return 1*(t>0)

def Leaky_relu(x):
  return np.maximum(0.1*x, x)

def Leaky_relu_grad(x):
  return 1**(x>0)*0.1**(1 - (x > 0))

def Tanh(x):
  t=np.tanh(x)
  return t

def Tanh_grad(x):
  return 1-x**2

def Linear(x):
  return x

def Linear_grad(x):
  return 1

def Softmax(x):
  ex = np.exp(x - np.max(x,axis=1,keepdims=True))
  return ex / np.sum(ex,axis=1,keepdims=True)
  
def Softmax_grad(x):
  return 0

def scores(model, trainX, trainY, testX, testY):
    print("Training score", model.score(trainX, trainY))
    print("Testing score", model.score(testX, testY))

activation = [[Sigmoid, Sigmoid_grad],[Relu,Relu_grad],[Leaky_relu,Leaky_relu_grad],[Tanh,Tanh_grad],[Linear,Linear_grad]]
weightInit = ['zero_init', 'random_init', 'normal_init']
A = [256, 128, 64, 32]

N = 4
A = [256, 128, 64, 32]
epochs = 100
batch = 128
lr = 0.001
def normalize(data, data1):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = np.divide(data-mean,std, out=np.zeros_like(data-mean), where=std!=0)
    data1 = np.divide(data1-mean,std, out=np.zeros_like(data1-mean), where=std!=0)
    return data, data1

n_train_X, n_test_X = normalize(train_X, test_X)

file="/content/sigmoid.sav"

model  = NeuralNetwork(N, A, lr, activation[0], weightInit[2], epochs, batch)
model.fit(n_train_X, train_Y, n_test_X, test_Y)
pickle.dump(model,open(file,'wb'))

scores(model,n_train_X,train_Y,n_test_X,test_Y)

model.plot()

file="/content/Relu.sav"

epochs = 100
batch = 128
lr = 0.001

model  = NeuralNetwork(N, A, lr, activation[1], weightInit[2], epochs, batch)
model.fit(train_X, train_Y,test_X,test_Y)
pickle.dump(model,open(file,'wb'))
  
scores(model,train_X,train_Y,test_X,test_Y)

model.plot()

file="/content/Leaky_ReLU.sav"
N = 4
epochs = 100
batch = 128
lr = 0.001

model  = NeuralNetwork(N, A, lr, activation[2], weightInit[2], epochs, batch)
model.fit(train_X, train_Y,test_X,test_Y)
pickle.dump(model,open(file,'wb'))
  
scores(model,train_X,train_Y,test_X,test_Y)

model.plot()

file="/content/tanh.sav"
N = 4
epochs = 100
batch = 128
lr = 0.1

model  = NeuralNetwork(N, A, lr, activation[3], weightInit[2], epochs, batch)
model.fit(train_X, train_Y,test_X,test_Y)
pickle.dump(model,open(file,'wb'))
  
scores(model,train_X,train_Y,test_X,test_Y)

model.plot()

file="/content/linear.sav"
N = 4
epochs = 100
batch = 128
lr = 0.001

model  = NeuralNetwork(N, A, lr, activation[4], weightInit[2], epochs, batch)
model.fit(train_X, train_Y,test_X,test_Y)
pickle.dump(model,open(file,'wb'))
  
scores(model,train_X,train_Y,test_X,test_Y)

model.plot()

"""Question3"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import Process
import gzip

fashion_mnist=keras.datasets.fashion_mnist
(train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()

train_X = train_X.reshape((train_X.shape[0],28*28))
test_X = test_X.reshape((test_X.shape[0],28*28))

train_X,val_X,train_Y,val_Y=train_test_split(train_X,train_Y,random_state=1,test_size=0.15)

sc=StandardScaler()
sc.fit(train_X)

def plot(tloss,vloss, act_func):
  plt.plot(tloss, label='training loss')
  plt.plot(vloss, label='validation loss') 
  plt.title(f"loss vs epochs for {act_func}")
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.legend()
  plt.show()

def MLP(train_x,train_y,test_x,test_y,layers,act_func,batch_size,epochs,lr=1000):

  if(lr==1000):
    model = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=0.001, batch_size=batch_size, activation=act_func)
  else:
    model = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=lr, batch_size=batch_size, activation=act_func)

  vloss=[]
  tloss=[]
  for i in range(epochs):
    model.partial_fit(train_x,train_y,classes=list(set(train_y)))
    loss=log_loss(test_y,model.predict_proba(test_x))
    vloss.append(loss)
    tloss.append(model.loss_)
  if(lr==1000):
    plot(tloss,vloss,act_func)
  else:
    plot(tloss,vloss,lr)
  return min(vloss)

# For finding the best activation function
activations=["logistic","relu","tanh","identity"]

layers=(256,32)
epochs=100
batch_size=512
min_loss = 100
best_activation = None
for activation in activations:
    loss = MLP(train_X, train_Y, val_X, val_Y,layers, activation, batch_size, epochs)
   
    print("Obtained a loss of:", loss, " with activation function: ", activation)
    if loss < min_loss:
        min_loss = loss
        best_activation = activation
      
print("Minimum loss is: ",min_loss," with activation function: ",best_activation)

learning_rates=[0.1, 0.01, 0.001]
min_loss = 100
best_lr = None

for lr in learning_rates:
    loss=MLP(train_X, train_Y, val_X, val_Y, layers, best_activation, batch_size, epochs,  lr)
    print("Learning Rate:", lr, ", loss:",loss)
    if loss < min_loss:
        min_loss = loss
        best_lr = lr

print("Minimum loss is: ",min_loss," with learning rate: ",best_lr," corresponding to activation: ",best_activation)

best_activation="logistic"
best_lr=0.001
layers = [(256, 32), (256, 16),(128, 32), (128, 16),(64, 32), (64, 16), (32, 32), (32, 16), (16, 16)]
batch__size = 512

min_loss=100
best_layer=None
for layer in layers:
    model = MLPClassifier(hidden_layer_sizes=layer, learning_rate_init=best_lr, batch_size=batch__size, activation=best_activation,verbose=True)
    model.fit(train_X, train_Y)
    plot(model.loss_curve_, [], layer)
    print("Layer: ", layer, " loss: ", model.best_loss_)

    if(model.best_loss_<min_loss):
      min_loss=model.best_loss_
      best_layer=layer
    
print("Layer: ",best_layer," Gives the minimum loss: ",min_loss)

# Decrease the number of neurons and plot the train loss.
layers = [(256, 32), (256, 16),(128, 32), (128, 16), (64, 32), (64, 16), (32, 32), (32, 16),(16,16)]

model = MLPClassifier(hidden_layer_sizes=layers, max_iter=10, batch_size=512,shuffle=True)
params = {
    'learning_rate_init': [0.1, 0.01, 0.001],
    'activation': ["logistic", "relu", "tanh", "identity"],
    'hidden_layer_sizes': layers, 
}

gsearch = GridSearchCV(model, param_grid=params, verbose=10, n_jobs=-1, cv = 5)
gsearch.fit(train_X, train_Y)
print(gsearch.best_params_)


