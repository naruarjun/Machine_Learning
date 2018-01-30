# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 18:29:56 2017

@author: naruarjun
"""

#I'm treating this as a neural network with just 2 layers:The input layer and the output layer
import numpy as np
from scipy.io import loadmat
from scipy.special import expit

#creating the class
class multiclass:
    
    #initializing everything needed
    def __init__(self):
        self.theta1=None
        self.a1=None
        self.delta1=None
        self.t1p=None
        self.m=None
        self.cost=None
        
    #function for getting sigmoid of an array
    def sigmoid(self,z):
        sig=np.zeros_like(z)
        for i in range(z.shape[0]):
            sig[i]=1 / (1 + np.exp(-z[i]))
        return sig
    
    #calculating the value of hypothesis as a parameterized ksum of the input parameters followed by sigmoid
    def forwardprop(self,train_x,train_y):
        self.a1=np.zeros((train_y.shape[0],train_x.shape[1]))
        self.a1=np.reshape(self.sigmoid(self.theta1.dot(train_x)),(self.sigmoid(self.theta1.dot(train_x)).shape[0],train_x.shape[1]))
        self.cost=-(train_y*np.log(self.a1)+(1-train_y)*np.log((1-self.a1)))
        self.cost=np.sum(self.cost,axis=1)
        self.cost=np.sum(self.cost)/self.m
                        
    #calculating the partial diff for gradient descent
    def backprop(self,train_x,train_y):
        self.delta1=np.zeros_like(self.a1)
        self.t1p=np.zeros_like(self.theta1)
        self.delta1=self.a1-train_y
        self.t1p=self.t1p+self.delta1.dot(train_x.T)
        self.t1p=self.t1p/self.m
        
    #calculating hypothesis for each training eg and then updating theta matrix through the partial diff
    def gradient(self,train_x,train_y):
        self.m=len(train_x[0])
        self.theta1=np.random.randn(train_y.shape[0],train_x.shape[0])
        for i in range(2500):
            self.forwardprop(train_x,train_y)
            self.backprop(train_x,train_y)
            self.theta1=self.theta1-alpha*self.t1p
            print(i,":",self.cost)
        return self.theta1

#optimum parameters for the network
alpha=1
lambd=1

#extracting the dataset
data = loadmat('ex4data1.mat')
X=data['X']
ytemp=data['y']
X=X.T

#randomizing the dataset
p=np.random.permutation(X.shape[1])
X=X[:,p]
ytemp=data['y']
Y=np.zeros((ytemp.shape[0],10),np.uint8)
for i in range(ytemp.shape[0]):
    Y[i,ytemp[i,0]-1]=1
Y=Y.T
Y=Y[:,p]

#creating the object
obj=multiclass()

#training the network
theta=obj.gradient(X[:,0:4500],Y[:,0:4500])
ytemp=ytemp.T
ytemp=ytemp[0,p]
acc=0

#checking the accuracy
for i in range(500):
    obj.forwardprop(np.reshape(X[:,4500+i],(X.shape[0],1)),np.reshape(Y[:,4500+i],(Y.shape[0],1)))
    a=np.amax(obj.a1)
    for j in range(len(obj.a1)):
        if obj.a1[j]==a:
            if j+1==ytemp[4500+i]:
                acc=acc+1
print(acc)

#achieved a accuracy between 88-92%

#