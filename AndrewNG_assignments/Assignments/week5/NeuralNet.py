# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:41:54 2017

@author: naruarjun
"""

import numpy as np
from scipy.io import loadmat
from scipy.special import expit

#creating the class
class NeuralNet:
    
    #initializing everything needed
    def __init__(self):
        self.theta1=None
        self.theta2=None
        self.a1=None
        self.a2=None
        self.delta2=None
        self.delta1=None
        self.t1p=None
        self.t2p=None
        self.m=None
        self.hiddenlayer=None
        self.cost=None
        
    #function to compute sigmoid of an array
    def sigmoid(self,z):
        sig=np.zeros_like(z)
        for i in range(z.shape[0]):
            sig[i]=1 / (1 + np.exp(-z[i]))
        return sig
    
    #function for forward propogation
    """
    create two arrays fro storing the contents of the hidden layer and the final output of the network
    values calculated as parametrized function followed by sigmoid
    cost is also computed here only
    """
    def forwardprop(self,train_x,train_y):
        self.a1=np.zeros((self.hiddenlayer,train_x.shape[1]))
        self.a2=np.zeros((train_y.shape[0],train_y.shape[1]))
        self.a1=np.reshape(self.sigmoid(self.theta1.dot(train_x)),(self.sigmoid(self.theta1.dot(train_x)).shape[0],train_x.shape[1]))
        self.a2=np.reshape(self.sigmoid(self.theta2.dot(self.a1)),(self.sigmoid(self.theta2.dot(self.a1)).shape[0],train_x.shape[1]))
        self.cost=-(train_y*np.log(self.a2)+(1-train_y)*np.log((1-self.a2)))
        self.cost=np.sum(self.cost,axis=1)
        self.cost=np.sum(self.cost)/self.m
                        
    #backpropogation
    """
    Here we create two arrays to store the delta and two further arrays to stor the partial differenciation of the two parameter arrays
    """
    def backprop(self,train_x,train_y):
        self.delta2=np.zeros_like(train_y)
        self.delta1=np.zeros_like(self.a1)
        self.t1p=np.zeros_like(self.theta1)
        self.t2p=np.zeros_like(self.theta2)
        self.delta2=self.a2-train_y
        self.delta1=np.multiply(np.reshape(self.theta2.T.dot(self.delta2),(self.theta2.T.dot(self.delta2).shape[0],train_x.shape[1])),self.a1)
        self.delta1=np.multiply(self.delta1,1-self.a1)
        self.t2p=self.t2p+self.delta2.dot(self.a1.T)
        self.t1p=self.t1p+self.delta1.dot(train_x.T)
        self.t1p=self.t1p/self.m
        self.t2p=self.t2p/self.m
        
    #Here we perform gradient descent 
    """
    The order of operations is
    forward propogation 
    calculation of deltas and partial diffs wrt all 5000 training examples
    calculate partial differenciationsof the thetas
    update the parameters than repeat this again
    """
    def gradient(self,train_x,train_y,hiddenlayer):
        self.m=len(train_x[0])
        self.hiddenlayer=hiddenlayer
        self.theta1=np.random.randn(hiddenlayer,train_x.shape[0])
        self.theta2=np.random.randn(train_y.shape[0],hiddenlayer)
        cost=0
        for i in range(2500):
            prevcost=cost
            cost=0
            self.forwardprop(train_x,train_y)
            self.backprop(train_x,train_y)
            self.theta1=self.theta1-alpha*self.t1p
            self.theta2=self.theta2-alpha*self.t2p
            print(i,":",self.cost)
        return self.theta1,self.theta2
    
#optimum parameters for the network
alpha=1
lambd=1

#getting the dataset
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
obj=NeuralNet()

#training the network
theta,fintheta=obj.gradient(X[:,0:4500],Y[:,0:4500],25)
ytemp=ytemp.T
ytemp=ytemp[0,p]
acc=0

#now checking the accuracy
for i in range(500):
    obj.forwardprop(np.reshape(X[:,4500+i],(X.shape[0],1)),np.reshape(Y[:,4500+i],(Y.shape[0],1)))
    a=np.amax(obj.a2)
    for j in range(len(obj.a2)):
        if obj.a2[j]==a:
            if j+1==ytemp[4500+i]:
                acc=acc+1
print(acc)

#we get a accuracy ranging between 92-94%