# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:10:49 2017

@author: naruarjun
"""


import numpy as np
import math


class logreg:
    
    def __init__(self):
        self.theta=None
        self.alpha=None
    
    def normalize(self,train_x):
        max=train_x[0]
        min=train_x[0]
        average=0
        for i in range(len(train_x)):
            if(max<train_x[i]):
                max=train_x[i]
            if(min>train_x[i]):
                min=train_x[i]
            average = average+train_x[i]
        average = average/len(train_x)
        for i in range(len(train_x)):
            train_x[i]=(train_x[i]-average)/(max-min)
    
    def sigmoid(self,train_x):
        sig=np.zeros(train_x.T.shape)
        for i in range(len(train_x.T)):
            sig[i]=1/(1+np.exp(-train_x.T[i]))
        return sig.T
    
    def gradient(self,train_x,train_y):
        self.theta=np.zeros([1,3])
        self.alpha=0.0001/80
        for i in range(1500):
            self.normalize(train_x[:,1])
            self.normalize(train_x[:,2])
            g=self.theta.dot(train_x.T)
            sig=self.sigmoid(g)
            diff=sig-train_y.T
            self.theta = np.subtract(self.theta,self.alpha*(diff.dot(train_x)))
        return self.theta


#Extract data to form training set


train=np.loadtxt('ex2data1.txt',delimiter=',')
train_y=train[:80,2]
train_x=train[:80,0:2]
train_x=np.reshape(train_x,[80,2])
train_y=np.reshape(train_y,[80,1])
ones=np.ones([80,1])
train_x=np.hstack((ones,train_x))


""" train_x is of 80,2 train_y is of 80,1
Extract test test"""


test_y=train[80:,2]
test_x=train[80:,0:2]
test_x=np.reshape(test_x,[20,2])
test_y=np.reshape(test_y,[20,1])
ones=np.ones([20,1])
test_x=np.hstack((ones,test_x))


"""test_x is of 20,2 test_y is of 20,1
Creating object of class"""


obj=logreg()


#Applying gradient descent


theta=obj.gradient(train_x=train_x,train_y=train_y) 
z=theta.dot(test_x.T)
predict_y=obj.sigmoid(z)
predict_y=predict_y.T
for i in range(len(predict_y)):
    if(predict_y[i]>=0.8):
        predict_y[i]=1
    else:
        predict_y[i]=0
        
       
