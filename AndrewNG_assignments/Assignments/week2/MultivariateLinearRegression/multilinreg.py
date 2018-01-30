# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:25:30 2017

@author: naruarjun
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:57:16 2017

@author: naruarjun
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# creating the class


class multlinreg:
    
    
    def __init__(self):
        self.theta = None
        self.alpha = None
    
    
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
            train_x=(train_x-average)/(max-min)
    
    
    def gradient(self,train_x,train_y):
        self.theta=np.zeros([1,3])
        cost=np.sum(np.power(self.theta.dot(train_x.T)-train_y.T,2)/35)/2
        print(cost)
        self.alpha = 0.0000000001
        self.normalize(train_x[:,1])
        self.normalize(train_x[:,2])
        for i in range(1500):
            diff=self.theta.dot(train_x.T)-train_y.T  
            self.theta=np.subtract(self.theta,self.alpha*(diff.dot(train_x)))
        return self.theta


#Extract data to form training set


train=np.loadtxt('ex1data2.txt',delimiter=',')
train_y=train[:35,2]
train_x=train[:35,0:2]
train_x=np.reshape(train_x,[35,2])
train_y=np.reshape(train_y,[35,1])
ones=np.ones([35,1])
train_x=np.hstack((ones,train_x))


""" train_x is of 80,2 train_y is of 80,1
Extract test test"""


test_y=train[31:47,2]
test_x=train[31:47,0:2]
test_x=np.reshape(test_x,[16,2])
test_y=np.reshape(test_y,[16,1])
ones=np.ones([16,1])
test_x=np.hstack((ones,test_x))


"""test_x is of 16,2 test_y is of 16,1
Creating object of class"""


obj=multlinreg()


#Applying gradient descent


theta=obj.gradient(train_x=train_x,train_y=train_y) 
cost=np.sum(np.power(theta.dot(train_x.T)-train_y.T,2)/35)/2 


#Predict the values for the test set


predict_y=theta.dot(test_x.T)
