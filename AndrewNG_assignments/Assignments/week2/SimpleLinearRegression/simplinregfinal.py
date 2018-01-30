# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:57:16 2017

@author: naruarjun
"""

import numpy as np
import matplotlib.pyplot as plt
#creating the class


class simplinreg:
    
    def __init__(self):
        self.theta = None
        self.alpha = None
    
    def gradient(self,train_x,train_y):
        self.theta=np.zeros([1,2])
        self.alpha = 0.01/80
        for i in range(1500):
            diff=self.theta.dot(train_x.T)-train_y.T  
            self.theta=np.subtract(self.theta,self.alpha*(diff.dot(train_x)))
        return self.theta
#Extract data to form training set


train=np.loadtxt('ex1data1.txt',delimiter=',')
train_y=train[:80,1]
train_x=train[:80,0]
train_x=np.reshape(train_x,[80,1])
train_y=np.reshape(train_y,[80,1])
ones=np.ones([80,1])
train_x=np.hstack((ones,train_x))


"""train_x is of 80,2 train_y is of 80,1
Extract test test"""


test_y=train[81:96,1]
test_x=train[81:96,0]
test_x=np.reshape(test_x,[15,1])
test_y=np.reshape(test_y,[15,1])
ones=np.ones([15,1])
test_x=np.hstack((ones,test_x))


"""test_x is of 15,2 test_y is of 15,1
Creating object of class"""


obj=simplinreg()


#Applying gradient descent


theta=obj.gradient(train_x=train_x,train_y=train_y)  


#to plot the data for test set


predict_y=theta.dot(test_x.T)
plt.scatter(test_x[:,1], test_y, color = 'red')
plt.plot(test_x[:,1], predict_y.T, color = 'blue')
plt.title('x vs y (Test set)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()