# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:10:49 2017

@author: naruarjun
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import expit

theta=np.zeros([1,3])
lambd=100
#creating the class


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
    
    def gradient(self,train_xcopy,train_y):
        self.theta=np.zeros([1,26])
        self.alpha=0.1/118
        train_x=train_xcopy
        rows,cols=train_x.shape
        ones=np.ones([rows,1])
        train_x=np.hstack((ones,train_x))
        #for l in range(6):
        #        self.normalize(train_x[:,l])
        for i in range(5000):
            cost=0
            g=self.theta.dot(train_x.T)
            sig=self.sigmoid(g)
            for j in range(len(train_y)):
                cost=cost-train_y[j]*np.log(1/(1+np.exp(-self.theta.dot((train_x[j,:]).T))))-(1-train_y[j])*np.log(1-1/(1+np.exp(-self.theta.dot((train_x[j,:]).T))))#-lambd*self.theta[0,0]**2+(lambd)*np.sum(np.power(self.theta,2))
            cost=cost/118
            print(cost)
            diff=sig-train_y.T
            self.theta = np.subtract(self.theta,self.alpha*(diff.dot(train_x)))
            sub=self.theta[0,0]
            self.theta=self.theta-(lambd/118)*self.theta
            self.theta[0,0]=sub
        return self.theta
    def predict(self,train_xcopy,theta):
        train_x=train_xcopy
        rows,cols=train_x.shape
        ones=np.ones([rows,1])
        print(train_x.shape)
        train_x=np.hstack((ones,train_x))
        z=theta.dot(train_x.T)
        predict_y=obj.sigmoid(z)
        predict_y=predict_y.T
        """for i in range(len(predict_y)):
            if(predict_y[i]<0.501718):
                predict_y[i]=1
            else:
                predict_y[i]=0"""
        return predict_y


#Extract data to form training set
import random
nums = [x for x in range(118)]
random.shuffle(nums)

train=np.loadtxt('ex2data2.txt',delimiter=',')
train_y=train[nums[0:100],2]
train_x=train[nums[0:100],0:2]
train_x=np.reshape(train_x,[100,2])
train_y=np.reshape(train_y,[100,1])


""" train_x is of 80,2 train_y is of 80,1
Extract test test"""


test_y=train[nums[100:],2]
test_x=train[nums[100:],0:2]
test_x=np.reshape(test_x,[18,2])
test_y=np.reshape(test_y,[18,1])




"""test_x is of 15,2 test_y is of 15,1
Creating object of class"""


obj=logreg()

for i in range(6):
    for j in range(6):
        if(i==0 and j==0):
            continue
        if(i==0 and j==1):
            continue
        if(j==0 and i==1):
            continue
        if((i+j)<=6):
            mat=np.multiply(np.power(train_x[:,0],i),np.power(train_x[:,1],j))
            mat=mat.reshape([100,1])
            train_x=np.hstack((train_x,mat))
            mat=np.multiply(np.power(test_x[:,0],i),np.power(test_x[:,1],j))
            mat=mat.reshape([18,1])
            test_x=np.hstack((test_x,mat))

#Applying gradient descent
obj.normalize(train_x[:,0])

theta=obj.gradient(train_xcopy=train_x,train_y=train_y)
predict_y=obj.predict(test_x,theta=theta)