# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:18:54 2018

@author: naruarjun
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
data=loadmat('ex5data1.mat')
X=data['X']
Y=data['y']
Xtest=data['Xtest']
ytest=data['ytest']
Xval=data['Xval']
yval=data['yval']
p=np.random.permutation(X.shape[0])
X=X[p,:]
Y=Y[p,:]
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
        self.theta=np.zeros([1,train_x.shape[1]])
        cost=np.sum(np.power(self.theta.dot(train_x.T)-train_y.T,2)/train_x.shape[0])/2
        print(cost)
        self.alpha = 0.000000001
        for i in range(train_x.shape[1]):
            self.normalize(train_x[:,i])
        for i in range(3500):
            diff=self.theta.dot(train_x.T)-train_y.T  
            self.theta=np.subtract(self.theta,self.alpha*(diff.dot(train_x)))
            cost=np.sum(np.power(self.theta.dot(train_x.T)-train_y.T,2)/train_x.shape[0])/2
            print(cost)
        return self.theta,cost
    def error(self,train_x,train_y,theta):
        cost=np.sum(np.power(theta.dot(train_x.T)-train_y.T,2)/train_x.shape[0])/2
        return cost

reg1=multlinreg()

#Firstly plotting errors for singlevariate linear regression vs number of training examples
theta1,cost1=reg1.gradient(X[:2,:],Y[:2,:])
theta2,cost2=reg1.gradient(X[:4,:],Y[:4,:])
theta3,cost3=reg1.gradient(X[:6,:],Y[:6,:])
theta4,cost4=reg1.gradient(X[:8,:],Y[:8,:])
theta5,cost5=reg1.gradient(X[:10,:],Y[:10,:])
theta6,cost6=reg1.gradient(X[:12,:],Y[:12,:])
val1=reg1.error(Xval,yval,theta1)
val2=reg1.error(Xval,yval,theta2)
val3=reg1.error(Xval,yval,theta3)
val4=reg1.error(Xval,yval,theta4)
val5=reg1.error(Xval,yval,theta5)
val6=reg1.error(Xval,yval,theta6)
axis=np.array([2,4,6,8,10,12])
trainerror=np.array([cost1,cost2,cost3,cost4,cost5,cost6])
valerror=np.array([val1,val2,val3,val4,val5,val6])
plt.plot(axis,trainerror,color='blue')
plt.plot(axis,valerror,color='red')
plt.show()
for i in range(3):
    mat=np.power(X[:,0],i)
    mat=mat.reshape([12,1])
    X=np.hstack((X,mat))
    mat=np.power(Xval[:,0],i)
    mat=mat.reshape([21,1])
    Xval=np.hstack((Xval,mat))
#For multivariate linear regression plotting error versus number of training examples
theta1,cost1=reg1.gradient(X[:2,:],Y[:2,:])
theta2,cost2=reg1.gradient(X[:4,:],Y[:4,:])
theta3,cost3=reg1.gradient(X[:6,:],Y[:6,:])
theta4,cost4=reg1.gradient(X[:8,:],Y[:8,:])
theta5,cost5=reg1.gradient(X[:10,:],Y[:10,:])
theta6,cost6=reg1.gradient(X[:12,:],Y[:12,:])
val1=reg1.error(Xval,yval,theta1)
val2=reg1.error(Xval,yval,theta2)
val3=reg1.error(Xval,yval,theta3)
val4=reg1.error(Xval,yval,theta4)
val5=reg1.error(Xval,yval,theta5)
val6=reg1.error(Xval,yval,theta6)
axis=np.array([2,4,6,8,10,12])
trainerror=np.array([cost1,cost2,cost3,cost4,cost5,cost6])
valerror=np.array([val1,val2,val3,val4,val5,val6])
plt.plot(axis,trainerror,color='blue')
plt.plot(axis,valerror,color='red')
plt.show()