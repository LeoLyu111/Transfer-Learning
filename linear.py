#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from scipy.stats import uniform
from datetime import datetime as dt
import math


rd.seed(0)


class DescentMethod(object):
    
    def __init__(self,size1,size2,size3,size4,dim,mu_norm,mu_1,mu_2):
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3
        self.size4 = size4
        self.dim = dim
        self.mu_norm = mu_norm
        self.mu_1 = mu_1 
        self.mu_2 = mu_2    
        
        
    def generate_data(self): # Gaussian mixture
        #training for task 1
        y_train1 = np.ones(self.size1)
        x_train1 = []
        for i in range(self.size1):
            u = uniform.rvs(0,1)
            if u<0.5:
                y_train1[i]=-1
                x_train1.append( -self.mu_1 + np.random.standard_normal(self.dim) )
            else:
                x_train1.append( self.mu_1 + np.random.standard_normal(self.dim) )
        x_train1 = np.array(x_train1)
        y_train1[y_train1<0]=0
        
        #test for task 1
        y_test1 = np.ones(self.size2)
        x_test1 = []
        for i in range(self.size2):
            u = uniform.rvs(0,1)
            if u<0.5:
                y_test1[i]=-1
                x_test1.append( -self.mu_1 + np.random.standard_normal(self.dim) )
            else:
                x_test1.append( self.mu_1 + np.random.standard_normal(self.dim) )
        x_test1 = np.array(x_test1)
        y_test1[y_test1<0]=0
        
        #training for task 2
        y_train2 = np.ones(self.size3)
        x_train2 = []
        for i in range(self.size3):
            u = uniform.rvs(0,1)
            if u<0.5:
                y_train2[i]=-1
                x_train2.append( -self.mu_2 + np.random.standard_normal(self.dim) )
            else:
                x_train2.append( self.mu_2 + np.random.standard_normal(self.dim) )
        x_train2 = np.array(x_train2)
        y_train2[y_train2<0]=0
        
        #test for task 2
        y_test2 = np.ones(self.size4)
        x_test2 = []
        for i in range(self.size4):
            u = uniform.rvs(0,1)
            if u<0.5:
                y_test2[i]=-1
                x_test2.append( -self.mu_2 + np.random.standard_normal(self.dim) )
            else:
                x_test2.append( self.mu_2 + np.random.standard_normal(self.dim) )
        x_test2 = np.array(x_test2)
        y_test2[y_test2<0]=0
        return x_train1,x_test1,y_train1,y_test1,x_train2,x_test2,y_train2,y_test2
    
    def loaddataset(self):
        x_train1,x_test1,y_train1,y_test1,x_train2,x_test2,y_train2,y_test2 = self.generate_data()
        
        x_train1 = x_train1.tolist()
        y_train1 = y_train1.tolist()
        b = np.ones(len(x_train1))
        train1 = np.column_stack((b,x_train1,y_train1)).tolist()  
        
        x_test1 = x_test1.tolist()
        y_test1 = y_test1.tolist()
        b = np.ones(len(x_test1))
        test1 = np.column_stack((b,x_test1,y_test1)).tolist()
        
        x_train2 = x_train2.tolist()
        y_train2 = y_train2.tolist()
        b = np.ones(len(x_train2))
        train2 = np.column_stack((b,x_train2,y_train2)).tolist()  
        
        x_test2 = x_test2.tolist()
        y_test2 = y_test2.tolist()
        b = np.ones(len(x_test2))
        test2 = np.column_stack((b,x_test2,y_test2)).tolist()
        return train1,test1,train2,test2
    
    def sigmoid(self,X):
        return .5 * (1 + np.tanh(.5 * X))
    
    def calAccuracyRate(self,dataMat,labelMat,weights):
        count = 0
        dataMat = np.mat(dataMat)
        labelMat = np.mat(labelMat).T
        m,n = np.shape(dataMat)

        for i in range(m):
#             h = self.sigmoid(dataMat[i,:] * weights)
#             if ( h>0.5 and int(labelMat[i,0]) == 1) or ( h<0.5 and int(labelMat[i,0]) == 0 ):
#                 count += 1 
            h2 = np.dot(dataMat[i,:], weights)
            if ( h2 > 0 and int(labelMat[i,0]) == 1) or ( h2<0 and int(labelMat[i,0]) == 0 ):
                count += 1 
        return count/m

    
    
    def SGD(self,train,test,epochs,mini_batch_size,alpha,theta):
        dataMatrix1 = np.mat(train) # training input
        dataMatrix2 = np.mat(test) # test input
        train_data = dataMatrix1[:,0:self.dim+1] # training data [x]
        train_label = dataMatrix1[:,self.dim+1] # train label [y]
        test_data = dataMatrix2[:,0:self.dim+1] # test data [x]
        test_label = dataMatrix2[:,self.dim+1] # test label [y]
        r1,c1 = np.shape(dataMatrix1[:,0:self.dim+1]) # r : number of data
                                                      # c : dimension fo data
        
        r2,c2 = np.shape(dataMatrix2[:,0:self.dim+1]) 

#         cost1 = [] ; err1 = [] # cost and error for training data 
        cost2 = [] ; err2 = [] # cost and error for test data 
        for j in range(epochs):
            idx = rd.sample(range(r1),r1)
            dataMatrix1 = np.mat(train)
            dataMatrix = dataMatrix1[idx]
            mini_batches = [
                dataMatrix[k:k+mini_batch_size] 
                for k in range(0, r1, mini_batch_size)]
            for mini_batch in mini_batches:
                batch_data = mini_batch[:,0:self.dim+1]
                batch_label = mini_batch[:,self.dim+1]
                a,b = np.shape(batch_data)
                h = batch_data * theta
                for i in range(len(h)):
                    h[i] = self.sigmoid(h[i])
                L = (h - batch_label) 
                theta = theta - (alpha * batch_data.T * L)/mini_batch_size 
#             h1 = train_data * theta
#             for i in range(len(h1)):
#                 h1[i] = self.sigmoid(h1[i])
            h2 = test_data * theta
            for i in range(len(h2)):
                h2[i] = self.sigmoid(h2[i])
    
#             c1 = (1/len(h1))*np.sum(-train_label.T*np.log(h1)-(np.ones((len(h1),1))-train_label).T*np.log(np.ones((len(h1),1))-h1))
            c2 = (1/r2)*np.sum(-test_label.T*np.log(h2+1e-5)-(np.ones((r2,1))-test_label).T*np.log(np.ones((r2,1))-h2+1e-5))
            
#             e1 = 1 - self.calAccuracyRate(train_data.tolist(),train_label.T.tolist(),theta)
            e2 = 1 - self.calAccuracyRate(test_data.tolist(),test_label.T.tolist(),theta) 
#             err1.append(e1)
            err2.append(e2)
#             cost1.append(c1)
            cost2.append(c2)
        return cost2, err2, theta
    
    def MIXSGD(self,train_1,train_2,test_1,test_2,epochs,mini_batch_size,alpha,theta,lmbda):
        dataMatrix1 = np.mat(train_1) # task1 train
        dataMatrix2 = np.mat(train_2) # task2 train
        dataMatrix3 = np.mat(test_1) # task1 test
        dataMatrix4 = np.mat(test_2) # task2 test

        task1_data = dataMatrix1[:,0:self.dim+1] # task 1 data [x]
        task1_label = dataMatrix1[:,self.dim+1] # task 1 label [y]
        task2_data = dataMatrix2[:,0:self.dim+1] # task 2 data [x]
        task2_label = dataMatrix2[:,self.dim+1] # task 2 label [y]
        task3_data = dataMatrix3[:,0:self.dim+1] # task 2 data [x]
        task3_label = dataMatrix3[:,self.dim+1] # task 2 label [y]
        task4_data = dataMatrix4[:,0:self.dim+1] # task 2 data [x]
        task4_label = dataMatrix4[:,self.dim+1] # task 2 label [y]
        
        r1,c1 = np.shape(dataMatrix1[:,0:self.dim+1]) # r: number of data; c: dimension fo data
        r2,c2 = np.shape(dataMatrix2[:,0:self.dim+1]) 
        r3,c3 = np.shape(dataMatrix3[:,0:self.dim+1]) 
        r4,c4 = np.shape(dataMatrix4[:,0:self.dim+1])

        
#         cost1 = [] ; err1 = [] # cost and error for task1 data 
#         cost2 = [] ; err2 = [] # cost and error for task2 data (train)
        cost4 = [] ; err4 = [] # cost and error for task2 data (test)
        for j in range(epochs):
            idx = rd.sample(range(r1),r1)
            dataMatrix = dataMatrix1[idx]
            mini_batches = [
                dataMatrix[k:k+mini_batch_size] 
                for k in range(0, r1, mini_batch_size)]
        
            h2 = task2_data * theta
            for i in range(len(h2)):
                h2[i] = self.sigmoid(h2[i])
            
            for mini_batch in mini_batches:
                batch_data = mini_batch[:,0:self.dim+1]
                batch_label = mini_batch[:,self.dim+1]
                a,b = np.shape(batch_data)
                h = batch_data * theta
                for i in range(len(h)):
                    h[i] = self.sigmoid(h[i])
                L1 = (h - batch_label) 
                L2 = (h2 - task2_label)
                L = (1-lmbda)*batch_data.T*L1/mini_batch_size + lmbda*task2_data.T*L2/r2
                theta = theta - (alpha * L)
            
            h4 = task4_data * theta
            for i in range(len(h4)):
                h4[i] = self.sigmoid(h4[i])
        
    
#             c1 = (1/len(h1))*np.sum(-task1_label.T*np.log(h1)-(np.ones((len(h1),1))-task1_label).T*np.log(np.ones((len(h1),1))-h1))
#             c2 = (1/r2)*np.sum(-task2_label.T*np.log(h2+1e-5)-(np.ones((r2,1))-task2_label).T*np.log(np.ones((r2,1))-h2+1e-5))
            c4 = (1/r4)*np.sum(-task4_label.T*np.log(h4+1e-5)-(np.ones((r4,1))-task4_label).T*np.log(np.ones((r4,1))-h4+1e-5))

#             e1 = 1 - self.calAccuracyRate(task1_data.tolist(),task1_label.T.tolist(),theta)
#             e2 = 1 - self.calAccuracyRate(task2_data.tolist(),task2_label.T.tolist(),theta) 
            e4 = 1 - self.calAccuracyRate(task4_data.tolist(),task4_label.T.tolist(),theta) 
            
#             err1.append(e1)
#             err2.append(e2) # error for the training data of task 2
            err4.append(e4) # error for the test dataof task 2
#             cost1.append(c1)
#             cost2.append(c2) # cost for the training data of task 2 
            cost4.append(c4) # cost for the test data of task 2
        return cost4, err4, theta

