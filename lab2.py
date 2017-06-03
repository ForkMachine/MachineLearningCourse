#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:47:26 2017

@author: user
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def evaluate(t, predict, criterion):
    tp=float(sum((predict==1) & (t==1)))
    tn=float(sum((predict==-1) & (t==0)))
    fp=float(sum((predict==1) & (t==0)))
    fn=float(sum((predict==-1) & (t==1)))
    try:
        precision=tp/(tp+fp)
    except ZeroDivisionError:
        precision = float('nan')
    try:
        recall=tp/(tp+fn)    
    except ZeroDivisionError:
        recall = float('nan')   
    try:
        accuracy=(tp+tn)/(tp+tn+fp+fn)
    except ZeroDivisionError:
        accuracy = float('nan')
    try:
        fscore= precision*recall/((precision+recall)/2)
    except ZeroDivisionError:
        fscore = float('nan')
    try:
        sensitivity=tp/(tp+fn)
    except ZeroDivisionError:
        sensitivity = float('nan')
    try:
        specificity=tn/(tn+fp)
    except ZeroDivisionError:
        specificity = float('nan')
    metric={
            'precision':precision,
            'recall':recall,
            'accuracy':accuracy,
            'fscore':fscore,
            'sensitivity':sensitivity,
            'specificity':specificity
            }
    try:
        return metric[criterion]
    except KeyError:
        return -1

data = pandas.read_csv('/home/user/Mixaniki_Mathisi/iris.data', header=None).values
d1=data.shape
d={"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 0}
x = np.array(data[:,0:4]).astype(float)
t=np.zeros(d1[0]).astype(float)
temp=np.ones((d1[0],1))
x=np.hstack((x,temp))
t=[d[key] for key in data[:,(d1[1]-1)]]
t=np.array(t,dtype=int)
mrecall=0
mprecision=0
mfmeasure=0
msensitivity=0
maccuracy=0
mspecificity=0

for i in range(9):
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
    #ttrain, ttest = train_test_split(t,test_size=0.1)
    ttrain=np.array(ttrain).astype(float)
    ttest=np.array(ttest).astype(float)
    ttrain1=np.zeros((ttrain.shape)[0]).astype(float)
    ttest1=np.zeros((ttest.shape)[0]).astype(float)
    ttrain1[i]=(ttrain[i]*2)-1
    ttest1[i]=(ttest[i]*2)-1

    w=np.linalg.pinv(xtrain).dot(ttrain1)
    ytest=xtest.dot(w)
    predicttest = (ytest>0)
    predicttest=predicttest.astype(int)
    recall=evaluate(ttest, predicttest, 'recall')
    precision=evaluate(ttest, predicttest, 'precision')
    fmeasure=evaluate(ttest, predicttest, 'fscore')
    sensitivity=evaluate(ttest, predicttest, 'sensitivity')
    accuracy=evaluate(ttest, predicttest, 'accuracy')
    specificity=evaluate(ttest, predicttest, 'specificity')
    mrecall=mrecall+recall
    mprecision=mprecision+precision
    mfmeasure=mfmeasure+fmeasure
    msensitivity=msensitivity+sensitivity
    maccuracy=maccuracy+accuracy
    mspecificity=mspecificity+specificity

    plt.subplot(3, 3, i+1)
    plt.plot(ttest,"b.")
    plt.subplot(3, 3, i+1)
    plt.plot(predicttest,"ro")
plt.show()
mrecall=mrecall/9
mprecision=mprecision/9
mfmeasure=mfmeasure/9
msensitivity=msensitivity/9
maccuracy=maccuracy/9
mspecificity=mspecificity/9
print 'Mean recall: ',mrecall
print 'Mean precision: ',mprecision
print 'Mean fmeasure: ',mfmeasure
print 'Mean sensitivity: ',msensitivity
print 'Mean accuracy: ',maccuracy
print 'Mean specificity: ',mspecificity
