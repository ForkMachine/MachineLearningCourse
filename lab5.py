#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.svm as svm
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
        fscore=fscore= precision*recall/((precision+recall)/2)
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
t=[d[key] for key in data[:,(d1[1]-1)]]
t=np.array(t,dtype=int)
mrecall=0
mprecision=0
mfmeasure=0
msensitivity=0
maccuracy=0
mspecificity=0

readC = input("Penalty parameter C of the error term? -float ")
ngamma = input("gamma? ")

for i in range(9):
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
    ttrain=np.array(ttrain).astype(float)
    ttest=np.array(ttest).astype(float)
    
    diktio=svm.SVC(C=readC, gamma=ngamma, kernel='rbf')
    
    diktio.fit(xtrain,ttrain)
    predictTest=diktio.predict(xtest)    
    
    recall=evaluate(ttest, predictTest, 'recall')
    precision=evaluate(ttest, predictTest, 'precision')
    fmeasure=evaluate(ttest, predictTest, 'fscore')
    sensitivity=evaluate(ttest, predictTest, 'sensitivity')
    accuracy=evaluate(ttest, predictTest, 'accuracy')
    specificity=evaluate(ttest, predictTest, 'specificity')
     
    mrecall=mrecall+recall
    mprecision=mprecision+precision
    mfmeasure=mfmeasure+fmeasure
    msensitivity=msensitivity+sensitivity
    maccuracy=maccuracy+accuracy
    mspecificity=mspecificity+specificity

    plt.subplot(3, 3, i+1)
    plt.plot(ttest,"bo")
    plt.subplot(3, 3, i+1)
    plt.plot(predictTest,"r.")
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