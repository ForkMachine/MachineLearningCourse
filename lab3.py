#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def evaluate(t, predict, criterion):
    tp=float(sum((predict==1) & (t==1)))
    tn=float(sum((predict==-1) & (t==0)))
    fp=float(sum((predict==1) & (t==0)))
    fn=float(sum((predict==-1) & (t==1)))
    try:
        print(tp+fp)
        precision=tp/(tp+fp)
    except ZeroDivisionError:
        precision = float('nan')
    try:
        recall=tp/(tp+fn)    
    except ZeroDivisionError:
        recall = float('nan')   
    try:
        accuracy=(tp+tn)/len(t)
    except ZeroDivisionError:
        accuracy = float('nan')
    try:
        fscore=2*precision + recall/(precision+recall)
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
    
def perceptron(x, t, MAXEPOCHS, beta ):
    # Είσοδος x : πίνακας με τα επαυξημένα πρότυπα
    # Είσοδος t : διάνυσμα με τους πραγματικούς στόχους (0/1 ή -1/1)
    # Είσοδοι MAXEPOCHS : μέγιστο πλήθος εποχών
    # Είσοδοι beta : βήμα εκπαίδευσης
    # Επιστρέφει w : τελικό επαυξημένο διάνυσμα βαρών.
    noChanges=True
    x=x.transpose()
    w=np.random.randn(5)
    i=0
    for i in range(MAXEPOCHS):
        for j in range(len(x)):
            u=w.dot(x[:, j])
            if(u<0):
                y=0
            else:
                y=1
            if t[j]==y:
                continue
            else:
                w=w+beta*(t[j]-y)*x[:,j]
                noChanges=False
        if noChanges:
            break
    return w
    
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

epochs = input("How many epochs? ")
step = input("Step? ")
for i in range(9):
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
    ttrain=np.array(ttrain).astype(float)
    ttest=np.array(ttest).astype(float)

    temp=np.ones((xtrain.shape[0],1))
    xtrain1=np.hstack((xtrain,temp))
    temp=np.ones((xtest.shape[0],1))
    xtest1=np.hstack((xtest,temp))
    utest=np.ones((xtest1.shape[0],xtest1.shape[1]))
    w=perceptron(xtrain1,ttrain,epochs,step)
    utest=xtest1.dot(w)
    predictTest = 2*(utest>0)-1    
    
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
    plt.plot(ttest,"b.")
    plt.subplot(3, 3, i+1)
    plt.plot(predictTest,"ro")
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