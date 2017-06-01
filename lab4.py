#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
def evaluate(t, predict, criterion):
    tp=float(sum((predict==1) & (t==1)))
    tn=float(sum((predict==0) & (t==0)))
    fp=float(sum((predict==1) & (t==0)))
    fn=float(sum((predict==0) & (t==1)))
    try:
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
hiddenlayersizes = input("hidden_layer_sizes? ")
activationn = "logistic"
solvern = raw_input("solver? ‘sgd’, ‘adam’")
nlearning_rate_init= 0.1
nlearning_rate = "constant"
for i in range(9):
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
    ttrain=np.array(ttrain).astype(float)
    ttest=np.array(ttest).astype(float)
    if(solvern=="sgd"):
        diktio=MLPClassifier(hidden_layer_sizes=hiddenlayersizes, \
                             activation=activationn, \
                             solver=solvern, \
                             learning_rate=nlearning_rate, \
                             learning_rate_init=nlearning_rate_init, \
                             max_iter=epochs, \
                             momentum=0.9) 
    else:
        diktio=MLPClassifier(hidden_layer_sizes=hiddenlayersizes, activation=activationn, solver=solvern, learning_rate=nlearning_rate, learning_rate_init=nlearning_rate_init, max_iter=epochs) 
    
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
    plt.plot(predictTest,"ro")
    plt.subplot(3, 3, i+1)
    plt.plot(ttest,"b.")
    plt.show(block=False)
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