#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm

def nbtrain( x, t ):
    # Είσοδος x : Pxn πίνακας με τα πρότυπα (P=πλήθος προτύπων, n=διάσταση)
    # Είσοδος t : διάνυσμα με τους στόχους (0/1)
    # Έξοδος model : dictionary που θα περιέχει τις παραμέτρους του μοντέλου
    clasiMiden=x[t==0,:]
    clasiEna=x[t==1,:]
    plithosAll=(x.shape[0])

    pithanotitaMiden=clasiMiden.shape[0]/float(plithosAll)
    pithanotitaEna=clasiEna.shape[0]/float(plithosAll)
    m0=np.zeros(clasiMiden.shape[0]).astype(float)
    m1=np.zeros(clasiEna.shape[0]).astype(float)
    
    
    m0=np.mean(clasiMiden, axis=0)
    s0=np.std(clasiMiden,axis=0)
    
    m1=np.mean(clasiEna,axis=0)
    s1=np.std(clasiEna,axis=0)
    
    model={
            'prior':[pithanotitaMiden, pithanotitaEna],
            'mu0':m0,
            'mu1':m1,
            'sigma0': s0,
            'sigma1': s1
            }
    
    return model

def nbpredict( x, model ):
    # Είσοδος x : Pxn πίνακας με τα πρότυπα
    # Είσοδος model : dictionary με τις παραμέτρους του μοντέλου NB
    # Έξοδος predict : διάνυσμα με τις εκτιμώμενες τιμές στόχου
    prior=model['prior']
    predict=np.zeros(x.shape[0]).astype(float)
    for p in range(x.shape[0]):
        l=prior[1]/prior[0]
        for i in range(x.shape[1]):
            l=l*( norm.pdf(x[p,i], model['mu1'][i], model['sigma1'][i])/norm.pdf(x[p,i], model['mu0'][i], model['sigma0'][i]))
        if l<1:
            predict[p]=0
        else:
            predict[p]=1
    return predict

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
    ttrain=np.array(ttrain).astype(float)
    ttest=np.array(ttest).astype(float)
    
    model=nbtrain(xtrain,ttrain)
    predictTest=nbpredict( xtest, model )
 
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