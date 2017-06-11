#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def regrevaluate(t, predict, criterion):
    # Είσοδοι:
    # t : διάνυσμα με τους πραγματικούς στόχους (πραγματικοί αριθμοί)
    # predict : διάνυσμα με τους εκτιμώμενους στόχους (πραγματικοί αριθμοί)
    # criterion : text-string με τις εξής πιθανές τιμές:
    # 'mse'
    # 'mae'
    # Έξοδος value : η τιμή του κριτηρίου που επιλέξαμε.
    if(criterion=="mse"):
        return np.mean(pow(np.linalg.norm(t-predict),2))
    elif(criterion=="mae"):
        return(np.mean(np.abs(t-predict)))
    
data = pandas.read_csv('/home/user/Mixaniki_Mathisi/housing.data', header=None).values
d1=data.shape

x = np.zeros((d1[1]-1,d1[0]))
t=np.zeros(d1[0])
x=data[:,:13]

t=data[:,13:14]
gamma=[0.0001, 0.001, 0.01,0.1]
c=[1, 10, 100, 1000]
mse=0.0
mae=0.0
i=0
mseTemp=np.zeros(len(gamma)*2+1)
maeTemp=np.zeros(len(gamma)*2+1)
for g in gamma:
    for ci in c:
        for i in range(9):
            xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
            ttrain=np.array(ttrain).astype(float)
            ttest=np.array(ttest).astype(float)
            
            diktio=SVR(gamma=g, C=ci)
            diktio.fit(xtrain,ttrain)
            
            predictTest=diktio.predict(xtest)
            mse=mse+regrevaluate(ttest,predictTest, "mse")
            mae=mae+regrevaluate(ttest,predictTest, "mae")
        mse=mse/10
        mae=mae/10
        mseTemp[i]=mse
        maeTemp[i]=mae
        i=i+1
minMse=mseTemp[0]
minMae=maeTemp[0]

for i in range(len(maeTemp)):
    if(maeTemp[i]<minMae):
        minMae=i
        index=i
if(index<4):
    gammaTemp=0
    cTemp=index
elif(index<8):
    gammaTemp=1
    cTemp=index-4
elif(index<12):
    gammaTemp=2
    cTemp=index-8
elif(index<16):
    gammaTemp=3
    cTemp=index-12

print("MAE min for gamma: ")
print(gamma[gammaTemp])
print(" and c: ")
print(c[cTemp])

for i in range(len(mseTemp)):
    if(mseTemp[i]<minMse):
        minMse=i
        index=i
if(index<4):
    gammaTemp=0
    cTemp=index
elif(index<8):
    gammaTemp=1
    cTemp=index-4
elif(index<12):
    gammaTemp=2
    cTemp=index-8
elif(index<16):
    gammaTemp=3
    cTemp=index-12

print("MSE min for gamma: ")
print(gamma[gammaTemp])
print(" and c: ")
print(c[cTemp])

train, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
ttrain=np.array(ttrain).astype(float)
ttest=np.array(ttest).astype(float)

diktio=SVR(gamma=gamma[gammaTemp], C=c[cTemp])
diktio.fit(xtrain,ttrain)
predictTest=diktio.predict(xtest)

plt.plot(ttest)
plt.plot(predictTest,"r.")
plt.show()

nArray=[5,10,20,30,40,50]


for n in nArray:
    for i in range(9):
        xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
        ttrain=np.array(ttrain).astype(float)
        ttest=np.array(ttest).astype(float)
        
        diktio=MLPRegressor(hidden_layer_sizes=n)
        diktio.fit(xtrain,ttrain)
        predictTest=diktio.predict(xtest)
        mse=mse+regrevaluate(ttest,predictTest, "mse")
        mae=mae+regrevaluate(ttest,predictTest, "mae")
    mse=mse/10
    mae=mae/10
    mseTemp[i]=mse
    maeTemp[i]=mae
    i=i+1
minMse=mseTemp[0]
minMae=maeTemp[0]

index=0
mseTemp=np.zeros(len(nArray)+1)
maeTemp=np.zeros(len(nArray)+1)

for i in range(len(maeTemp)):
    if(maeTemp[i]<minMae):
        minMae=i
        index=i

print("MAE min for hidden layers: ")
print(nArray[index])

for i in range(len(mseTemp)):
    if(mseTemp[i]<minMse):
        minMse=i
        index=i

print("MSE min for hidden layers: ")
print(nArray[index])


train, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
ttrain=np.array(ttrain).astype(float)
ttest=np.array(ttest).astype(float)

diktio=MLPRegressor(hidden_layer_sizes=nArray[index])
diktio.fit(xtrain,ttrain)
predictTest=diktio.predict(xtest)

plt.plot(ttest)
plt.plot(predictTest,"r.")
plt.show()