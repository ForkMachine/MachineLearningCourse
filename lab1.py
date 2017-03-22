# -*- coding: utf-8 -*-
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pandas.read_csv('/home/user/Mixaniki_Mathisi/iris.data', header=None).values
d1=data.shape
d={"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 0}
x = np.array(data[:,0:4]).astype(float)
t=np.zeros(d1[0])
temp=d1[0]
"""for i in range (temp-1):
    t[i]=d[data[i,4]]"""
t=[d[key] for key in data[:,(d1[1]-1)]]
for i in range(9):
    xtrain, xtest = train_test_split(x,test_size=0.1)
    ttrain, ttest = train_test_split(t,test_size=0.1)
    plt.subplot(3, 3, i+1)
    plt.plot(xtrain[:,0], xtrain[:,2],"b.")
    plt.subplot(3, 3, i+1)
    plt.plot(xtest[:,0], xtest[:,2],"r.")
plt.show()