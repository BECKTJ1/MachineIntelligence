# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:36:53 2020

@author: tomb1
"""

from sklearn.datasets import load_iris
iris = load_iris()

iris.keys()

n_samples, n_features = iris.data.shape
print (n_samples, n_features)

print(iris.data[0])

print(iris.target.shape)

print(iris.data[0])
print(iris.target[0])

print(iris.target_names)

print(iris.target)


# You don't need to include the previous line in your IDE code
import numpy as np
import matplotlib.pyplot as plt

def plot_iris_projection(x_index, y_index):
    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
                c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])

x_index=1
y_index=2
plot_iris_projection(x_index,y_index)

from sklearn import datasets

from sklearn.datasets import get_data_home
get_data_home()

from sklearn.datasets import load_digits
digits = load_digits()

digits.keys()

n_samples, n_features = digits.data.shape
print (n_samples, n_features)