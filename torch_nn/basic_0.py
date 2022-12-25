"""
The torch nn module
the module is used for the building the models 
creating a linear regression model 
y = ax + b
"""

import torch as th
from torch import nn
import matplotlib.pyplot as plt


#params
weight = 0.7
bias = 0.3


#create the linear regression
start = 0
end = 1

step = 0.02

x = th.arange(start , end , step).unsqueeze(dim = 0)

Y = weight*x + bias

#create the test and the train dataset

train_split = int(0.8* x.shape[1])

X_train, y_train = x[:train_split],Y[:train_split]

X_test, y_test = x[train_split:],Y[train_split:]


def plot_predictions():
    """
    Plot the dataset in the 
    
    """
    pass