# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split # A quick way to obtain random subsets of songs

def get_data(ntrain=10000):
    #  Dataset from http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD#
    data=np.genfromtxt('YearPredictionMSD.txt',delimiter=',')
    X_train=data[0:463715,1:]
    y_train=data[0:463715,0]
    X_test=data[-51630:,1:]
    y_test=data[-51630:,0]

    # Use a subset of the training data (you are free to use a larger set if you want)
    assert(ntrain <= 463715)
    X_unused, X_train, y_unused, y_train = train_test_split(X_train, y_train, test_size=ntrain, random_state=42)

    # Split test data into a smaller test set and a validation set (arbitrarily)
    ntest=31630
    nval=20000
    assert(ntest + nval <= len(y_test))
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=ntest, random_state=42)
    X_val=X_val[0:nval,:]
    y_val=y_val[0:nval]
    
    return X_train, y_train, X_val, y_val, X_test, y_test