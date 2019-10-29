# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:05:53 2019

@author: Omid
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#log transformation to overcome multicolinearity
def logTransformVariable(df, col, tol = 0.5):
    currentSkew = df[col].skew()
    print ("Skew is:", currentSkew)

    target = np.log(df.loc[df[col]>0, col])
    potentialSkew = target.skew()
    print ("Skew is:", potentialSkew)

    if (abs(currentSkew) - abs(potentialSkew))/abs(currentSkew) > tol:
        df.loc[df[col]>0, col] = target
        print("result flag is True")
        return True
    print("result flag is False")
    return False

def transformNumericVars(data, cols, tol = 0.5, removeOtherCols = False):
    logFlagDict = {}
    if removeOtherCols:
        df = data[cols].copy(deep=True)
    else:
        df = data.copy(deep=True)
    for col in cols:
        logFlagDict[col] = logTransformVariable(df, col, tol = tol)
    return [logFlagDict, df]

def prepareTrainAndTest(data, predictors):
    XDict = {}
    yDict = {}
    for key in predictors.keys():
        XDict[key] = data[predictors[key]]
        yDict[key] = data[key]
    
    return (XDict, yDict)

def standardScaleData(X_train, X_test, y_train):
    sc_y = {}
    for key in X_train.keys():
        sc_X = StandardScaler()
        X_train[key] = sc_X.fit_transform(X_train[key].values)
        X_test[key] = sc_X.transform(X_test[key].values)
    
        sc_y[key] = StandardScaler()
        y_train[key] = sc_y[key].fit_transform(y_train[key].values.reshape(-1, 1))
    
    return sc_y

def reverseTransformPredictions(preds, sc, logFlag):
    res = sc.inverse_transform(preds)
    if logFlag:
        res[res>0] = np.exp(res[res>0])
        return res
    return res