# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:01:31 2019

@author: Omid
"""

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

def simplifyNames(df, names):
    df.rename(columns = names, inplace = True)

def reventBackNames(df, names):
    invNames = {v: k for k, v in names.items()}
    df.rename(columns = invNames, inplace = True)

def readAndCleanData(dataPath, datafile):
    dataset = pd.read_csv(dataPath+datafile)
    columnNames = {'Archive No': 'id','D (m)': 'd', 'l': 'lambda', 'm (kg)': 'm', 'v (m/s)': 'v', 'E (kPa)': 'e', 'cu (kPa)': 'cu', 
                   'E/cu': 'e/cu', 'G/cu': 'g/cu', 'Total penetration/ D': 'tp/d', 'Time of travel (s) ': 'tt',
                   'Initial Kinetic Energy = mv2/2 (kgm2/s2)': 'ike', '(cupD3/4) (kgm2/s2)': 'cup', '(mv2/2)/ (cupD3/4)': 'ke/cup'}
    simplifyNames(dataset, columnNames)
    dataset['tp'] = dataset.apply(lambda x: x['tp/d']*x['d'], axis=1)
    dataset['v2'] = dataset.apply(lambda x: x['v']*x['v'], axis=1)
    dataset['AB'] = dataset.apply(lambda x: (math.pi)*(x['cu']+0.01*x['e'])*x['d']*3./4., axis=1)
    return dataset

def obtainTrainAndTestData(dataset, dataPath, firstTime = False, transFlag = False):
    if transFlag:
        trainName = "train_trans.csv"
        testName = "test_trans.csv"
    else:
        trainName = "train_data.csv"
        testName = "test_data.csv"
        
    if firstTime:
        targetCols = ['e', 'cu', 'lambda', 'AB']
        trainDf = pd.DataFrame()
        for col in targetCols:
            for val in dataset[col].unique():
                filteredData = dataset[dataset[col] == val]
                trainDf = trainDf.append(filteredData.loc[np.random.choice(filteredData.index, 11), :])
    
        trainDf.to_csv(dataPath+trainName, index=False)
        testDf = pd.concat([dataset, trainDf]).drop_duplicates(keep=False)
        testDf.to_csv(dataPath+testName, index=False)
    else:
        trainDf = pd.read_csv(dataPath+trainName)
        testDf = pd.read_csv(dataPath+testName)
    return [trainDf, testDf]