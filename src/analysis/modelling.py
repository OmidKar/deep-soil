# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:32:56 2019

@author: Omid
"""

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def createRegressors():
    random_state = 0
    regressors = []
    
    #regressors.append(LinearRegression())
    #regressors.append(Lasso(random_state=random_state))
    #regressors.append(Ridge(random_state=random_state))
    #regressors.append(ElasticNet(random_state=random_state))
    #regressors.append(BayesianRidge())
    #regressors.append(ARDRegression())
    #regressors.append(MLPRegressor(random_state=random_state))
    #regressors.append(SVR(kernel='linear'))
    #regressors.append(SVR(kernel='poly', degree=2))
    #regressors.append(SVR(kernel='poly'))
    #regressors.append(SVR(kernel='rbf'))
    #regressors.append(AdaBoostRegressor(random_state=random_state))
    #regressors.append(RandomForestRegressor(random_state=random_state))
    #regressors.append(BaggingRegressor(random_state=random_state))
    regressors.append(XGBRegressor(random_state=random_state))
    regNames = ["XGB"] #["MLP", "SVR", "RF", "XGB"] #["LR", "L2", "BR", "ARD", "MLP", "SVR-LR", "SVR-Rbf", "RF", "XGB"] #, "Bag"
    scorers = ["r2"]
    return [regressors, regNames, scorers]

def XGBPrediction(X, y, X_test, n_estimators, learning_rate, max_depth,
                 min_child_weight, gamma, subsample,
                 colsample_bytree, reg_alpha, reg_lambda):
    
    # Fitting tuned XGBoost and get the results
    regressor = XGBRegressor(n_estimators= n_estimators,
                             learning_rate= learning_rate,
                             max_depth= max_depth,
                             min_child_weight= min_child_weight,
                             gamma= gamma,
                             subsample= subsample,
                             colsample_bytree= colsample_bytree,
                             reg_alpha= reg_alpha,
                             reg_lambda= reg_lambda)
    
    regressor.fit(X, y)
    return regressor.predict(X_test)
    
def plotCVResults(data, stdData, title, figNum):
    plt.figure(figNum)
    g = sns.barplot("CrossValMeans", "Algorithm", data = data, palette="Set3", 
                    orient = "h", **{'xerr':stdData})
    g.set_title(title)
    
def obtainCVRegressionResults(X, y, scorer, title, regNames, regressors, 
                              figNum = 0, withPlot = True, printFlag = True):
    cvResults = []
    names = []
    for index, regressor in enumerate(regressors):
        try:
            cvResults.append(cross_val_score(regressor, X, y = y, scoring = scorer, 
                                             cv = 5, n_jobs = 1))
            names.append(regNames[index])
        except:
            print("Regressor ", regNames[index], "generated an error!")

    cvMeans = []
    cvStd = []
    for cvResult in cvResults:
        cvMeans.append(cvResult.mean())
        cvStd.append(cvResult.std())

    cvRes = pd.DataFrame({"CrossValMeans":cvMeans, "CrossValerrors":cvStd, "Algorithm":names})
    
    if printFlag:
        print(title)
        print(cvRes,"\n")
    
    if withPlot:
        plotCVResults(cvRes, cvStd, title, figNum)
    
    return cvRes

def obtainTestRegressionResults(X, y, X_test, y_test, scorer, title, regNames, regressors, sc, logFlag):
    cvResults = []
    names = []
    for index, regressor in enumerate(regressors):
        try:
            regressor.fit(X, y)
            y_res = regressor.predict(X_test)
            y_res = sc.inverse_transform(y_res)
            if logFlag:
                pred = np.exp(y_res)
                actual = np.exp(y_test)
                cvResults.append(r2_score(actual, pred))
            names.append(regNames[index])    
        except:
            print("Regressor ", regNames[index], "generated an error!")

    cvMeans = []
    cvStd = []
    for cvResult in cvResults:
        cvMeans.append(cvResult.mean())
        cvStd.append(cvResult.std())

    cvRes = pd.DataFrame({"CrossValMeans":cvMeans, "CrossValerrors":cvStd, "Algorithm":names})
    return cvRes  
    
def runAllModelsWithAllMetrics(X, y, X_test, y_test, sc, logFlag, targetVar, scorers, regNames, regressors, CVFlag = True,
                               figNumRoot = 0, withKernelPca = False, withPlot = True, printFlag = True):
    results = {}
    if CVFlag:
        for index, metric in enumerate(scorers):
            title = "Modelling '"+str(targetVar)+"' - Metric: "+str(metric)
            results[metric] = obtainCVRegressionResults(X, y, metric, title, regNames, regressors, figNum=figNumRoot+index, 
                                                        withPlot = withPlot, printFlag = printFlag)
    else:
        for index, metric in enumerate(scorers):
            title = "Modelling '"+str(targetVar)+"' - Metric: "+str(metric)
            results[metric] = obtainTestRegressionResults(X, y, X_test, y_test, metric, title, regNames, regressors, sc, logFlag)
            
    return results
    
def runAllModelsForAllTargetVariables(X_trainDict, y_trainDict, X_testDict, y_testDict, scorers, regNames, regressors, 
                                      scDict, logFlagDict, CVFlag = True, withKernelPca = False, withPlot = True, printFlag = True):
    results = {}
    for index, targetVar in enumerate(X_trainDict.keys()):       
        results[targetVar] = runAllModelsWithAllMetrics(X_trainDict[targetVar], y_trainDict[targetVar],
                                                        X_testDict[targetVar], y_testDict[targetVar],
                                                        targetVar, scorers, regNames, regressors, 
                                                        scDict[targetVar], logFlagDict[targetVar], 
                                                        CVFlag = CVFlag, figNumRoot = index, withKernelPca = withKernelPca, 
                                                        withPlot = withPlot, printFlag = printFlag)
    return results

def transformWithKernelPca(X, X_test, comp, kernel, deg, gamma, coef):
    transformer = KernelPCA(n_components=comp, kernel=kernel, degree = deg, 
                            gamma = gamma, coef0 = coef, random_state = 0)
    return (transformer.fit_transform(X), transformer.transform(X_test))
    
    
    
    
    