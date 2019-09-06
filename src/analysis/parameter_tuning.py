# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:50:58 2019

@author: Omid
"""
import process_results
import modelling 
import time
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def pcaTuning(X_trainDict, y_trainDict, X_testDict, y_testDict, targetVar, sc_yDict, logFlagDict, regressors, regNames, scorers, CVFlag = True):
    best = ["None", 0.0] 
    bestPCA = {'comp': 0, 'kernel': 'poly', 'deg': 0, 'gamma': 0.0, 'coef': 0.0}
    start = time.time()
    for n in [None, 2, 3, 4, 5, 6]: #[5]:
        for g in [0.005, 0.01, 0.015, 0.02, 0.05, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
            for d in [2, 3, 4, 5, 6]:
                for c in [0, 0.5, 1, 1.5, 1.75, 2]:
                    (X, X_test) = modelling.transformWithKernelPca(X_trainDict[targetVar], X_testDict[targetVar], n, 'poly', d, g, c)
                    results = modelling.runAllModelsWithAllMetrics(X, y_trainDict[targetVar], 
                                                                   X_test, y_testDict[targetVar],
                                                                   sc_yDict[targetVar], logFlagDict[targetVar],
                                                                   targetVar, scorers, regNames, regressors, CVFlag = CVFlag, 
                                                                   withKernelPca = True, withPlot = False, printFlag=False)
                    (best, bestPCA) = process_results.updateBestResults(results, best, n, 'poly', d, g, c, bestPCA)
                    
    end = time.time()
    print("elapsed time for PCA parameter tuning loop: ", end-start)
    print(best)
    print(bestPCA)
    return bestPCA

def evaluateModelPerformance(est, params, X, y, scorer):
    grid_search = GridSearchCV(estimator = est,
                               param_grid = params,
                               scoring = scorer,
                               cv = 5, n_jobs = 2)
    
    grid_search = grid_search.fit(X, y)
    best_accuracy = grid_search.best_score_
    best_parameter = grid_search.best_params_
    return best_accuracy, best_parameter


    
    
    
    
    
    