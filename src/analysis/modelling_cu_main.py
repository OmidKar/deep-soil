# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:12:40 2019

@author: Omid
"""

import read_data
import preprocess_data
import modelling
import parameter_tuning
import warnings
warnings.filterwarnings('ignore')

dataPath = "../../data/"
datafile = "Clean_RawData.csv"
dataset = read_data.readAndCleanData(dataPath, datafile)

#[logFlagDict, dataset] = preprocess_data.transformNumericVars(dataset, 
#                                                              ['m', 'v', 'd', 'tt', 'tp', 'ike', 
#                                                               'cu', 'e', 'lambda', 'AB'], removeCols = True)

[trainDf, testDf] = read_data.obtainTrainAndTestData(dataset, dataPath, firstTime = False)
  
predictors = {'cu': ['m', 'v', 'd', 'tt', 'tp', 'ike'],
              'AB': ['m', 'v', 'd', 'tt', 'tp', 'ike'],
              'e': ['m', 'v', 'd', 'tt', 'tp', 'ike'], 
              'lambda': ['m', 'v', 'd', 'tt', 'tp', 'ike']}

(X_trainDict, y_trainDict) = preprocess_data.prepareTrainAndTest(trainDf, predictors)
(X_testDict, y_testDict) = preprocess_data.prepareTrainAndTest(testDf, predictors)
sc_yDict = preprocess_data.standardSaleData(X_trainDict, X_testDict, y_trainDict)

[regressors, regNames, scorers] = modelling.createRegressors()

targetVar = 'AB'
bestPCA = parameter_tuning.pcaTuning(X_trainDict, y_trainDict, targetVar, regressors, regNames, scorers)
print(bestPCA)

#**** stepwise tuning of XGB ******
X = modelling.transformWithKernelPca(X_trainDict[targetVar], 6, 'poly', 2, 0.16, 0.5)
y = y_trainDict[targetVar]
regressor = regressors[0]
tuningDict = {}

#Step 1:fix learning rate and determine optimum n_estimators:
parameters = {'n_estimators': [i for i in range(900, 2000, 50)],
              'learning_rate': [0.1]}
score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-1'] = {'params': params, 'score': score}
print(tuningDict)

#Step 2:Tune max_depth and min_child_weight:
parameters = {'n_estimators': [950],
             'learning_rate': [0.1],
             'max_depth': [i for i in range(1,11)],
             'min_child_weight': [1, 2, 3, 5, 6, 7, 8, 9, 10]}
score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-2'] = {'params': params, 'score': score}
print(tuningDict)

#Step 3:Tune gamma:
parameters = {'n_estimators': [950],
             'learning_rate': [0.1],
             'max_depth': [5],
             'min_child_weight': [1],
             'gamma': [i*0.1 for i in range(0, 100, 2)]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-3'] = {'params': params, 'score': score}
print(tuningDict)

#Step 4:Tune subsample and colsample_bytree:
parameters = {'n_estimators': [950],
             'learning_rate': [0.1],
             'max_depth': [5],
             'min_child_weight': [1],
             'gamma': [0.0],
             'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-4'] = {'params': params, 'score': score}
print(tuningDict)

#Step 5:Tuning Regularization Parameters:
parameters = {'n_estimators': [950],
             'learning_rate': [0.1],
             'max_depth': [5],
             'min_child_weight': [1],
             'gamma': [0.0],
             'subsample': [1.0],
             'colsample_bytree': [0.8],
             'reg_alpha':[0.0001, 0.0005, 0.001],
             'reg_lambda':[0.8, 0.9, 1.0, 1.05, 1.1]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-5'] = {'params': params, 'score': score}
print(tuningDict)

#Step 6: Tuning (decreasing) learning rate and n_estimators again:
parameters = {'n_estimators': [i for i in range(50, 2000, 50)],
             'learning_rate': [0.01*i for i in range(1, 20, 1)],
             'max_depth': [5],
             'min_child_weight': [1],
             'gamma': [0.0],
             'subsample': [1.0],
             'colsample_bytree': [0.8],
             'reg_alpha':[0.0001],
             'reg_lambda':[1.0]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-6'] = {'params': params, 'score': score}
print(tuningDict)

#Prediction
X_test = modelling.transformWithKernelPca(X_testDict[targetVar], 5, 'poly', 5, 0.2, 1.75)
testData = testDf.copy(deep=True)
results = modelling.XGBPrediction(X, y, X_test, 400, 0.04, 5, 1, 0.0, 1.0, 0.8, 0.0001, 1.0)
testData[str(targetVar)+"_pred"] = preprocess_data.reverseTransformPredictions(results, sc_yDict[targetVar], logFlagDict[targetVar])
testData.to_csv(dataPath+"output/test_tuned_xgb_with_tuned_pca_AB.csv", index=False)