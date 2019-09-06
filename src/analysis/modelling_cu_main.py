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

[logFlagDict, dataset] = preprocess_data.transformNumericVars(dataset, ['m', 'v', 'd', 'tt', 'tp', 'ike', 'cu', 
                                                              'e', 'lambda', 'AB', 'e/cu'], removeOtherCols = True)

[trainDf, testDf] = read_data.obtainTrainAndTestData(dataset, dataPath, firstTime = False, transFlag = True)
  
predictors = {'cu': ['m', 'v', 'd', 'tt', 'tp', 'ike'],
              'AB': ['m', 'v', 'd', 'tt', 'tp', 'ike'],
              'e': ['m', 'v', 'd', 'tt', 'tp', 'ike'], 
              'lambda': ['m', 'v', 'd', 'tt', 'tp', 'ike'], 
              'e/cu': ['m', 'v', 'd', 'tt', 'tp', 'ike']}

(X_trainDict, y_trainDict) = preprocess_data.prepareTrainAndTest(trainDf, predictors)
(X_testDict, y_testDict) = preprocess_data.prepareTrainAndTest(testDf, predictors)
sc_yDict = preprocess_data.standardScaleData(X_trainDict, X_testDict, y_trainDict)

[regressors, regNames, scorers] = modelling.createRegressors()

#***** PCA tuning ******
CVFlag = False
targetVar = 'cu'
bestPCA = parameter_tuning.pcaTuning(X_trainDict, y_trainDict, X_testDict, y_testDict, targetVar, 
                                     sc_yDict, logFlagDict, regressors, regNames, scorers, CVFlag = CVFlag)

#**** stepwise tuning of XGB ******
(X, X_test) = modelling.transformWithKernelPca(X_trainDict[targetVar], X_testDict[targetVar], None, 'poly', 5, 0.16, 1)
y = y_trainDict[targetVar]
regressor = regressors[0]
tuningDict = {}

#Step 1:fix learning rate and determine optimum n_estimators:
parameters = {'n_estimators': [i for i in range(50, 1000, 50)],
              'learning_rate': [0.1]}
score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-1'] = {'params': params, 'score': score}
print(tuningDict)

#Step 2:Tune max_depth and min_child_weight:
parameters = {'n_estimators': [650],
             'learning_rate': [0.1],
             'max_depth': [i for i in range(1,21)],
             'min_child_weight': [i for i in range(1,21)]}
score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-2'] = {'params': params, 'score': score}
print(tuningDict)

#Step 3:Tune gamma:
parameters = {'n_estimators': [650],
             'learning_rate': [0.1],
             'max_depth': [3],
             'min_child_weight': [2],
             'gamma': [i*0.1 for i in range(0, 100, 2)]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-3'] = {'params': params, 'score': score}
print(tuningDict)

#Step 4:Tune subsample and colsample_bytree:
parameters = {'n_estimators': [650],
             'learning_rate': [0.1],
             'max_depth': [3],
             'min_child_weight': [2],
             'gamma': [0.0],
             'subsample': [i*0.1 for i in range(1,11)],
             'colsample_bytree': [i*0.1 for i in range(1,11)]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-4'] = {'params': params, 'score': score}
print(tuningDict)

#Step 5:Tuning Regularization Parameters:
parameters = {'n_estimators': [650],
             'learning_rate': [0.1],
             'max_depth': [3],
             'min_child_weight': [2],
             'gamma': [0.0],
             'subsample': [1.0],
             'colsample_bytree': [1.0],
             'reg_alpha':[i*0.1 for i in range(0,11)],
             'reg_lambda':[i*0.1 for i in range(1,21)]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-5'] = {'params': params, 'score': score}
print(tuningDict)

#Step 6: Tuning (decreasing) learning rate and n_estimators again:
parameters = {'n_estimators': [i for i in range(50, 1000, 10)],
             'learning_rate': [0.01*i for i in range(1, 20, 1)],
             'max_depth': [3],
             'min_child_weight': [2],
             'gamma': [0.0],
             'subsample': [1.0],
             'colsample_bytree': [1.0],
             'reg_alpha':[0.0],
             'reg_lambda':[1.0]}

score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, scorers[0])
tuningDict['Step-6'] = {'params': params, 'score': score}
print(tuningDict)

#Prediction
testData = testDf.copy(deep=True)
results = modelling.XGBPrediction(X, y, X_test, 650, 0.1, 3, 2, 0.0, 1.0, 1.0, 0.0, 1.0)
testData[str(targetVar)+"_pred"] = preprocess_data.reverseTransformPredictions(results, sc_yDict[targetVar], logFlagDict[targetVar])
testData.to_csv(dataPath+"output/test_trans_tuned_xgb_with_tuned_pca_CV-"+CVFlag+"_"+targetVar+".csv", index=False)

trainData = trainDf.copy(deep=True)
results = modelling.XGBPrediction(X, y, X, 650, 0.1, 3, 2, 0.0, 1.0, 1.0, 0.0, 1.0)
trainData[str(targetVar)+"_pred"] = preprocess_data.reverseTransformPredictions(results, sc_yDict[targetVar], logFlagDict[targetVar])
trainData.to_csv(dataPath+"output/train_trans_tuned_xgb_with_tuned_pca_CV-"+CVFlag+"_"+targetVar+".csv", index=False)