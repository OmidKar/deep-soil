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
              'lambda': ['m', 'v', 'd', 'tt', 'tp', 'ike']}

(X_trainDict, y_trainDict) = preprocess_data.prepareTrainAndTest(trainDf, predictors)
(X_testDict, y_testDict) = preprocess_data.prepareTrainAndTest(testDf, predictors)
sc_yDict = preprocess_data.standardScaleData(X_trainDict, X_testDict, y_trainDict)

[regressors, regNames, scorers] = modelling.createRegressors()

#loop over target variables
pcaCVFlag = True
xgbCVFlag = True
logFile = open(dataPath+"output/settings_results.txt","w+")
logFile.write("Original Flags are: pca flag = "+str(pcaCVFlag)+" and XGB flag = "+str(xgbCVFlag)+".\n")

for targetVar in ['cu', 'AB', 'e', 'lambda', 'e/cu']:
    logFile.write("Runing the analysis for "+str(targetVar)+" ...\n")
    sc = sc_yDict[targetVar]
    logFlag = logFlagDict[targetVar]
    y_test = y_testDict[targetVar]
    
    #***** PCA tuning ******
    (bestPCA, best) = parameter_tuning.pcaTuning(X_trainDict, y_trainDict, X_testDict, y_testDict, targetVar, 
                                         sc, logFlag, regressors, regNames, scorers, CVFlag = pcaCVFlag)
    
    logFile.write("Best PCA params are: \n"+str(bestPCA))
    logFile.write("Best result achieved is: \n"+str(best)+"\n")
    
    #**** stepwise tuning of XGB ******
    (X, X_test) = modelling.transformWithKernelPca(X_trainDict[targetVar], X_testDict[targetVar], 
                                                   bestPCA['comp'], 'poly', bestPCA['deg'], bestPCA['gamma'], bestPCA['coef'])
    y = y_trainDict[targetVar]
    regressor = regressors[0]
    tuningDict = {}
    
    #Step 1:fix learning rate and determine optimum n_estimators:
    parameters = {'n_estimators': [i for i in range(50, 1000, 50)],
                  'learning_rate': [0.1]}
    score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, X_test, y_test, 
                                                              scorers[0], sc, logFlag, CVFlag = xgbCVFlag)
    tuningDict['Step-1'] = {'params': params, 'score': score}
    print(tuningDict)
    
    #Step 2:Tune max_depth and min_child_weight:
    parameters = {'n_estimators': [params['n_estimators']],
                 'learning_rate': [params['learning_rate']],
                 'max_depth': [i for i in range(1,21)],
                 'min_child_weight': [i for i in range(1,21)]}
    score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, X_test, y_test, 
                                                              scorers[0], sc, logFlag, CVFlag = xgbCVFlag)
    tuningDict['Step-2'] = {'params': params, 'score': score}
    print(tuningDict)
    
    #Step 3:Tune gamma:
    #parameters = {'n_estimators': [tuningDict['Step-2']['params']['n_estimators']],
    #             'learning_rate': [tuningDict['Step-2']['params']['learning_rate']],
    #             'max_depth': [tuningDict['Step-2']['params']['max_depth']],
    #             'min_child_weight': [tuningDict['Step-2']['params']['min_child_weight']],
    #             'gamma': [i*0.1 for i in range(0, 100, 2)]}
    
    #score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, X_test, y_test, 
    #                                                          scorers[0], sc, logFlag, CVFlag = xgbCVFlag)
    #tuningDict['Step-3'] = {'params': params, 'score': score}
    #print(tuningDict)
    
    #Step 4:Tune subsample and colsample_bytree:
    parameters = {'n_estimators': [params['n_estimators']],
                 'learning_rate': [params['learning_rate']],
                 'max_depth': [params['max_depth']],
                 'min_child_weight': [params['min_child_weight']],
                 'gamma': [0.0],
                 'subsample': [i*0.1 for i in range(1,11)],
                 'colsample_bytree': [i*0.1 for i in range(1,11)]}
    
    score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, X_test, y_test, 
                                                              scorers[0], sc, logFlag, CVFlag = xgbCVFlag)
    tuningDict['Step-4'] = {'params': params, 'score': score}
    print(tuningDict)
    
    #Step 5:Tuning Regularization Parameters:
    parameters = {'n_estimators': [params['n_estimators']],
                 'learning_rate': [params['learning_rate']],
                 'max_depth': [params['max_depth']],
                 'min_child_weight': [params['min_child_weight']],
                 'gamma': [0.0],
                 'subsample': [params['subsample']],
                 'colsample_bytree': [params['colsample_bytree']],
                 'reg_alpha':[0.0],
                 'reg_lambda':[i*0.1 for i in range(1,21)]}
    
    score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, X_test, y_test, 
                                                              scorers[0], sc, logFlag, CVFlag = xgbCVFlag)
    tuningDict['Step-5'] = {'params': params, 'score': score}
    print(tuningDict)
    
    #Step 6: Tuning (decreasing) learning rate and n_estimators again:
    parameters = {'n_estimators': [i for i in range(10, 1000, 10)],
                 'learning_rate': [0.01*i for i in range(2, 100, 2)],
                 'max_depth': [params['max_depth']],
                 'min_child_weight': [params['min_child_weight']],
                 'gamma': [0.0],
                 'subsample': [params['subsample']],
                 'colsample_bytree': [params['colsample_bytree']],
                 'reg_alpha':[0.0],
                 'reg_lambda':[params['reg_lambda']]}
    
    score, params = parameter_tuning.evaluateModelPerformance(regressor, parameters, X, y, X_test, y_test, 
                                                              scorers[0], sc, logFlag, CVFlag = xgbCVFlag)
    tuningDict['Step-6'] = {'params': params, 'score': score}
    print(tuningDict)
    
    logFile.write("Best XGB params are: \n"+str(params))
    logFile.write("Best result achieved is: \n"+str(tuningDict['Step-6']['score'])+"\n")
    
    #Prediction
    testData = testDf.copy(deep=True)
    results = modelling.XGBPrediction(X, y, X_test, params['n_estimators'], params['learning_rate'], 
                                      params['max_depth'], params['min_child_weight'], 0.0, 
                                      params['subsample'], params['colsample_bytree'], 0.0, params['reg_lambda'])
    
    testData[str(targetVar)+"_pred"] = preprocess_data.reverseTransformPredictions(results, sc_yDict[targetVar], logFlagDict[targetVar])
    testData.to_csv(dataPath+"output/test_trans_XGB-CV-"+str(xgbCVFlag)+"_"+targetVar+".csv", index=False)
    logFile.write("File "+dataPath+"output/test_trans_XGB-CV-"+str(xgbCVFlag)+"_"+targetVar+".csv"+" was created!\n")
    
    trainData = trainDf.copy(deep=True)
    results = modelling.XGBPrediction(X, y, X, params['n_estimators'], params['learning_rate'], 
                                      params['max_depth'], params['min_child_weight'], 0.0, 
                                      params['subsample'], params['colsample_bytree'], 0.0, params['reg_lambda'])
    
    trainData[str(targetVar)+"_pred"] = preprocess_data.reverseTransformPredictions(results, sc_yDict[targetVar], logFlagDict[targetVar])
    trainData.to_csv(dataPath+"output/train_trans_XGB-CV-"+str(xgbCVFlag)+"_"+targetVar+".csv", index=False)
    logFile.write("File "+dataPath+"output/train_trans_XGB-CV-"+str(xgbCVFlag)+"_"+targetVar+".csv"+" was created!\n")
 
logFile.close()