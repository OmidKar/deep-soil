# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:57:55 2019

@author: Omid
"""

def getBestR2(resDf):
    m = max(resDf['CrossValMeans'])
    df = resDf[resDf['CrossValMeans'] == max(resDf['CrossValMeans'])].reset_index()
    return [df.loc[0,'Algorithm'], m]

def updateBestResults(results, best, n, k, d, g, c, bestPCA):
    [algo, r2] = getBestR2(results['r2'])
    if r2 > best[1]:
        print("for values: ", n, k, d, g, c)
        print("achieved:", [algo, r2], "\n")        
        return ([algo, r2], {'comp': n, 'kernel': k, 'deg': d, 'gamma': g, 'coef': c})
    else:
        return (best, bestPCA)