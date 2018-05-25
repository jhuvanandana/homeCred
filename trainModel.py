#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Kaggle, Home Credit project
Author: Jacqueline Huvanandana
Created: 25/05/2018
"""
import csv
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import string

from datetime import datetime
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

    ## initialise
    start = datetime.now()
    os.chdir('C:/Users/Jacqueline/Documents/DataScience/Projects/homeCred')

    ## import data
    dfTr = pd.read_csv('data/application_train.csv')
    dfTe = pd.read_csv('data/application_test.csv')

    buTr  = pd.read_csv('data/bureau.csv')
    buGrp = buTr.groupby(by='SK_ID_CURR').sum()

    ## join to other data
    dfTr = dfTr.join(buGrp, on='SK_ID_CURR',how='left',rsuffix='_bu')
    dfTe = dfTe.join(buGrp, on='SK_ID_CURR',how='left',rsuffix='_bu')

    ## pre-processing
    dfTr['CODE_GENDER'] = dfTr['CODE_GENDER'].map(lambda k: int(k=='M'))
    dfTe['CODE_GENDER'] = dfTe['CODE_GENDER'].map(lambda k: int(k=='M'))

    dfTr['FLAG_OWN_CAR'] = dfTr['FLAG_OWN_CAR'].map(lambda k: int(k=='Y'))
    dfTe['FLAG_OWN_CAR'] = dfTe['FLAG_OWN_CAR'].map(lambda k: int(k=='Y'))

    ## feature engineering
    dfTr['sexIncome'] = dfTr['CODE_GENDER']*dfTr['AMT_INCOME_TOTAL']
    dfTe['sexIncome'] = dfTe['CODE_GENDER']*dfTe['AMT_INCOME_TOTAL']
    ## percentage employed
    dfTr['pcEmployed'] = dfTr['DAYS_EMPLOYED']/dfTr['DAYS_BIRTH']
    dfTe['pcEmployed'] = dfTe['DAYS_EMPLOYED']/dfTe['DAYS_BIRTH']

    # mod = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(3,2),random_state=0)
    mod = LogisticRegression()
    # mod = RandomForestClassifier(n_estimators=2)

    ## model fitting
    selColNames = ['CODE_GENDER'
                ,'AMT_INCOME_TOTAL'
                ,'AMT_CREDIT'
                ,'AMT_ANNUITY'
                ,'sexIncome'
                ,'FLAG_OWN_CAR'
                ,'LANDAREA_AVG'
                ,'CNT_CHILDREN'
                ,'DAYS_EMPLOYED'
                ,'DAYS_BIRTH'
                ,'pcEmployed'
                ,'AMT_CREDIT_SUM'
                ,'CREDIT_DAY_OVERDUE'
                ,'DAYS_CREDIT_ENDDATE'
                ,'AMT_CREDIT_MAX_OVERDUE'
                ,'CNT_CREDIT_PROLONG'
                ,'AMT_CREDIT_SUM'
                ,'AMT_CREDIT_SUM_DEBT'
                ,'AMT_CREDIT_SUM_LIMIT'
                ,'AMT_CREDIT_SUM_OVERDUE'
                ,'DAYS_CREDIT_UPDATE'
                ,'AMT_ANNUITY_bu'
                ]

    nRow = len(dfTr)
    reshapeTarg = np.reshape(dfTr.TARGET,(nRow,1))
    aucVec = []
    selNameVec = []
    for selName in selColNames:
        feats   = dfTr[selName]
        feats.fillna(0,inplace=True)
        selFeat = np.reshape(feats,(nRow,1))
        mod.fit(selFeat, reshapeTarg)
        probs = mod.predict_proba(selFeat)
        fpr, tpr, thresh = roc_curve(reshapeTarg, probs[:,1], pos_label=1)
        calcAuc = auc(fpr,tpr)
        aucVec.append(calcAuc)
        if calcAuc>0.54:
            selNameVec.append(selName)

    selNameVec.append('pcEmployed')
    filtTr = dfTr
    filtTr.is_copy = False
    filtTr.fillna(value=0,inplace=True)

    filtTe = dfTe
    filtTe.is_copy = False
    filtTe.fillna(value=0,inplace=True)

    mod.fit(filtTr[selNameVec],filtTr.TARGET)
    probTr = mod.predict_proba(filtTr[selNameVec])
    fpr, tpr, thresh = roc_curve(filtTr.TARGET,probTr[:,1],pos_label=1)
    print auc(fpr,tpr)
    probTe = mod.predict_proba(filtTe[selNameVec])
    predTe = probTe[:,1]

    outDf = pd.DataFrame(np.column_stack((dfTe.SK_ID_CURR, predTe)),columns=['SK_ID_CURR','TARGET'])
    outDf.SK_ID_CURR = outDf.SK_ID_CURR.astype(np.int32)
    outDf.to_csv('output/prediction_lm.csv', index=False)

print 'Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS'
