#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Kaggle, Home Credit project
Author: Jacqueline Huvanandana
Created: 25/05/2018
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import stats

from datetime import datetime
from sklearn import linear_model
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from keras import backend as K
from keras import optimizers
from keras import callbacks
from keras import layers
from keras import models
from keras.wrappers.scikit_learn import KerasClassifier

## initialise
start = datetime.now()
os.chdir('C:/Users/Jacqueline/Documents/DataScience/Projects/homeCred')

## import data
dfTr = pd.read_csv('data/application_train.csv')
dfTe = pd.read_csv('data/application_test.csv')

buData = pd.read_csv('data/bureau.csv')
buGrp = buData.groupby(by='SK_ID_CURR').mean()
del buData

bbData = pd.read_csv('data/bureau_balance.csv')
bbGrp  = bbData.groupby(by='SK_ID_BUREAU').mean()
del bbData

pcbData = pd.read_csv('data/POS_CASH_BALANCE.csv')
pcbGrp  = pcbData.groupby(by='SK_ID_CURR').mean()
del pcbData

ccbData = pd.read_csv('data/credit_card_balance.csv')
ccbGrp  = ccbData.groupby(by='SK_ID_CURR').mean()
del ccbData

paData = pd.read_csv('data/previous_application.csv')
paGrp  = paData.groupby(by='SK_ID_CURR').mean()
del paData

ipData = pd.read_csv('data/installments_payments.csv')
ipGrp  = ipData.groupby(by='SK_ID_CURR').mean()
del ipData


mrgGrp = buGrp.join(bbGrp, on='SK_ID_BUREAU', how='left', rsuffix='_bb')
mrgGrp = mrgGrp.join(pcbGrp, on='SK_ID_CURR', how='left', rsuffix='_pc')
mrgGrp = mrgGrp.join(ccbGrp, on='SK_ID_CURR', how='left', rsuffix='_cc')
mrgGrp = mrgGrp.join(paGrp, on='SK_ID_CURR', how='left', rsuffix='_pa')
mrgGrp = mrgGrp.join(ipGrp, on='SK_ID_CURR', how='left', rsuffix='_ip')

##join to other data
dfTr = dfTr.join(mrgGrp, on='SK_ID_CURR', how='left', lsuffix='_l',rsuffix='_r')
dfTe = dfTe.join(mrgGrp, on='SK_ID_CURR', how='left', lsuffix='_l',rsuffix='_r')

## pre-processing
# categorical data

# sex
cat_gen = pd.api.types.CategoricalDtype(categories=['M','F','O'])
dfTr['CODE_GENDER'] = dfTr['CODE_GENDER'].astype(cat_gen)
dfTe['CODE_GENDER'] = dfTe['CODE_GENDER'].astype(cat_gen)
dfTr['CODE_GENDER'].fillna('O',inplace=True)
dfTe['CODE_GENDER'].fillna('O',inplace=True)

# cash/revolving loans
cat_loans = pd.api.types.CategoricalDtype(categories=['Cash loans','Revolving loans'])
dfTr['NAME_CONTRACT_TYPE'] = dfTr['NAME_CONTRACT_TYPE'].astype(cat_loans)
dfTe['NAME_CONTRACT_TYPE'] = dfTe['NAME_CONTRACT_TYPE'].astype(cat_loans)

# accompanied by
cat_suite = pd.api.types.CategoricalDtype(categories=['Children',
 'Family',
 'Group of people',
 'Other_A',
 'Other_B',
 'Spouse, partner',
 'Unaccompanied'])
dfTr['NAME_TYPE_SUITE'] = dfTr['NAME_TYPE_SUITE'].astype(cat_suite)
dfTe['NAME_TYPE_SUITE'] = dfTe['NAME_TYPE_SUITE'].astype(cat_suite)
# assume unaccompanied
dfTr['NAME_TYPE_SUITE'].fillna('Unaccompanied',inplace=True)
dfTe['NAME_TYPE_SUITE'].fillna('Unaccompanied',inplace=True)

# education
cat_edu = pd.api.types.CategoricalDtype(categories=['Academic degree',
 'Higher education',
 'Incomplete higher',
 'Lower secondary',
 'Secondary / secondary special'])
dfTr['NAME_EDUCATION_TYPE'] = dfTr['NAME_EDUCATION_TYPE'].astype(cat_edu)
dfTe['NAME_EDUCATION_TYPE'] = dfTe['NAME_EDUCATION_TYPE'].astype(cat_edu)

# income
cat_income = pd.api.types.CategoricalDtype(categories=['Businessman',
 'Commercial associate',
 'Maternity leave',
 'Pensioner',
 'State servant',
 'Student',
 'Unemployed',
 'Working'])
dfTr['NAME_INCOME_TYPE'] = dfTr['NAME_INCOME_TYPE'].astype(cat_income)
dfTe['NAME_INCOME_TYPE'] = dfTe['NAME_INCOME_TYPE'].astype(cat_income)

# family status
cat_fam = pd.api.types.CategoricalDtype(categories=['Civil marriage',
 'Married',
 'Separated',
 'Single / not married',
 'Unknown',
 'Widow'])
dfTr['NAME_FAMILY_STATUS'] = dfTr['NAME_FAMILY_STATUS'].astype(cat_fam)
dfTe['NAME_FAMILY_STATUS'] = dfTe['NAME_FAMILY_STATUS'].astype(cat_fam)

# housing type
cat_house = pd.api.types.CategoricalDtype(categories=['Co-op apartment',
 'House / apartment',
 'Municipal apartment',
 'Office apartment',
 'Rented apartment',
 'With parents'])
dfTr['NAME_HOUSING_TYPE'] = dfTr['NAME_HOUSING_TYPE'].astype(cat_house)
dfTe['NAME_HOUSING_TYPE'] = dfTe['NAME_HOUSING_TYPE'].astype(cat_house)

# occupation type
cat_occup = pd.api.types.CategoricalDtype(categories=['Accountants',
 'Cleaning staff',
 'Cooking staff',
 'Core staff',
 'Drivers',
 'HR staff',
 'High skill tech staff',
 'IT staff',
 'Laborers',
 'Low-skill Laborers',
 'Managers',
 'Medicine staff',
 'Private service staff',
 'Realty agents',
 'Sales staff',
 'Secretaries',
 'Security staff',
 'Waiters/barmen staff'
 ,'Other'])
dfTr['OCCUPATION_TYPE'] = dfTr['OCCUPATION_TYPE'].astype(cat_occup)
dfTe['OCCUPATION_TYPE'] = dfTe['OCCUPATION_TYPE'].astype(cat_occup)
dfTr['OCCUPATION_TYPE'].fillna('Other',inplace=True)
dfTe['OCCUPATION_TYPE'].fillna('Other',inplace=True)

# weekday start
cat_wkday = pd.api.types.CategoricalDtype(categories=['MONDAY',
 'TUESDAY',
 'WEDNESDAY',
 'THURSDAY',
 'FRIDAY',
 'SATURDAY',
 'SUNDAY',
 ])

dfTr['WEEKDAY_APPR_PROCESS_START'] = dfTr['WEEKDAY_APPR_PROCESS_START'].astype(cat_wkday)
dfTe['WEEKDAY_APPR_PROCESS_START'] = dfTe['WEEKDAY_APPR_PROCESS_START'].astype(cat_wkday)

# organisation
cat_org = pd.api.types.CategoricalDtype(categories=set(dfTr.ORGANIZATION_TYPE))
dfTr['ORGANIZATION_TYPE'] = dfTr['ORGANIZATION_TYPE'].astype(cat_org)
dfTe['ORGANIZATION_TYPE'] = dfTe['ORGANIZATION_TYPE'].astype(cat_org)

# living
cat_liv = pd.api.types.CategoricalDtype(categories=['not specified',
 'org spec account',
 'reg oper account',
 'reg oper spec account'])
dfTr['FONDKAPREMONT_MODE'] = dfTr['FONDKAPREMONT_MODE'].astype(cat_liv)
dfTe['FONDKAPREMONT_MODE'] = dfTe['FONDKAPREMONT_MODE'].astype(cat_liv)
dfTr['FONDKAPREMONT_MODE'].fillna('not specified',inplace=True)
dfTe['FONDKAPREMONT_MODE'].fillna('not specified',inplace=True)

# house type
cat_htype = pd.api.types.CategoricalDtype(categories=['block of flats',
 'specific housing',
 'terraced house'
 ,'other'])
dfTr['HOUSETYPE_MODE'] = dfTr['HOUSETYPE_MODE'].astype(cat_htype)
dfTe['HOUSETYPE_MODE'] = dfTe['HOUSETYPE_MODE'].astype(cat_htype)
dfTr['HOUSETYPE_MODE'].fillna('other',inplace=True)
dfTe['HOUSETYPE_MODE'].fillna('other',inplace=True)

# walls material
cat_walls = pd.api.types.CategoricalDtype(categories=['Block',
 'Mixed',
 'Monolithic',
 'Others',
 'Panel',
 'Stone, brick',
 'Wooden'])
dfTr['WALLSMATERIAL_MODE'] = dfTr['WALLSMATERIAL_MODE'].astype(cat_walls)
dfTe['WALLSMATERIAL_MODE'] = dfTe['WALLSMATERIAL_MODE'].astype(cat_walls)
dfTr['WALLSMATERIAL_MODE'].fillna('Others',inplace=True)
dfTe['WALLSMATERIAL_MODE'].fillna('Others',inplace=True)

# emergency state (assume 0 where nan)
dfTr['EMERGENCYSTATE_MODE'] = dfTr['EMERGENCYSTATE_MODE'].map(lambda k: int(k=='Yes'))
dfTe['EMERGENCYSTATE_MODE'] = dfTe['EMERGENCYSTATE_MODE'].map(lambda k: int(k=='Yes'))

# owns a car (assume 0 where null)
dfTr['FLAG_OWN_CAR'] = dfTr['FLAG_OWN_CAR'].map(lambda k: int(k=='Y'))
dfTe['FLAG_OWN_CAR'] = dfTe['FLAG_OWN_CAR'].map(lambda k: int(k=='Y'))
# owns realty (assume 0 where null)
dfTr['FLAG_OWN_REALTY'] = dfTr['FLAG_OWN_REALTY'].map(lambda k: int(k=='Y'))
dfTe['FLAG_OWN_REALTY'] = dfTe['FLAG_OWN_REALTY'].map(lambda k: int(k=='Y'))

## feature engineering
# interaction: sex:income
dfTr['sexIncome'] = dfTr['CODE_GENDER'].cat.codes * dfTr['AMT_INCOME_TOTAL']
dfTe['sexIncome'] = dfTe['CODE_GENDER'].cat.codes * dfTe['AMT_INCOME_TOTAL']
# percentage of days employed
dfTr['pcEmployed'] = dfTr['DAYS_EMPLOYED'] / dfTr['DAYS_BIRTH']
dfTe['pcEmployed'] = dfTe['DAYS_EMPLOYED'] / dfTe['DAYS_BIRTH']
# house:wallType
dfTr['houseWalls'] = dfTr['HOUSETYPE_MODE'].cat.codes * dfTr['WALLSMATERIAL_MODE'].cat.codes
dfTe['houseWalls'] = dfTe['HOUSETYPE_MODE'].cat.codes * dfTe['WALLSMATERIAL_MODE'].cat.codes

## model fitting
X_train = dfTr
X_test  = dfTe
Y_train = X_train.TARGET.astype(np.int64).copy()

# fillna for numerics
obj_numer = X_test.select_dtypes(include=['int8','int64','float64']).copy()
for colName in list(obj_numer):
    X_train[colName].fillna(value=0,inplace=True)
    X_test[colName].fillna(value=0,inplace=True)

obj_categ = X_train.select_dtypes(include=['category']).copy()
for colName in list(obj_categ):
    X_train[colName] = X_train[colName].cat.codes
    X_test[colName]  = X_test[colName].cat.codes

## model fitting
def freq_item(series, set_list):
    series = list(np.array(series,dtype=np.float64))
    countVec = []
    for set_item in set_list:
        countVec.append(series.count(set_item))
    return countVec

negIdx = Y_train.loc[Y_train==0].index
posIdx = Y_train.loc[Y_train==1].index

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

featImp = rf.feature_importances_
sortIdx = np.argsort(featImp)[-50:]
allNameList = list(X_test)
#pVec   = []
#aucVec = []
#lr = linear_model.LogisticRegression()
#nRow = len(Y_train)
#nFeats = len(allNameList)
#for iCol in range(nFeats):
#    colName = allNameList[iCol]
#    print('%d of %d, %s'%(iCol,nFeats,colName))
#    colVar  = X_train[colName]
#    colVar.fillna(0,inplace=True)
#    
#    rsFeat  = np.reshape(np.array(colVar,dtype=colVar.dtype), (nRow,1))
#    lr.fit(rsFeat,Y_train)
#    
#    probs = lr.predict_proba(rsFeat)
#    fpr,tpr,thresh = metrics.roc_curve(Y_train,probs[:,1])
#    aucVec.append(metrics.auc(fpr,tpr))
#    
#    is_binary = len(set(colVar))==2
#    grp0  = colVar.loc[negIdx]
#    grp1  = colVar.loc[posIdx]
#    
#    if is_binary:
#        setTarg = list(set(colVar))
#        mat     = np.column_stack((freq_item(grp0, setTarg), freq_item(grp1, setTarg)))
#        fstat, pval = stats.fisher_exact(mat)
#    else:
#        ustat, pval = stats.mannwhitneyu(grp0,grp1)
#        
#    pVec.append(pval)
#    
#sortIdx  = np.argsort(aucVec)
#mapNames = map(lambda k: allNameList[k], sortIdx[-20:])

svc = svm.LinearSVC()
featSel = feature_selection.SelectFromModel(svc)
featSel.fit(X_train, Y_train)
sortIdx = featSel.get_support(indices=True)-1

mapNames = map(lambda k: allNameList[k], sortIdx)
colNames = list(mapNames)

# user tensorflow backend
sess = tf.Session()
K.set_session(sess)


def model():
    model = models.Sequential([
        layers.Dense(64, input_dim=X_train[colNames].shape[1], activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='sigmoid'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

pipe = pipeline.Pipeline([
    ('rescale', preprocessing.StandardScaler()),
#    ('logit',linear_model.LogisticRegression())
    ('logit', KerasClassifier(build_fn=model, nb_epoch=20, batch_size=128,
                           validation_split=0.2, callbacks=[early_stopping]))
])

print('Beginning fit with: %d variables'%(len(colNames)))
pipe.fit(X_train[colNames], Y_train)

probTr = pipe.predict_proba(X_train[colNames])
fpr, tpr, thresh = metrics.roc_curve(Y_train, probTr[:, 1], pos_label=1)
aucScore = metrics.auc(fpr, tpr)
print(aucScore)

probTe = pipe.predict_proba(X_test[colNames])
predTe = probTe[:, 1]

outDf = pd.DataFrame(np.column_stack((dfTe.SK_ID_CURR, predTe)), columns=['SK_ID_CURR', 'TARGET'])
outDf.SK_ID_CURR = outDf.SK_ID_CURR.astype(np.int32)
outDf.to_csv('output/prediction_lm.csv', index=False)

print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')
