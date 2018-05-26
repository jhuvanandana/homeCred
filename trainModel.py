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

from datetime import datetime
from sklearn import linear_model
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing

from keras import backend as be
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

buTr = pd.read_csv('data/bureau.csv')
buGrp = buTr.groupby(by='SK_ID_CURR').sum()

##join to other data
dfTr = dfTr.join(buGrp, on='SK_ID_CURR', how='left', rsuffix='_bu')
dfTe = dfTe.join(buGrp, on='SK_ID_CURR', how='left', rsuffix='_bu')

## pre-processing
dfTr['CODE_GENDER'] = dfTr['CODE_GENDER'].map(lambda k: int(k == 'M'))
dfTe['CODE_GENDER'] = dfTe['CODE_GENDER'].map(lambda k: int(k == 'M'))

dfTr['FLAG_OWN_CAR'] = dfTr['FLAG_OWN_CAR'].map(lambda k: int(k == 'Y'))
dfTe['FLAG_OWN_CAR'] = dfTe['FLAG_OWN_CAR'].map(lambda k: int(k == 'Y'))

## feature engineering
dfTr['sexIncome'] = dfTr['CODE_GENDER'] * dfTr['AMT_INCOME_TOTAL']
dfTe['sexIncome'] = dfTe['CODE_GENDER'] * dfTe['AMT_INCOME_TOTAL']
## percentage employed
dfTr['pcEmployed'] = dfTr['DAYS_EMPLOYED'] / dfTr['DAYS_BIRTH']
dfTe['pcEmployed'] = dfTe['DAYS_EMPLOYED'] / dfTe['DAYS_BIRTH']

mod = linear_model.LogisticRegression()

## model fitting
selColNames = ['CODE_GENDER'
    , 'AMT_INCOME_TOTAL'
    , 'AMT_CREDIT'
    , 'AMT_ANNUITY'
    , 'sexIncome'
    , 'FLAG_OWN_CAR'
    , 'LANDAREA_AVG'
    , 'CNT_CHILDREN'
    ,'CNT_FAM_MEMBERS'
    , 'DAYS_EMPLOYED'
    , 'DAYS_BIRTH'
    , 'pcEmployed'
    , 'AMT_CREDIT_SUM'
    , 'CREDIT_DAY_OVERDUE'
    , 'DAYS_CREDIT_ENDDATE'
    , 'AMT_CREDIT_MAX_OVERDUE'
    , 'CNT_CREDIT_PROLONG'
    , 'AMT_CREDIT_SUM'
    , 'AMT_CREDIT_SUM_DEBT'
    , 'AMT_CREDIT_SUM_LIMIT'
    , 'AMT_CREDIT_SUM_OVERDUE'
    , 'DAYS_CREDIT_UPDATE'
    , 'AMT_ANNUITY_bu'
   ]

nRow = len(dfTr)
reshapeTarg = np.reshape(np.array(dfTr.TARGET), (nRow, 1))
aucVec = []
selNameVec = []
for selName in selColNames:
    feats = dfTr[selName]
    feats.fillna(0, inplace=True)
    selFeat = np.reshape(np.array(feats), (nRow, 1))
    mod.fit(selFeat, reshapeTarg)
    probs = mod.predict_proba(selFeat)
    fpr, tpr, thresh = metrics.roc_curve(reshapeTarg, probs[:, 1], pos_label=1)
    calcAuc = metrics.auc(fpr, tpr)
    aucVec.append(calcAuc)
    if calcAuc > 0.50:
        selNameVec.append(selName)

selNameVec.append('pcEmployed')
X_train = dfTr
X_train.is_copy = False
X_train.fillna(value=0, inplace=True)

X_test = dfTe
X_test.is_copy = False
X_test.fillna(value=0, inplace=True)


## model fitting
# user tensorflow backend
sess = tf.Session()
be.set_session(sess)

def model():
    model = models.Sequential([
        layers.Dense(64, input_dim=X_train[selNameVec].shape[1], activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

pipe = pipeline.Pipeline([
    ('rescale', preprocessing.StandardScaler()),
    ('nn', KerasClassifier(build_fn=model, nb_epoch=10, batch_size=128,
                           validation_split=0.2, callbacks=[early_stopping]))
])

Y_train = np.array(X_train.TARGET,dtype=np.float64)

pipe.fit(X_train[selNameVec], Y_train)

probTr = pipe.predict_proba(X_train[selNameVec])
fpr, tpr, thresh = metrics.roc_curve(Y_train, probTr[:, 1], pos_label=1)
aucScore = metrics.auc(fpr, tpr)
print(aucScore)

probTe = mod.predict_proba(X_test[selNameVec])
predTe = probTe[:, 1]

outDf = pd.DataFrame(np.column_stack((dfTe.SK_ID_CURR, predTe)), columns=['SK_ID_CURR', 'TARGET'])
outDf.SK_ID_CURR = outDf.SK_ID_CURR.astype(np.int32)
outDf.to_csv('output/prediction_lm.csv', index=False)

print('Script took ', datetime.now() - start, ' HH:MM:SS.SSSSSS')
