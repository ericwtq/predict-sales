#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:33:59 2017

@author: ericwtq
"""


import time
import numpy as np
import pandas as pd
import gc; gc.enable()
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, metrics
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

np.random.seed(11122)

# store the total processing time
start_time = time.time()
tcurrent   = start_time

print('Multiple regressors - Part II\n')
print('Datasets reading')


# read datasets
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}
data = {
    'tra': pd.read_csv('./data/final.csv', dtype=dtypes, parse_dates=['date']),
    'tes': pd.read_csv('./data/test.csv', dtype=dtypes, parse_dates=['date']),
    'ite': pd.read_csv('./data/items.csv'),
    'sto': pd.read_csv('./data/stores.csv'),
    'trn': pd.read_csv('./data/transactions.csv', parse_dates=['date']),
    'hol': pd.read_csv('./data/holiday.csv', dtype={'transferred':str}, parse_dates=['date']),
    'oil': pd.read_csv('./data/oil.csv',parse_dates=['date'])
    }


# dataset processing
print('Datasets processing')

train = data['tra'][(data['tra']['date'].dt.month == 8) & (data['tra']['date'].dt.day > 15)]
del data['tra']; gc.collect();

target = train['unit_sales'].values
target[target < 0.] = 0.
train['unit_sales'] = np.log1p(target)

def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

def df_transform(df):
    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['date'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    df = df.fillna(-1)
    return df


def NWRMSLE(y, pred, w):
    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5


data['ite'].columns.values[2] = 'ItemClass'
ItemFamily = pd.get_dummies(data['ite'].family)
type(ItemFamily)
data['ite'] = pd.merge(data['ite'],ItemFamily,left_index=True,right_index=True)
data['ite'].ix[:1,['class']]
data['ite'] = data['ite'].drop(['family'],axis=1)
del ItemFamily;gc.collect();


train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])
test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])
del data['tes']; gc.collect();
del data['ite']; gc.collect();

train = pd.merge(train, data['trn'], how='left', on=['date','store_nbr'])
test = pd.merge(test, data['trn'], how='left', on=['date','store_nbr'])
del data['trn']; gc.collect();

target = train['transactions'].values
target[target < 0.] = 0.
train['transactions'] = np.log1p(target)

StoreDataDM = pd.get_dummies(data['sto'].ix[:,1:3])
type(StoreDataDM)
StoreDataDM.shape
StoreDataDM.columns
StoreDataDM
StoreDataDMCluster = pd.get_dummies(data['sto'].cluster)
StoreDataDMCluster.columns
StoreDataDMCluster.shape
StoreData1=pd.merge(StoreDataDM,StoreDataDMCluster,left_index=True,right_index=True)
del StoreDataDM;gc.collect();
del StoreDataDMCluster;gc.collect();

StoreData1.shape
StoreData1.ix[:1,:]
StoreData2 = pd.merge(data['sto'], StoreData1, left_index=True,right_index=True)
del data['sto'];gc.collect();
del StoreData1;gc.collect();

StoreData2.columns
StoreData3 = StoreData2.drop(['city','state','type','cluster'],axis=1)
del StoreData2;gc.collect();

train = pd.merge(train, StoreData3, how='left', on=['store_nbr'])
test = pd.merge(test, StoreData3, how='left', on=['store_nbr'])
del StoreData3; gc.collect();


train = pd.merge(train, data['hol'], how='left', on=['date', 'store_nbr'])
test = pd.merge(test, data['hol'], how='left', on=['date', 'store_nbr'])
del data['hol']; gc.collect();

train = pd.merge(train, data['oil'], how='left', on=['date'])
test = pd.merge(test, data['oil'], how='left', on=['date'])
del data['oil']; gc.collect();



train = df_transform(train)
test = df_transform(test)
col = [c for c in train if c not in ['id', 'unit_sales','perishable','date']]

x1 = train[(train['yea'] != 2016)]
x2 = train[(train['yea'] == 2016)]
del train; gc.collect;

y1 = x1['unit_sales'].values
y2 = x2['unit_sales'].values
#random forest for sales score:0.8748  mse:0.481

model1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),n_estimators=80, random_state=66)

model1.fit(x1[col],y1)
model1.score(x2[col],y2)
a = NWRMSLE(y2,model1.predict(x2[col]), x2['perishable'])
model1.feature_importances_


model2 = GradientBoostingRegressor(n_estimators=85, max_depth=6, learning_rate = 0.04, 
                                       random_state=None, verbose=0, warm_start=True,
                                       subsample= 0.87, max_features = 0.8) 
model2.fit(x1[col],y1)
print("gradient boost" + model2.score(x2[col],y2))
a = NWRMSLE(y2,model2.predict(x2[col]), x2['perishable'])
print("gradient boost" + a)


model3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),n_estimators=80, random_state=77)

model3.fit(x1[col],y1)
print("ada boost" + model3.score(x2[col],y2))
a = NWRMSLE(y2,model3.predict(x2[col]), x2['perishable'])
print("ada boost" + a)






#gradient boosting #SCORE:0.87 #SCORE:0.57
#bagging for transaction test #SCORE: #NWRMSLE:2.005   #SCORE -0.19
model_transaction_1 = RandomForestRegressor(n_estimators=70,max_depth=6,bootstrap=True,random_state=99,
                                            warm_start=True) 

model_transaction_1.fit(x1[col],y1)
model_transaction_1.score(x2[col],y2)
a = NWRMSLE(y2,model_transaction_1.predict(x2[col]), x2['perishable'])
print(a)

#usemodel1=AdaBoostRegressor(DecisionTreeRegressor(max_depth=0.6),n_estimators=100, random_state=88)
#usemodel1.fit(train[col],train['transactions'])
#test['transactions']=usemodel1.predict(test[col])
#
#col1 = [c for c in train if c not in ['id', 'unit_sales','perishable','date']]
#usemodel2=MLPRegressor(hidden_layer_sizes=(60,2 ),max_iter=30)
#usemodel2.fit(train[col1],train['unit_sales'])
#test['unit_sales']=usemodel2.predict(test[col1])
#cut = 0.+1e-12 # 0.+1e-15
#test['unit_sales'] = (np.exp(test['unit_sales']) - 1).clip(lower=cut)
#output_file = 'prediction2.csv'
#test[['id','unit_sales']].to_csv(output_file, index=False, float_format='%.2f')




#a = NWRMSLE(y2,usemodel.predict(x2[col]), x2['perishable'])

