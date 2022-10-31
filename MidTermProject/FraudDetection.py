#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import itertools
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

import sklearn
from sklearn import *

from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings("ignore")

def print_metrics(X, y, model):
    
    y_pred = model.predict(X)
    print('------------------')
    print('Accuracy score = ',accuracy_score(y, y_pred.round()))
    print('Precision score = ',precision_score(y, y_pred.round()))
    print('Recall score = ',recall_score(y, y_pred.round()))
    print('F1 score = ' ,f1_score(y, y_pred.round()))
    print('ROC-AUC score = ',roc_auc_score(y, y_pred))

# paramiters 
d = 25
n = 20
# output_file = f"model_{d}_{n}"
output_file = f"model_{d}_{n}"

print(f"Doing validation with depth = {d} and estimator = {n}")

df = pd.read_csv('creditcard.csv')


scaler = StandardScaler()
df["NormAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

df_del = df.drop(["Amount", "Time"], axis = 1)
y = df_del["class"]
X = df_del.drop(["class"], axis = 1)

detection_true = df_del[df_del['class'] == 1.0]
detection_false = df_del[df_del['class'] == 0.0]

# Upsample the minority class
detection_true_upsampled = resample(detection_true, random_state=13, n_samples=280000)
detection_upsampled = pd.concat([detection_true_upsampled, detection_false])

y = detection_upsampled["class"]
X = detection_upsampled.drop(["class"], axis = 1)
X_train_full, X_test = train_test_split(X, test_size=0.2, random_state=11)
X_train, X_val = train_test_split(X_train_full, test_size=0.25, random_state=11)
y_train_full, y_test = train_test_split(y, test_size=0.2, random_state=11)
y_train, y_val = train_test_split(y_train_full, test_size=0.25, random_state=11)

rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=d,
        random_state=1,
        n_jobs=-1)
rf.fit(X_train, y_train)

print("Metrics on validation data set are: ")
print_metrics(X_val, y_val, rf)


from sklearn.feature_extraction import DictVectorizer
dv =  DictVectorizer(sparse=False)

train_dicts = X_train.reset_index(drop=True).to_dict(orient='records')



with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)
    
card = {'V1': -1.377245329,
 'V2': 0.714822534,
 'V3': 2.507512766,
 'V4': 0.865082434,
 'V5': -0.290489181,
 'V6': 1.077327555,
 'V7': 0.032507083,
 'V8': 0.510945608,
 'V9': 0.717788301,
 'V10': -0.256417813,
 'V11': 0.316423719,
 'V12': 0.759520508,
 'V13': -1.439420629,
 'V14': -0.592772797,
 'V15': -2.397140825,
 'V16': -1.072538941,
 'V17': 0.615996407,
 'V18': -0.58145074,
 'V19': 0.714980257,
 'V20': -0.125335333,
 'V21': -0.341852731,
 'V22': -0.60673121,
 'V23': -0.0997405,
 'V24': -0.009123052,
 'V25': 0.32837917,
 'V26': -0.506683335,
 'V27': -0.032234738,
 'V28': 0.139840723,
 'NormAmount': -0.299135286762843}
print("Card: ", card)

X = dv.fit_transform([card])

print("Prediction: ", rf.predict(X)[0])
