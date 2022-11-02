#!/usr/bin/env python
# coding: utf-8

# # <center> Credit Card Fraud Detection</center>
# 
# ![image-3.png](attachment:image-3.png)

# # Agenda
# 
# 1. Business challenge
# 
# 2. Data review
# 
# 3. Data processing
# 
# 4. Resampling
# 
# 5. Decision Tree
# 
# 6. Random forest
# 
# 7. Tune the model
# 
# 8. Save the model

# ### Business challenge
# 
# Detecting fraud transactions is of great importance for any credit card company. My task is to detect potential frauds so that customers are not charged for items that they did not purchase. The goal is to build a classifier that tells if a transaction is a fraud or not.

# ### Data review

# The dataset is the Kaggle Credit Card Fraud Detection dataset [here](https://www.kaggle.com/datasets/yashpaloswal/fraud-detection-credit-card?datasetId=2467696).

# In[1]:


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


# In[2]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def print_metrics(X, y, model):
    
    y_pred = model.predict(X)
    print('------------------')
    print('Accuracy score = ',accuracy_score(y, y_pred.round()))
    print('Precision score = ',precision_score(y, y_pred.round()))
    print('Recall score = ',recall_score(y, y_pred.round()))
    print('F1 score = ' ,f1_score(y, y_pred.round()))
    print('ROC-AUC score = ',roc_auc_score(y, y_pred))


# In[3]:


df = pd.read_csv('creditcard.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# <div style="background-color:#a2dce8">This dataset has no missig values, it only contains numerical variables. Principal components (V1-V28) obtained with PCA transformation. The only features which have not been transformed are ‘Time’ and ‘Amount’. ‘Time’ is the seconds elapsed between each transaction and the first. ‘Amount’ is the transaction amount. ‘Class’ is the response variable with 1 as fraud and 0 otherwise.</div>

# In[6]:


df['class'].value_counts()


#  <div style="background-color:#a2dce8">The dataset contains 492 frauds out of 284,807 transactions. Thus, it is highly unbalanced. In such unbalansed data prediction of bigger class will be 99%. So the data need to be resampled</div>

# In[7]:


df.plot(
    kind='kde', 
    subplots=True, 
    layout=(8,4), 
    sharex=False, 
    legend=True,
    fontsize=1, 
    figsize=(16,24)
);


# <div style="background-color:#a2dce8">Conclutions:
#     
# - class 0 = Non Fraudulent
# 
# - class 1 = Fraudulent
# 
# - V's 1 a 28 = confidential data that goes from negative numbers to positive numbers, PCA transformed
#     
# - Amount = Transaction value
# 
# - Time - Transaction time </div>

# In[8]:


df.describe()


# <div style="background-color:#a2dce8">The variable ‘Amount’ ranges from 0 to 25,691.16. Use Standardization so that more then 50% of the values lie in between (-1, 1)</style>

# ### Data processing

# In[9]:


scaler = StandardScaler()
df["NormAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))


# In[10]:


df_del = df.drop(["Amount", "Time"], axis = 1)
y = df_del["class"]
X = df_del.drop(["class"], axis = 1)


# In[11]:


X_train_full, X_test = train_test_split(X, test_size=0.2, random_state=11)
X_train, X_val = train_test_split(X_train_full, test_size=0.25, random_state=11)
y_train_full, y_test = train_test_split(y, test_size=0.2, random_state=11)
y_train, y_val = train_test_split(y_train_full, test_size=0.25, random_state=11)


# In[12]:


decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)


# In[13]:


print('Metrics for test data set: ')
print_metrics(X_test, y_test, decision_tree_model)


# In[14]:


print('Metrics for validation data set: ')
print_metrics(X_val, y_val, decision_tree_model)


# In[15]:


y_pred = decision_tree_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred.round())
plot_confusion_matrix(cm, classes = [0, 1], title = 'Confusion Matrix - Test dataset')


# <div style="background-color:#a2dce8">The problem of unmalanced dataset id very well demonstrated here. The accuracy is hight but in dousn't indicate good performance. Accuracy is the sum of Ture Negative and True Positive divided by total dataset size. If 95% of the dataset is Negative (non-frauds), the network will cleverly predict all to be Negative, leading to 95% accuracy. However, for fraud detection, detecting Positive matters more than detecting Negative.</div>

# In[16]:


rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(X_train, y_train)


# In[17]:


print('Metrics for test data set: ')
print_metrics(X_test, y_test, rf_model)


# In[18]:


print('Metrics for validation data set: ')
print_metrics(X_val, y_val, rf_model)


# In[19]:


y_pred = rf_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred.round())
plot_confusion_matrix(cm, classes = [0, 1], title = 'Confusion Matrix - Test dataset')


# <div style="background-color:#a2dce8">Change the model do not change the situation. We need to resample the data to make the dataset balances. There are more way how to do it but we well you only one of them.</div>

# ## Resampling

# In[20]:


detection_true = df_del[df_del['class'] == 1.0]
detection_false = df_del[df_del['class'] == 0.0]

# Upsample the minority class
detection_true_upsampled = resample(detection_true, random_state=13, n_samples=280000)
detection_upsampled = pd.concat([detection_true_upsampled, detection_false])


# In[21]:


y = detection_upsampled["class"]
X = detection_upsampled.drop(["class"], axis = 1)
X_train_full, X_test = train_test_split(X, test_size=0.2, random_state=11)
X_train, X_val = train_test_split(X_train_full, test_size=0.25, random_state=11)
y_train_full, y_test = train_test_split(y, test_size=0.2, random_state=11)
y_train, y_val = train_test_split(y_train_full, test_size=0.25, random_state=11)


# ## Decision tree

# In[22]:


decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)


# In[23]:


print('Metrics for test data set: ')
print_metrics(X_test, y_test, decision_tree_model)


# In[24]:


print('Metrics for validation data set: ')
print_metrics(X_val, y_val, decision_tree_model)


# In[25]:


y_pred = decision_tree_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred.round())
plot_confusion_matrix(cm, classes = ['Not Fraud', 'Fraud'], title = 'Confusion Matrix - Test dataset')


# ## Random forest

# In[26]:


rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(X_train, y_train)


# In[27]:


print('Metrics for test data set: ')
print_metrics(X_test, y_test, rf_model)


# In[28]:


print('Metrics for val data set: ')
print_metrics(X_val, y_val, rf_model)


# In[29]:


y_pred = rf_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred.round())
plot_confusion_matrix(cm, classes = ['Not Fraud', 'Fraud'], title = 'Confusion Matrix - Test dataset')


# ## Tune the model

# <div style="background-color:#a2dce8">Before the model tuning 4 models were rested: Decision Tree with under-sampling, Random Forest with under-sampling, and Decision Tree, Random Forest with resamplesd data sets. The best performer was Random Forest with balances data set.
# This model will be saved and precessed to the production.</div>
# 
# |                          | Accuracy score | Precision score | Recall score | F1 score | ROC-AUC |
# |--------------------------|----------------|-----------------|--------------|----------|---------|
# | Decision Tree unbalanced | 0.999          | 0.754           | 0.768        | 0.761    | 0.884   |
# | Random Forest unbalanced | 0.999          | 0.921           | 0.759        | 0.832    | 0.879   |
# | Decision tree resample   | 0.9996         | 0.9993          | 1.0          | 0.9996   | 0.9997  |
# | Random Forest resample   | 0.9999         | 0.9998          | 1.0          | 0.9999   | 0.9999  |

# In[30]:


scores = []
for d in [10, 15, 20, 25, 30, 35]:
    rf = RandomForestClassifier(
    n_estimators=20,
    max_depth=d,
    random_state=1,
    n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    score = roc_auc_score(y_val, y_pred)
    scores.append((d, score))
    print(d, score)


# In[31]:


scores = []
for d in [25, 30]:
    for n in range(10, 41, 10):
        rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=d,
        random_state=1,
        n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        score = roc_auc_score(y_val, y_pred)
        scores.append((d, n, score))
        print(d, n, score)


# In[32]:


df_scores = pd.DataFrame(scores, columns=['max_depth', "n_est", 'roc_auc_score'])


# In[33]:


df_scores.head()


# In[34]:


for d in [25, 30]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_est, df_subset.roc_auc_score, label='max_depth=%d' %d)
    
plt.legend()


# In[35]:


d = 25
n = 20


# In[36]:


rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=d,
        random_state=1,
        n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred.round())
plot_confusion_matrix(cm, classes = ['Not Fraud', 'Fraud'], title = 'Confusion Matrix - Test dataset')


# ## Save the model

# In[58]:


# Creating dictionary vectorizer

from sklearn.feature_extraction import DictVectorizer
dv =  DictVectorizer(sparse=False)

train_dicts = X_train.reset_index(drop=True).to_dict(orient='records')


# In[59]:


import pickle


# In[60]:


output_file = f'model_{d}_{n}'


# In[67]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,rf), f_out)


# In[62]:


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

print(card)


# In[66]:


X = dv.fit_transform([card])

print(rf.predict(X)[0])

