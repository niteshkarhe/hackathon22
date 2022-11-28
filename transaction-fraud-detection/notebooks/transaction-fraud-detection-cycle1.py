#!/usr/bin/env python
# coding: utf-8

# # i. Business Understanding

# ## i.i Blocker Fraud Company

# * The Blocker Fraud Company is a company specialized in detecting fraud in financial transactions made through mobile devices. The company has a service called “Blocker Fraud” with no guarantee of blocking fraudulent transactions.
# 
# * And the business model of the company is of the Service type with the monetization made by the performance of the service provided, that is, the user pays a fixed fee on the success in detecting fraud in the customer's transactions.

# ### i.i.i Expansion Problem

# Blocker Fraud Company is expanding in Brazil and to acquire customers more quickly, it has adopted a very aggressive strategy. The strategy works as follows:
# 1. The company will receive 25% of the value of each transaction that is truly detected as fraud.
# 1. The company will receive 5% of the value of each transaction detected as fraud, but the transaction is truly legitimate.
# 1. The company will return 100% of the value to the customer, for each transaction detected as legitimate, however the transaction is truly a fraud.

# ## i.ii The Challenge

# You need to deliver to the CEO of Blocker Fraud Company a production model in which your access will be done via API, that is, customers will send their transactions via API so that your model classifies them as fraudulent or legitimate.

# ### i.ii.i Business Questions

# 1. What is the model's Precision and Accuracy?
# 1. How Reliable is the model in classifying transactions as legitimate or fraudulent?
# 1. What is the Expected Billing by the Company if we classify 100% of transactions with the model?
# 1. What is the Loss Expected by the Company in case of model failure?
# 1. What is the Profit Expected by the Blocker Fraud Company when using the model?

# # 0.0 Imports and Helper Functions

# ## 0.1 Imports

# In[229]:


import joblib
import warnings
import inflection

import numpy             as np
import pandas            as pd
import seaborn           as sns

import matplotlib.pyplot as plt

from scipy   import stats
from boruta  import BorutaPy
from category_encoders import OneHotEncoder

from IPython.display      import Image
from IPython.core.display import HTML 

from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.svm          import SVC
from sklearn.dummy        import DummyClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics         import balanced_accuracy_score, precision_score, classification_report
from sklearn.metrics         import recall_score, f1_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold


# ## 0.2 Helper Functions

# In[230]:


warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)


# In[231]:


# def jupyter_settings():
#     get_ipython().run_line_magic('matplotlib', 'inline')
#     get_ipython().run_line_magic('pylab', 'inline')
#     
#     sns.set(font_scale=1.6)
#     
#     plt.style.use('seaborn-darkgrid')
#     plt.rcParams['figure.figsize'] = [25, 12]
#     plt.rcParams['font.size'] = 16
#     
#     display( HTML('<style>.container { width:100% !important; }</style>'))
#     pd.options.display.max_columns = None
#     pd.options.display.max_rows = None
#     pd.set_option('display.expand_frame_repr', False)
#     
# jupyter_settings()


# In[232]:


def ml_scores(model_name, y_true, y_pred):
    
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return pd.DataFrame({'Balanced Accuracy': np.round(accuracy, 3), 
                         'Precision': np.round(precision, 3), 
                         'Recall': np.round(recall, 3),
                         'F1': np.round(f1, 3),
                         'Kappa': np.round(kappa, 3)}, 
                        index=[model_name])


# In[233]:


def calcCramerV(x, y):
    cm = pd.crosstab(x, y).values
    n = cm.sum()
    r, k = cm.shape
    
    chi2 = stats.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k-1)*(r-1)/(n-1))
    
    kcorr = k - (k-1)**2/(n-1)
    rcorr = r - (r-1)**2/(n-1)
    
    return np.sqrt((chi2corr/n) / (min(kcorr-1, rcorr-1)))


# In[234]:


def ml_cv_results(model_name, model, x, y, verbose=1):
    
    '''initial'''
    balanced_accuracies = []
    precisions = []
    recalls = []
    f1s = []
    kappas = []
    
    mm = MinMaxScaler()
    
    x_ = x.to_numpy()
    y_ = y.to_numpy()
    
    count = 0
    
    '''cross-validation'''
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    
    for index_train, index_test in skf.split(x_, y_):
        ## Showing the Fold
        if verbose > 0:
            count += 1
            print('Fold K=%i' % (count))
    
        ## selecting train and test
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y.iloc[index_train], y.iloc[index_test]
        
        ## applying the scale
        x_train = mm.fit_transform(x_train)
        x_test = mm.transform(x_test)
    
        ## training the model
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        ## saving the metrics
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        kappas.append(cohen_kappa_score(y_test, y_pred))
        
        
    '''results'''    
    accuracy_mean, accuracy_std = np.round(np.mean(balanced_accuracies), 3), np.round(np.std(balanced_accuracies), 3)
    precision_mean, precision_std = np.round(np.mean(precisions), 3), np.round(np.std(precisions), 3)
    recall_mean, recall_std = np.round(np.mean(recalls), 3), np.round(np.std(recalls), 3)
    f1_mean, f1_std = np.round(np.mean(f1s), 3), np.round(np.std(f1s), 3)
    kappa_mean, kappa_std = np.round(np.mean(kappas), 3), np.round(np.std(kappas), 3)
    
    ## saving the results in a dataframe
    return pd.DataFrame({"Balanced Accuracy": "{} +/- {}".format(accuracy_mean, accuracy_std),
                        "Precision": "{} +/- {}".format(precision_mean, precision_std),
                        "Recall": "{} +/- {}".format(recall_mean, recall_std),
                        "F1": "{} +/- {}".format(f1_mean, f1_std),
                        "Kappa": "{} +/- {}".format(kappa_mean, kappa_std)},
                       index=[model_name])


# # 1.0 Data Description

# ## 1.1 Loading Data

# In[235]:


df1 = pd.read_csv('../data/raw/fraud_0.1origbase.csv')


# In[236]:


df1.head()


# In[237]:


df1.tail()


# ## 1.2 Columns

# ### 1.2.1 Column Descriptions

# **step:** maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
# 
# **type:** CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
# 
# **amount:** amount of the transaction in local currency.
# 
# **nameOrig:** customer who started the transaction
# 
# **oldbalanceOrg:** initial balance before the transaction
# 
# **newbalanceOrig:** new balance after the transaction
# 
# **nameDest:** customer who is the recipient of the transaction
# 
# **oldbalanceDest:** initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
# 
# **newbalanceDest:** new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
# 
# **isFraud:** This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
# 
# **isFlaggedFraud:** The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

# ### 1.2.2 Column Rename

# In[238]:


cols_old = df1.columns.tolist()

snakecase = lambda x: inflection.underscore(x)
cols_new = list(map(snakecase, cols_old))

df1.columns = cols_new


# In[239]:


df1.columns


# ## 1.3 Data Dimension

# In[240]:


print('Number of Rows: {}'.format(df1.shape[0]))
print('Number of Cols: {}'.format(df1.shape[1]))


# ## 1.4 Data Types and Structure

# In[241]:


df1.info()


# ## 1.5 Check NA

# In[242]:


df1.isna().mean()


# ## 1.6 Fill Out NA

# There's no NaN values to fill.

# ## 1.7 Change Data Type

#  I will change the values 0 and 1 to 'yes' and 'no'. It'll help on the data description and analysis sections.

# In[243]:


df1['is_fraud'] = df1['is_fraud'].map({1: 'yes', 0: 'no'})
df1['is_flagged_fraud'] = df1['is_flagged_fraud'].map({1: 'yes', 0: 'no'})


# ## 1.8 Description Statistics

# In[244]:


num_attributes = df1.select_dtypes(exclude='object')
cat_attributes = df1.select_dtypes(include='object')


# ### 1.8.1 Numerical Attributes

# In[245]:


describe = num_attributes.describe().T

describe['range'] = (num_attributes.max() - num_attributes.min()).tolist()
describe['variation coefficient'] = (num_attributes.std() / num_attributes.mean()).tolist()
describe['skew'] = num_attributes.skew().tolist()
describe['kurtosis'] = num_attributes.kurtosis().tolist()

describe


# * All the data has a coeficient of variation greater than 25%, therefore they aren't homogeneous.
# 
# * The step variable starts from 1 hour to 742 hour (30 days).
# 
# * Some variables are higher shap and right skewed.
# 
# * 50% of the newbalance_orig is 0. Maybe there are some transfers that don't go to the destination.
# 
# * The skew is higher positive, therefore the values may be in less values.

# ### 1.8.2 Categorical Attributes

# In[246]:


cat_attributes.describe()


# * The majority type is cash_out with 2237500.
# 
# * There's a lot of variability in name_orig, so it could be hard to use one hot encoding.
# 
# * There's less name_orig than name_dest. There's more users sending than receiving, however use one hot encoding will not help.
# 
# * There's more fraud than the flagged fraud, it shows that the current method can't recognize fraud efficiently.

# # 2.0 Feature Engineering

# In[247]:


df2 = df1.copy()


# ## 2.1 Mind Map

# In[248]:


Image('../images/Fraud-mindmap.png')


# ## 2.2 Hypothesis Creation

# ### 2.2.1 User

# * 90% of the twentyone-year-old users did a fraud transiction.
# 
# * The majority fraud transiction occours for the same initial letter user.
# 
# * The fraud amount is greater than 10.000.
# 
# * The 60% of the age is greater than 30 year old.

# ### 2.2.2 Type

# * 60% of fraud transaction occours using cash-out-type method.
# 
# * The majority transfers occours using tranfers-type method.
# 
# * Values greater than 100.000 occours using transfers-type method.
# 
# * Payment type occurs with values lower than 100.000

# ### 2.2.3 Origin and Destiny Transactions

# * 60% of the difference between origin destiny transactions is equal 0 for frauds.
# 
# * Origin values are greater than destiny values for fraud transaction.

# ### 2.2.4 Time

# * Fraud transactions occours at least in 3 days.
# 
# * 40% of the cash-out transactions occours less than 1 day.
# 
# * 60% of the transaction less than 100.000 occours at least 10 days.
# 
# * The transactions greater than 10.000 occours at most in 2 weeks. 

# ## 2.3 Hipothesys List

# 1. The majority fraud transiction occours for the same initial letter user.
# 
# 1. All the fraud amount is greater than 10.000.
# 
# 1. 60% of fraud transaction occours using cash-out-type method.
# 
# 1. The majority transfers occours using tranfers-type method.
# 
# 1. Fraud transactions occours at least in 3 days.

# ## 2.4 Feature Engineering

# In[249]:


# step
df2['step_days'] = df2['step'].apply(lambda i: i/24)
df2['step_weeks'] = df2['step'].apply(lambda i: i/(24*7))

# difference between initial balance before the transaction and new balance after the transaction
df2['diff_new_old_balance'] = df2['newbalance_orig'] - df2['oldbalance_org']

# difference between initial balance recipient before the transaction and new balance recipient after the transaction.
df2['diff_new_old_destiny'] = df2['newbalance_dest'] - df2['oldbalance_dest']

# name orig and name dest
df2['name_orig'] = df2['name_orig'].apply(lambda i: i[0])
df2['name_dest'] = df2['name_dest'].apply(lambda i: i[0])


# # 3.0 Selecting Columns

# In[250]:


df3 = df2.copy()


# ## 3.1 Selecting Columns

# I'll use all the columns for data analysis

# ## 3.2 Selecting Lines

# I'll use all the lines.

# # 4.0 Exploratory Data Analisys

# In[251]:


df4 = df3.copy()


# ## 4.1 Univariate Analysis

# ### 4.1.1 Response Variable

# In[252]:


ax = sns.countplot(y='is_fraud', data=df4);

total = df4['is_fraud'].size
for p in ax.patches:
        percentage = ' {:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))


# ### 4.1.2 Numerical Variables

# In[253]:


num_attributes = df4.select_dtypes(exclude='object')
columns = num_attributes.columns.tolist()
j = 1

for column in columns:
    plt.subplot(2, 5, j)
    sns.distplot(num_attributes[column]);
    
    j += 1


# ### 4.1.3 Categorical Variables

# In[254]:


cat_attributes = df4.select_dtypes(include='object')
columns = cat_attributes.columns.tolist()
j = 1

for column in columns:
    plt.subplot(3, 2, j)
    ax = sns.countplot(y=column, data=cat_attributes)
    
    total = cat_attributes[column].size
    for p in ax.patches:
        percentage = ' {:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
    
    j += 1


# ## 4.2 Bivariate Analysis

# ### H1 The majority fraud transiction occours for the same user.
# **TRUE:** The same user origem and destiny has got the same inital letter.

# In[255]:


aux1 = df4[df4['is_fraud'] == 'yes']
sns.countplot(y='name_orig', data=aux1);


# In[256]:


sns.countplot(y='name_dest', data=aux1);


# ### H2 All the fraud amount is greater than 10.000.
# 
# **TRUE:** The values are greater than 10.000. But it's important to note that the no-fraud values is greater than 100.000 also.

# In[257]:


sns.barplot(y='amount', x='is_fraud', data=df4);


# ### H3 60% of fraud transaction occours using cash-out-type method.
# 
# **FALSE:** The fraud transaction occours in transfer and cash-out type. However they're almost the same value.

# In[258]:


aux1 = df4[df4['is_fraud'] == 'yes']
ax = sns.countplot(y='type', data=aux1)

total = aux1['type'].size
for p in ax.patches:
        percentage = ' {:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))


# To see the complete transiction-type and I'll plot them here.

# In[259]:


ax = sns.countplot(y='type', hue='is_fraud', data=df4)

total = df4['type'].size
for p in ax.patches:
        percentage = ' {:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))


# ### H4 Values greater than 100.000 occours using transfers-type method.
# 
# **FALSE:** The majority transactions occours in trasnfer-type, however transactions greater than 100.000 occour in cash-out and cash-in too.

# In[260]:


ax = sns.barplot(y='type', x='amount', data=df4);

total = df4['type'].size
for p in ax.patches:
        percentage = ' {:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))


# ### H5 Fraud transactions occours at least in 3 days.
# 
# **TRUE:** The values for transactions and days in fraud aren't similar.

# In[261]:


aux1 = df4[df4['is_fraud'] == 'yes']
sns.regplot(x='step_days', y='amount', data=aux1);


# ## 4.3 Multivariaty Analysis

# ### 4.3.1 Numerical Analysis

# In[262]:


corr = num_attributes.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, mask=mask, vmin=-1, center=0, vmax=1, square=True)


# ### 4.3.2 Categorical Variables

# In[263]:


dict_corr = {}
columns = cat_attributes.columns.tolist()

for column in columns:
    dict_corr[column] = {}
    
    for column2 in columns:
        dict_corr[column][column2] = calcCramerV(cat_attributes[column], cat_attributes[column2])
        
corr = pd.DataFrame(dict_corr)


# In[264]:


mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, mask=mask, vmin=0, vmax=1, square=True)


# # 5.0 Data Preparation

# In[347]:


df5 = df4.copy()


# ## 5.1 Spliting into Train, Valid and Test

# In[348]:


X = df5.drop(columns=['is_fraud', 'is_flagged_fraud', 'name_orig', 'name_dest', 
                      'step_weeks', 'step_days'], axis=1)
y = df5['is_fraud'].map({'yes': 1, 'no': 0})


# In[349]:


# spliting into temp and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=.2, stratify=y)


# In[350]:


# spliting into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=.2, stratify=y_temp)


# ## 5.2 One Hot Encoder

# In[351]:


ohe = OneHotEncoder(cols=['type'], use_cat_names=True)

X_train = ohe.fit_transform(X_train)
X_valid = ohe.transform(X_valid)

X_temp = ohe.fit_transform(X_temp)
X_test = ohe.transform(X_test)


# ## 5.3 Rescaling

# In[271]:


num_columns = ['amount', 'oldbalance_org', 'newbalance_orig', 'oldbalance_dest', 'newbalance_dest',
               'diff_new_old_balance', 'diff_new_old_destiny']
mm = MinMaxScaler()
X_params = X_temp.copy()

X_train[num_columns] = mm.fit_transform(X_train[num_columns])
X_valid[num_columns] = mm.transform(X_valid[num_columns])

X_params[num_columns] = mm.fit_transform(X_temp[num_columns])
X_test[num_columns] = mm.transform(X_test[num_columns])


# # 6.0 Feature Selection

# ## 6.1 Boruta

# In[272]:


# X_boruta = X_params.values
# y_boruta = y_temp.values.ravel()


# In[273]:


# boruta = BorutaPy(RandomForestClassifier(), n_estimators='auto')
# boruta.fit(X_boruta, y_boruta)


# ### 6.1.1 Best Features

# In[274]:


# cols_selected_boruta = boruta.support_.tolist()


# In[275]:


# columns_selected = X_params.loc[:, cols_selected_boruta].columns.tolist()


# In[276]:


# columns_selected


# In[277]:


# ['step',
#  'amount',
#  'oldbalance_org',
#  'newbalance_orig',
#  'oldbalance_dest',
#  'newbalance_dest',
#  'diff_new_old_balance',
#  'diff_new_old_destiny',
#  'type_TRANSFER']


# In[278]:


final_columns_selected = ['step', 'oldbalance_org', 
                          'newbalance_orig', 'newbalance_dest', 
                          'diff_new_old_balance', 'diff_new_old_destiny', 
                          'type_TRANSFER']


# # 7.0 Machine Learning Modeling

# In[279]:


X_train_cs = X_train[final_columns_selected]
X_valid_cs = X_valid[final_columns_selected]

X_temp_cs = X_temp[final_columns_selected]
X_test_cs = X_test[final_columns_selected]

X_params_cs = X_params[final_columns_selected]


# ## 7.1 Baseline

# In[280]:


dummy = DummyClassifier()
dummy.fit(X_train_cs, y_train)

y_pred = dummy.predict(X_valid_cs)


# In[281]:


dummy_results = ml_scores('dummy', y_valid, y_pred)
dummy_results


# ### 7.1.1 Classification Report

# In[282]:


print(classification_report(y_valid, y_pred))


# ### 7.1.2 Cross Validation

# In[283]:


dummy_cv = ml_cv_results('Dummy', DummyClassifier(), X_temp, y_temp)
dummy_cv


# ## 7.2 Logistic Regression

# In[284]:


lg = LogisticRegression()
lg.fit(X_train_cs, y_train)

y_pred = lg.predict(X_valid_cs)


# In[285]:


lg_results = ml_scores('Logistic Regression', y_valid, y_pred)
lg_results


# ### 7.2.1 Classification Report

# In[286]:


print(classification_report(y_valid, y_pred))


# ### 7.2.2 Cross Validation

# In[287]:


lg_cv = ml_cv_results('Logistic Regression',
                      LogisticRegression(), 
                      X_temp_cs, y_temp)
lg_cv


# ## 7.3 K Nearest Neighbors

# In[288]:


knn = KNeighborsClassifier()
knn.fit(X_train_cs, y_train)

y_pred = knn.predict(X_valid_cs)


# In[289]:


knn_results = ml_scores('K Nearest Neighbors', y_valid, y_pred)
knn_results


# ### 7.3.1 Classification Report

# In[290]:


print(classification_report(y_valid, y_pred))


# ### 7.3.2 Cross Validation

# In[291]:


knn_cv = ml_cv_results('K Nearest Neighbors', KNeighborsClassifier(), 
                       X_temp_cs, y_temp)
knn_cv


# ## 7.4 Support Vector Machine

# In[292]:


svm = SVC()
svm.fit(X_train_cs, y_train)

y_pred = svm.predict(X_valid_cs)


# In[293]:


svm_results = ml_scores('SVM', y_valid, y_pred)
svm_results


# ### 7.4.1 Classification Report

# In[294]:


print(classification_report(y_valid, y_pred))


# ### 7.4.2 Cross Validation

# In[295]:


svm_cv = ml_cv_results('SVM', SVC(), X_temp_cs, y_temp)
svm_cv


# ## 7.5 Random Forest

# In[296]:


rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train_cs, y_train)

y_pred = rf.predict(X_valid_cs)


# In[297]:


rf_results = ml_scores('Random Forest', y_valid, y_pred)
rf_results


# ### 7.5.1 Classification Report

# In[298]:


print(classification_report(y_valid, y_pred))


# ### 7.5.2 Cross Validation

# In[299]:


rf_cv = ml_cv_results('Random Forest', 
                      RandomForestClassifier(), 
                      X_temp_cs, y_temp)
rf_cv


# ## 7.6 XGBoost

# In[301]:


xgb = XGBClassifier()
xgb.fit(X_train_cs, y_train)

y_pred = xgb.predict(X_valid_cs)


# In[302]:


xgb_results = ml_scores('XGBoost', y_valid, y_pred)
xgb_results


# ### 7.6.1 Classification Report

# In[303]:


print(classification_report(y_valid, y_pred))


# ### 7.6.2 Cross Validation

# In[305]:


xgb_cv = ml_cv_results('XGBoost', XGBClassifier(),
                       X_temp_cs, y_temp)
xgb_cv


# ## 7.7 LightGBM

# In[306]:


lightgbm = LGBMClassifier()
lightgbm.fit(X_train_cs, y_train)

y_pred = lightgbm.predict(X_valid_cs)


# In[307]:


lightgbm_results = ml_scores('LightGBM', y_valid, y_pred)
lightgbm_results


# ### 7.7.1 Classification Report

# In[308]:


print(classification_report(y_valid, y_pred))


# ### 7.7.2 Cross Validation

# In[309]:


lightgbm_cv = ml_cv_results('LightGDM', LGBMClassifier(), 
                            X_temp_cs, y_temp)
lightgbm_cv


# ## 7.8 Comparing Model's Performance

# ### 7.8.1 Single Performance

# In[310]:


modeling_performance = pd.concat([dummy_results, lg_results, knn_results, 
                                  rf_results, xgb_results, lightgbm_results,
                                 svm_results])
modeling_performance.sort_values(by="F1", ascending=True)


# ### 7.8.2 Cross Validation Performance

# In[311]:


modeling_performance_cv = pd.concat([dummy_cv, lg_cv, knn_cv, rf_cv, 
                                     xgb_cv, lightgbm_cv, svm_cv])

modeling_performance_cv.sort_values(by="F1", ascending=True)


# # 8.0 Hyperparameter Fine Tuning

# In[312]:


f1 = make_scorer(f1_score)


# In[313]:


params = {
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eta': [0.3, 0.1, 0.01],
    'scale_pos_weight': [1, 774, 508, 99]
}


# In[315]:


gs = GridSearchCV(XGBClassifier(), 
                  param_grid=params, 
                  scoring=f1, 
                  cv=StratifiedKFold(n_splits=5))

gs.fit(X_params_cs, y_temp)


# In[316]:


best_params = gs.best_params_
best_params


# In[317]:


best_params = {'booster': 'gbtree', 'eta': 0.3, 'scale_pos_weight': 1}


# In[318]:


gs.best_score_


# ## 8.1 Results

# In[319]:


xgb_gs = XGBClassifier(
    booster=best_params['booster'],
    eta=best_params['eta'],
    scale_pos_weight=best_params['scale_pos_weight']
)


# In[320]:


xgb_gs.fit(X_train_cs, y_train)


# In[321]:


y_pred = xgb_gs.predict(X_valid_cs)


# ### 8.1.2 Single Results

# In[322]:


xgb_gs_results = ml_scores('XGBoost GS', y_valid, y_pred)
xgb_gs_results


# ### 8.1.3 Cross Validation

# In[323]:


xgb_gs_cv = ml_cv_results('XGBoost GS', xgb_gs, X_temp_cs, y_temp)
xgb_gs_cv


# # 9.0 Conclusions 

# ## 9.1 Final Model

# In[324]:


final_model = XGBClassifier(
    booster=best_params['booster'],
    eta=best_params['eta'],
    scale_pos_weight=best_params['scale_pos_weight']
)

final_model.fit(X_params_cs, y_temp)


# ### 9.1.1 Unseen Data Score

# In[325]:


y_pred = final_model.predict(X_test_cs)


# In[326]:


unseen_scores = ml_scores('unseen', y_test, y_pred)
unseen_scores


# ## 9.2 Blocker Fraud Company Expasion

# ### 9.2.1 The company receives 25% of each transaction value truly detected as fraud.

# In[327]:


df_test = df5.loc[X_test.index, :]
df_test['predictions'] = y_pred


# In[328]:


aux1 = df_test[(df_test['is_fraud'] == 'yes') & (df_test['predictions'] == 1)]
receives = aux1['amount'].sum() * 0.25


# In[329]:


print('The company can receive %.2f detecting fraud transactions.' % (receives))


# ### 9.2.2 The company receives 5% of each transaction value detected as fraud, however the transaction is legitimate.

# In[330]:


aux1 = df_test[(df_test['is_fraud'] == 'no') & (df_test['predictions'] == 1)]
receives = aux1['amount'].sum() * 0.05

print('For wrong decisions, the company can receive %.2f.' % (receives))


# ### 9.2.3 The company gives back 100% of the value for the customer in each transaction detected as legitimate, however the transaction is actually a fraud.

# In[331]:


aux1 = df_test[(df_test['is_fraud'] == 'yes') & (df_test['predictions'] == 0)]
receives = aux1['amount'].sum()

print('However, the company must return the amount of %.2f.' % (receives))


# ## 9.3 Model's Performance

# ### 9.3.1 What is the model's Precision and Accuracy?

# In[332]:


print('For unseen data, the values of balanced accuracy is equal %.2f and precision is equal %.2f.' % (unseen_scores['Balanced Accuracy'], unseen_scores['Precision']))


# ### 9.3.2 How reliable is the model in classifying transactions as legitimate or fraudulent?

# In[340]:


print('The model can detect 0.851 +/- 0.023 of the fraud. However it detected 0.84 of the frauds from a unseen data.')


# ### 9.3.3 What is the revenue expected by the company  classify 100% of transactions with the model?

# In[334]:


aux1 = df_test[(df_test['is_fraud'] == 'yes') & (df_test['predictions'] == 1)]
receives = aux1['amount'].sum() * 0.25

aux2 = df_test[(df_test['is_fraud'] == 'no') & (df_test['predictions'] == 1)]
receives2 = aux2['amount'].sum() * 0.05

print('Using the model the company can revenue %.2f.' % (receives + receives2))


# In[335]:


aux3 = df_test[(df_test['is_fraud'] == 'yes') & (df_test['is_flagged_fraud'] == 'yes')]
curr_receives = aux3['amount'].sum() * 0.25

aux4 = df_test[(df_test['is_fraud'] == 'no') & (df_test['is_flagged_fraud'] == 'yes')]
curr_receives2 = aux4['amount'].sum() * 0.05

print('However the currently method the revenue is %.2f.' % (curr_receives + curr_receives2))


# ### 9.3.4 What is the loss expected by the Company if it classifies 100% of the transactions with the model?

# In[336]:


aux1 = df_test[(df_test['is_fraud'] == 'yes') & (df_test['predictions'] == 0)]
loss = aux1['amount'].sum()

print('For wrong classifications the company must return the amount of %.2f.' % (loss))


# In[337]:


aux1 = df_test[(df_test['is_fraud'] == 'yes') & (df_test['is_flagged_fraud'] == 'no')]
curr_loss = aux1['amount'].sum()

print('For wrong classifications using the currently method, the company must return the amount of %.2f.' % (curr_loss))


# ### 9.3.5 What is the profit expected by the blocker fraud company when using the model?

# In[338]:


print('The company can expect the profit of %.2f.' % (receives + receives2 - loss))


# In[339]:


print('Using the currently method, the profit is %.2f.' % (curr_receives + curr_receives - curr_loss))


# # 10.0 Model Deploy

# ## 10.1 Saving

# In[341]:


final_model = XGBClassifier(
    booster=best_params['booster'],
    eta=best_params['eta'],
    scale_pos_weight=best_params['scale_pos_weight']
)

final_model.fit(X_params_cs, y_temp)

joblib.dump(final_model, '../models/model_cycle1.joblib')


# In[342]:


mm = MinMaxScaler()
mm.fit(X_params_cs, y_temp)

joblib.dump(mm, '../parameters/minmaxscaler_cycle1.joblib')


# In[352]:


joblib.dump(ohe, '../parameters/onehotencoder_cycle1.joblib')


# ## 10.2 Fraud Class

# In[353]:


import joblib
import inflection
import pandas as pd

class Fraud:
    
    def __init__(self):
        self.minmaxscaler = joblib.load('../parameters/minmaxscaler_cycle1.joblib')
        self.onehotencoder = joblib.load('../parameters/onehotencoder_cycle1.joblib')
        
    def data_cleaning(self, df1):
        cols_old = df1.columns.tolist()
        
        snakecase = lambda i: inflection.underscore(i)
        cols_new = list(map(snakecase, cols_old))
        
        df1.columns = cols_new
        
        return df1
    
    def feature_engineering(self, df2):
        # step
        df2['step_days'] = df2['step'].apply(lambda i: i/24)
        df2['step_weeks'] = df2['step'].apply(lambda i: i/(24*7))

        # difference between initial balance before the transaction and new balance after the transaction
        df2['diff_new_old_balance'] = df2['newbalance_orig'] - df2['oldbalance_org']

        # difference between initial balance recipient before the transaction and new balance recipient after the transaction.
        df2['diff_new_old_destiny'] = df2['newbalance_dest'] - df2['oldbalance_dest']

        # name orig and name dest
        df2['name_orig'] = df2['name_orig'].apply(lambda i: i[0])
        df2['name_dest'] = df2['name_dest'].apply(lambda i: i[0])
        
        return df2.drop(columns=['name_orig', 'name_dest', 'step_weeks', 'step_days'], axis=1)
    
    def data_preparation(self, df3):
        # One Hot Encoding
        df3 = pd.get_dummies(df3)
        
        # Rescaling 
        num_columns = ['amount', 'oldbalance_org', 'newbalance_orig', 'oldbalance_dest', 
                       'newbalance_dest', 'diff_new_old_balance', 'diff_new_old_destiny']
        df3[num_columns] = self.minmaxscaler.transform(df3[num_columns])
        
        # OneHotEncoder
        df3 = self.onehotencoder.transform(df3)
        
        # selected columns
        final_columns_selected = ['step', 'oldbalance_org', 'newbalance_orig', 'newbalance_dest', 
                                  'diff_new_old_balance', 'diff_new_old_destiny', 'type_TRANSFER']
        return df3[final_columns_selected]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        original_data['prediction'] = pred
        
        return original_data.to_json(orient="records", date_format="iso")


# ## 10.3 API Handler

# In[354]:


import joblib
import pandas as pd
from churn.Churn import Churn
from flask import Flask, request, Response

# loading model
model = joblib.load('../models/model_cycle1.joblib')

# initialize API
app = Flask(__name__)

@app.route('/fraud/predict', methods=['POST'])
def churn_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Rossmann class
        pipeline = Fraud()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
        
    else:
        return Reponse('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')


# ## 10.4 API Tester

# In[166]:


# loading test dataset
df10 = pd.read_csv("../database/fraud_0.1origbase.csv").iloc[5, :-2]


# In[167]:


df10.head()


# In[168]:


# convert Dataframe to json
data = df10.to_json()


# In[169]:


data


# In[170]:


import requests


# In[224]:


url = 'http://0.0.0.0:5000/fraud/predict'
header = {'Content-type': 'application/json'}

r = requests.post(url=url, data=data, headers=header)


# In[225]:


r.status_code


# In[226]:


r.json()


# In[227]:


d = pd.DataFrame(r.json(), columns=r.json()[0].keys())


# In[228]:


d


# In[ ]:




