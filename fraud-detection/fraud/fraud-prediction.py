#!/usr/bin/env python
# coding: utf-8

# # Fraud transaction analysis and prediction model.
# _____

# *About Dataset
# Context*
# 
# *Develop a model for predicting fraudulent transactions for a financial company and use insights from the model to develop an actionable plan. Data for the case is available in CSV format having 6362620 rows and 10 columns.*
# 
# *Data Dictionary:*
# 
# *step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).*
# 
# *type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.*
# 
# *amount - amount of the transaction in local currency.*
# 
# *nameOrig - customer who started the transaction*
# 
# *oldbalanceOrg - initial balance before the transaction*
# 
# *newbalanceOrig - new balance after the transaction*
# 
# *nameDest - customer who is the recipient of the transaction*
# 
# *oldbalanceDest - initial balance recipient before the transaction. Note that there is no information for customers that start with M (Merchants).*
# 
# *newbalanceDest - new balance recipient after the transaction. Note that there is no information for customers that start with M (Merchants).*
# 
# *isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.*
# 
# *isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.
# _____

# # Data Cleansing

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt
import math
import statsmodels.api as sm

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sb.set_style('darkgrid')
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.show(block=True)


# In[2]:


df2 = pd.read_csv('../input/Fraud.csv')
df2.head()


# In[3]:


df2.info()


# Check for NA values

# In[4]:


df2.isna().sum()


# In[5]:


df2.describe()


# ____
# Let's create a new column indicating type of receiving and original account, 'C' stands for customer and 'M' for Merchant account.

# In[6]:


df2['typeOrigAccount'] = df2['nameOrig'].str[0]
df2['typeDestAccount'] = df2['nameDest'].str[0]
df2.head()


# In[7]:


df2.type.unique()


# ____
# Let's check if there are merchant origin accounts in the dataframe:

# In[8]:


df2.typeOrigAccount.unique()


# > Only customer origin accounts are present in the dataframe.

# Let's check if there are merchant origin accounts in the dataframe:

# In[9]:


df2.typeDestAccount.unique()


# > Both customer and Merchant destination accounts are present in the dataframe.
# ___
# ___
# ### Data Wrangling
#  Copy of the dataset to work with

# In[10]:


df_c= df2.copy()


# In[11]:


df_c.head()


# It's possible to get rid off the columns that do not bring any information to our dataset analysis. Columns 'typeOrigAccount' , 'nameOrig' can be removed.

# In[12]:


df_c = df_c.drop(['typeOrigAccount','nameOrig'], axis =1)
df_c.head()


# ____

# In[13]:


fraud_ratio1 = round(100 * df_c.isFraud.sum()/df_c.isFraud.count(), 2)
print('Percentage of the fraud transactions is {}% out of all the transactions.'.format(fraud_ratio1))


# Let's check how many of Fraud transactions where flagged as Fraud one.

# In[14]:


perc1 = 100*round(df_c['isFlaggedFraud'][df_c.isFlaggedFraud == 1].count()/df_c['isFraud'][df_c.isFraud == 1].count(),4)
print('Percentage of the fraud transactions that were flagged is extremely low - {}%.'.format(perc1))


# We can see that there is a problem in the existing algorithm for flagging the fraud transactions.
# ____
# It's necessary to check how many of the fraudulent transactions were sent to the Merchant account (Account number starts with M).

# In[15]:


df_c[(df_c['nameDest'].str[0] == 'M') & (df_c['isFraud'] == 1)]


# > No fraud transactions were made to the 'Merchant' accounts. We may conclude that in the case of fraud, the account receiving money won't be a Merchant account.

# We can replace 'typeDestAccount' column with column indicating if the destination account is Customer account by value '1'. '0' will mean Merchant destination account in this case.

# In[16]:


df_c.loc[df_c['typeDestAccount'] == 'C', 'CustomerDestAccount'] = '1'
df_c.loc[df_c['typeDestAccount'] != 'C', 'CustomerDestAccount'] = '0'
df_c = df_c.drop(['typeDestAccount'], axis =1)
df_c.head()


# _____

# In[17]:


df_c.info()


# _____
# Transaction details.

# In[18]:


df_c[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].describe()


# In[19]:


df_c.type.unique()


# ___
# Let's create the column indicating time of the transaction in 24 hour format.

# In[20]:


df_c['hour'] = df_c['step']-(24*(df_c['step']//24))
df_c.info()


# ______
# ______
# # Data Visualising
# Let's plot distributions of the transactions in 24 hours and distribution of fraud transactions in 24 hours.

# In[21]:


plt.figure(figsize=(14,6))
sb.histplot(data = df_c.sample(10000), x='hour', bins=23, kde=True)
plt.title('Distribution of the Transactions in 24 hours')
plt.xlabel('Hour of the day')
plt.xlim(0,23);

plt.figure(figsize=(14,6))
sb.histplot(data = df_c[df_c['isFraud'] == 1], x='hour', bins=23, kde=True)
plt.title('Distribution of the Fraud Transactions in 24 hours')
plt.xlabel('Hour of the day')
plt.xlim(0,23);

plt.figure(figsize=(14,6))
sb.histplot(data = df_c[df_c['isFraud'] == 0], x='hour', bins=23, kde=True, kde_kws={'cut':500})
plt.title('Distribution of the Not Fraud Transactions in 24 hours')
plt.xlabel('Hour of the day')
plt.xlim(0,23);


# > From the plots above it can be seen that fraud transactions happen during all the day, while transactions in general were more likely to occure during the regular work hours.
# Let's create catecorical column to represent time of the day: night(00-07), morning (08-12), day(13-18), evening (19-23)

# In[22]:


df_c.loc[df_c['hour'] <= 7, 'time'] = 'NGHT'
df_c.loc[((df_c['hour'] > 7) & (df_c['hour'] <=12)), 'time'] = 'MRNG'
df_c.loc[((df_c['hour'] > 12) & (df_c['hour'] <=18)), 'time'] = 'DAY'
df_c.loc[((df_c['hour'] > 18) & (df_c['hour'] <=23)), 'time'] = 'EVE'
df_c.time.unique()


# In[23]:


df_c.time = pd.Categorical(df_c.time, ['NGHT', 'MRNG', 'DAY', 'EVE'])


# In[24]:


plt.figure(figsize=(14,6))
sb.histplot(data = df_c.sample(10000), x='time', bins=23, kde=False)
plt.title('Distribution of the Transactions in 24 hours')
plt.xlabel('Time of the day');

plt.figure(figsize=(14,6))
sb.histplot(data = df_c[df_c['isFraud'] == 1], x='time', bins=23, kde=False)
plt.title('Distribution of the Fraud Transactions in 24 hours')
plt.xlabel('Time of the day');


# In[25]:


df_c = df_c.drop(['step', 'hour'], axis =1)
df_c.head()


# ___
# Plot of distributions of the amount of money sent.

# In[26]:


binse1 = np.arange(0, max(df_c.amount)+4e4, 4e4)
plt.figure(figsize=(14,6))
sb.histplot(data = df_c.sample(10000), x='amount', bins=binse1, kde=True)
plt.title('Distribution of the Transaction amount')
plt.xlabel('Amount sent by the transaction, n')
plt.xlim(0,2e6);


# ____
# Plot of distributions of the amount of money sent.

# In[27]:


binse2 = np.arange(0, max(df_c.amount)+2e4, 2e4)
plt.figure(figsize=(14,6))
sb.histplot(data = df_c.sample(10000), x='oldbalanceOrg', bins=binse2, kde=True)
plt.title('Distribution of the Amount of assets on the account')
plt.xlabel('Balance on the sending account')
plt.xlim(0,1e6);


# ____
# Plot of distributions of the amount of money sent.

# In[28]:


binse3 = np.arange(0, max(df_c.sample(10000).amount)+2e4, 2e4)
plt.figure(figsize=(14,6))
sb.histplot(data = df_c.sample(10000), x='oldbalanceDest', bins=binse3, kde=True)
plt.title('Distribution of the Transaction amount')
plt.xlabel('Amount of the Transaction, s')
plt.xlim(0,1e6);


# ____
# Plot of distributions of the amount of money sent.

# In[29]:


binse4 = np.arange(0, max(df_c.sample(10000).amount)+2e4, 2e4)
plt.figure(figsize=(14,6))
sb.histplot(data = df_c.sample(10000), x='newbalanceDest', bins=binse4, kde=True)
plt.title('Distribution of the Transaction amount')
plt.xlabel('Amount of the Transaction, s')
plt.xlim(0,1e6);


# ____
# It may be helpful to create a dataframe with fraud transactions only. 

# In[30]:


df_c_fraudulent = df_c[df_c.isFraud == 1]
df_c_fraudulent.head()


# _____
# Let's check if there is pattern in the fraud transaction of sending 100% of assets.

# In[31]:


df_c_not_fraudulent = df_c[df_c['isFraud'] == 0]
percentage_of_sending_all_assets_fraud = round(100*df_c_fraudulent.newbalanceOrig[df_c_fraudulent.newbalanceOrig == 0].count()/df_c_fraudulent.newbalanceOrig.count(),2)
percentage_of_sending_all_assets_fulldataset = round(100*df_c.newbalanceOrig[df_c.isFraud == 0][df_c.newbalanceOrig == 0].count()/df_c.newbalanceOrig.count(),2)
plt.bar(x=[1,2], height =[percentage_of_sending_all_assets_fraud, percentage_of_sending_all_assets_fulldataset],tick_label=['sending_all_assets_fraud','sending_all_assets_dataset'])
print('{}% of fraud transactions transfered all the money from the account. Percentage of the Not Fraud transactions that transferred everything from the account of Origin is {}%'.format(percentage_of_sending_all_assets_fraud,percentage_of_sending_all_assets_fulldataset))


# > It looks like there is high probability that if the all of assets are being sent this is fraud transaction.

# In[32]:


perc2 = 100*round(df_c_fraudulent['newbalanceOrig'][df_c_fraudulent.newbalanceOrig != 0].count()/df_c_fraudulent['newbalanceOrig'][df_c_fraudulent.newbalanceOrig == 0].count(),3)
print('In general the majority of the Fraud transactions emptied the accounts completely. Percentage of the Fraud transactions that left something on the account of Origin is {}%'.format(perc2))


# Let's create a column indicating this, as this may be an important factor in finding fraud transactions.

# In[33]:


df_c['emptied_account'] = 0
df_c['emptied_account'][df_c['newbalanceOrig'] == 0] = 1


# ______
# Let's check how many fraudulent transactions occured during the recorded time and how often fraudulent transactions transfered money to the same accounts.

# In[34]:


# Number of fraudulent transactions
df_c_fraudulent.count()


# In[35]:


accounts_used_fewtimes = round(100 * df_c_fraudulent.nameDest.nunique()/df_c_fraudulent.nameDest.count(), 2)
print('Percentage of the accounts received more than one fraudulent transaction is {}%.'.format(accounts_used_fewtimes))


# As we can see, accounts that received fraudulent transactions were used multiple times. Let's drop innecessary column.

# In[36]:


df_c = df_c.drop(['nameDest'], axis =1)
df_c.head()


# _____

# In[37]:


fraudulent_transfers = 100*round(df_c_fraudulent['type'][df_c_fraudulent['type'] == 'TRANSFER'].count()/df_c_fraudulent['type'].count(), 3)
print('Percentage of the fraudulent transactions that were Transfers: {}%.'.format(fraudulent_transfers))


# In[38]:


print('For fraudulent transactions only two types where used: {}. Both of theme weere used almost equally often.'.format(df_c_fraudulent.type.unique()))


# In[39]:


plt.figure(figsize=(8,6))
sb.barplot(data = df_c.sample(10000), x='type', y='amount')
plt.title('Total amount sent by transaction type')
plt.xlabel('Transaction type')
plt.ylabel('Total amount of transaction');

plt.figure(figsize=(8,6))
sb.countplot(data = df_c_fraudulent, x='type')
plt.title('Total number of fraud transactions by transaction type')
plt.xlabel('Transaction type')
plt.ylabel('Total number of transactions');


# > Transfers and cashing outs were used for fraudulent transactions. They were used almost equally as often.
# _____
# Let's plot the amount of money transfered in a fraud transaction by transaction type:

# In[40]:


plt.figure(figsize=(8,6))
sb.barplot(data = df_c_fraudulent, x ='type' ,y ='newbalanceOrig')
plt.title('Total amount sent in fraud by transaction type')
plt.xlabel('Transaction type')
plt.ylabel('Total amount of transaction');


# > Overwhelming majority of money were actually sent by transfer in a fraudulent transactions, despite the fact that number of a transfer and cash out transactions were almost the same.
# _____
# ____

# # Building the categorisation/forecasting model
# 
# Let's create a dataframe for analysis.

# In[41]:


df_c.head()


# Getting rid off the columns that won't be used:

# In[42]:


df_model = df_c.copy()


# In[43]:


df_model.head()


# Building a correlation table for the dataframe.

# In[44]:


df_model.corr()


# ____

# In[45]:


df_model['type'].value_counts()


# In[46]:


df_model[df_model['isFraud'] == 1]['type'].value_counts()


# In[47]:


df_model.loc[(df_model['type'] == 'CASH_IN') | (df_model['type'] == 'DEBIT') | (df_model['type'] == 'PAYMENT'), 'ReceivingTransfer'] = 1
df_model.loc[(df_model['type'] == 'CASH_OUT') | (df_model['type'] == 'TRANSFER'), 'ReceivingTransfer'] = 0

df_model.loc[df_model['type'] == 'CASH_OUT', 'CASH_OUT'] = 1
df_model.loc[df_model['type'] != 'CASH_OUT', 'CASH_OUT'] = 0

df_model.loc[df_model['type'] == 'TRANSFER', 'TRANSFER'] = 1
df_model.loc[df_model['type'] != 'TRANSFER', 'TRANSFER'] = 0


# In[48]:


df_model['time'].value_counts()


# In[49]:


df_model['CustomerDestAccount'].value_counts()


# ____
# Creating dummy tables for the categorical values.

# In[50]:


df_model_dum = df_model.copy()
df_model_dum[['NGHT', 'MRNG', 'DAY', 'EVE']] = pd.get_dummies(df_model_dum['time'])

df_model_dum.head()


# In[51]:


df_model_dum.drop(['type','isFlaggedFraud'], axis = 1, inplace= True)
df_model_dum.head()


# In[52]:


df_model_dum.corr()


# In[53]:


df_model_dum.info()


# ____
# ____
# ## Logistic regression

# 
# - Fraud transactions were sent to Customer accounts only.
# - Accounts receiving fraud transactions were used a couple of times.
# - Cash out and trasfer were used as a fraudulent transactions. Transfers took overwhelming majority of money.
# - Most of the fraudulent transactions emptied account completely.
# - Fraud transactions happen during all the day, while transactions in general were more likely to occur during the regular work hours.

# In[54]:


df_model_dum.columns


# In[55]:


df_model_dum.isFraud.value_counts()


# In[56]:


# ___________
# ___________

# ## statsmodel

# In[57]:


df_model_dum.head()


# In[58]:


df_model_dum.info()


# In[59]:


df_model_dum[['CustomerDestAccount',
              'ReceivingTransfer', 'CASH_OUT', 'TRANSFER',
              'NGHT', 'MRNG', 'DAY', 'EVE']] = df_model_dum[['CustomerDestAccount',
                                                             'ReceivingTransfer', 'CASH_OUT', 'TRANSFER',
                                                             'NGHT', 'MRNG', 'DAY', 'EVE']].astype('int64')


# In[60]:


df_model_dum.info()


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(df_model_dum[['amount', 'newbalanceOrig', 'newbalanceDest',
    'CustomerDestAccount', 'emptied_account', 'ReceivingTransfer', 'NGHT', 'MRNG','DAY']],
                                                    df_model_dum['isFraud'], train_size=0.8 , random_state=10)


# In[62]:


sb.clustermap(x_train.corr())


# In[63]:


df_model_dum['intercept']=1

logist_model = sm.Logit(y_train, x_train)
results = logist_model.fit()
results.summary()


# As R_squared is 0.366, that means that we can describe 33.6% of transactions on being fraudulent or not.
# _____
# ____

# In[64]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(df_model_dum[['amount','newbalanceDest','CustomerDestAccount', 
 'NGHT', 'MRNG','DAY']],
                                                    df_model_dum['isFraud'], train_size=0.8 , random_state=10)


# In[65]:


sb.clustermap(x_train2.corr())


# In[66]:


df_model_dum['intercept']=1

logist_model2 = sm.Logit(y_train2, x_train2)
results2 = logist_model2.fit()
results2.summary()


# _____
# ____
# ## sklearn model

# In[67]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(df_model_dum[['amount', 'newbalanceOrig', 'newbalanceDest',
    'CustomerDestAccount', 'emptied_account', 'ReceivingTransfer', 'NGHT', 'MRNG','DAY']],
                                                    df_model_dum['isFraud'], train_size=0.8, random_state=10)


# In[68]:


LogitModel = LogisticRegression()
results2 = LogitModel.fit(x_train2, y_train2)


# looking a the train data

# In[69]:


from sklearn.metrics import accuracy_score
predictions2 = LogitModel.predict(x_train2)
accuracy_score(y_train2, predictions2)


# looking a the test data

# In[70]:


predictions3 = LogitModel.predict(x_test2)
accuracy_score(y_test2, predictions3)


# Confusion matrix:

# In[71]:


confusion_matrix(y_test2, predictions3)


# In[72]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Log_roc_auc= roc_auc_score(y_test2, LogitModel.predict(x_test2))
fpr, tpr, threshold=roc_curve(y_test2, LogitModel.predict_proba(x_test2)[:,1])


# In[73]:


plt.figure()
plt.plot(fpr, tpr, label = 'Logit Model 1 (area = %0.2f)'% Log_roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()


# ____
# ____

# In[74]:


x_train4, x_test4, y_train4, y_test4 = train_test_split(df_model_dum[['amount','newbalanceDest','CustomerDestAccount', 
'ReceivingTransfer', 'NGHT', 'MRNG','DAY']], df_model_dum['isFraud'], train_size=0.8, random_state=100)


# In[75]:


LogitModel4 = LogisticRegression()
results4 = LogitModel4.fit(x_train4, y_train4)


# In[76]:


from sklearn.metrics import accuracy_score
predictions_train = LogitModel4.predict(x_train4)
accuracy_score(y_train4, predictions_train)


# In[77]:


predictions_test = LogitModel4.predict(x_test4)
accuracy_score(y_test4, predictions_test)


# In[78]:


print('The model accuracy is {}% .'.format(round(100*(LogitModel4.score(x_test4, y_test4)), 2)))


# In[79]:


confusion_matrix(y_test4, predictions_test, labels= [1,0])


# In[80]:


TP, FN, FP, TN = confusion_matrix(y_test4, predictions_test, labels= [1,0]).ravel()
TP, FN, FP, TN


# In[81]:


print('Precision P=(TP/(TP+FP)) is {}% .'.format(100*round(TP/(TP+FP),4)))
print('Negative predictive value (TN/(TN+FN)) is {}% .'.format(100*round(TN/(TN+FN),4)))
print('Sensitivity Sn=(TP/(TP+FN)) is {}% .'.format(100*round(TP/(TP+FN),5)))
print('Specificity Sp=(TN/(TN+FP)) is {}% .'.format(100*round(TN/(TN+FP),4)))
print('F-score 2*(P*Sn/(P+Sn)) is {}% .'.format(100*round(2*((TP/(TP+FP))*(TP/(TP+FN)))/((TP/(TP+FP))+(TP/(TP+FN))),4)))


# In[82]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
Log_roc_auc= roc_auc_score(y_test4, LogitModel4.predict(x_test4))
fpr, tpr, threshold=roc_curve(y_test4, LogitModel4.predict_proba(x_test4)[:,1])


# In[83]:


plt.figure()
plt.plot(fpr, tpr, label = 'Logit Model 1 (area = %0.2f)'% Log_roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()


# In[84]:


print('coeficcient = ',LogitModel4.coef_)
print('intercept = ',LogitModel4.intercept_)


# In[85]:


y_pred = LogitModel4.predict(x_test4)


# _____

# In[86]:


print('This prediction model will predict correctly {}% of the fraud transactions and {}% of the not fraud transactions.'.format(100*round(TP/(TP+FP),5),100*round(TN/(TN+FN),4)))


# This model is far from perfection, but it is definitely improvement compared to the prediction model that was used previously. False Postive results obtained in this model can be mitigated during manual review as the number of these transaction remains pretty low. In order to get better accuracy more data needs to be collected, the data can be cleaned from outliers, etc.
