import joblib
import pandas as pd
from flask import Response, Flask, request
import plotly.express as px

from api.fraud.Fraud import Fraud
from api.fraud.RealtimeFraudCheck import RealtimeFraudCheck

model = joblib.load('../models/model_cycle1.joblib')

# initialize API
app = Flask(__name__)

# data = pd.read_csv("fraud_0.1origbase.csv")
# print(data.head())
#
# print(data.isnull().sum())
#
# # Exploring transaction type
# print(data.type.value_counts())
#
# type = data["type"].value_counts()
# transactions = type.index
# quantity = type.values
#
# #figure = px.pie(data,
#              # values=quantity,
#              # names=transactions,hole = 0.5,
#              # title="Distribution of Transaction Type")
# #figure.show()
#
# # Checking correlation
# correlation = data.corr()
# print(correlation["isFraud"].sort_values(ascending=False))
#
# # transform the values of the isFraud column into No Fraud and Fraud labels
# data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
#                                  "CASH_IN": 3, "TRANSFER": 4,
#                                  "DEBIT": 5})
# data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
# print(data.head())
#
# # splitting the data
# x = pd.np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
# y = pd.np.array(data[["isFraud"]])
#
# # training a machine learning model
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
# model = DecisionTreeClassifier()
# model.fit(xtrain, ytrain)
# print("Dec Tree Accuracy = "+ str(model.score(xtest, ytest)))
#
# # prediction
# #features = [type, amount, oldbalanceOrg, newbalanceOrig]
# features = pd.np.array([[4, 9000.60, 9000.60, 0.0], [4, 9000.60, 8000.60, 1000.0],[4, 9000.60, 0.60, 9000.0]])
# print("Dec Tree = " +model.predict(features))
#
# rf = RandomForestClassifier(class_weight='balanced')
# rf.fit(xtrain, ytrain)
#
# y_pred = rf.predict(features)
# print("Random Forest = " + y_pred)
#
#
# scr = 'recall'
# accuracy_dict = {}
# model_lr = LogisticRegression()
# model_rf = RandomForestClassifier()
# skf = StratifiedKFold(5)
#
# model_lr.fit(xtrain, ytrain)
# lr = model_lr.predict(features)
# # sc_lr = cross_val_score(model_lr, xtrain, ytrain, cv=skf,
# # scoring=scr)
#
# print("Log Regr & Rand For  = " + lr)
#
# print("kuch nahi")



@app.route('/fraud/predict', methods=['POST'])
def churn_predict():
    input_transaction = request.get_json()
    # data = pd.read_csv("fraud_0.1origbase.csv")
    if input_transaction: # there is data
        # Instantiate Rossmann class
        fraud_obj = Fraud()

        if isinstance(input_transaction, dict): # unique example
            input_trans_pd = pd.DataFrame(input_transaction, index=[0])
            
        else: # multiple example
            input_trans_pd = pd.DataFrame(input_transaction, columns=input_transaction[0].keys())

        input_trans_pd["newbalanceOrg"] = input_trans_pd["oldbalanceOrg"] - input_trans_pd["amount"]

        input_trans_pd = fraud_obj.get_decision_tree_prediction(input_trans_pd)


        realtimeFraudCheck_obj = RealtimeFraudCheck()
        realtimeFraudCheck_obj.check_transaction_to_new_customer(input_trans_pd)


        return input_trans_pd.to_json(orient="records", date_format="iso")

        
    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('0.0.0.0') 
