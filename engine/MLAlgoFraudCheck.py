import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class MLAlgoFraudCheck:
    def __init__(self):
        self.data = pd.read_csv("../data/customer_transactions.csv")
        self.data["type"] = self.data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                                   "CASH_IN": 3, "TRANSFER": 4,
                                                   "DEBIT": 5})
        self.data["isFraud"] = self.data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

    def get_decision_tree_prediction(self, input_trans_pd):
        # type, amount, oldbalanceOrg, newbalanceOrig
        transaction_pd = input_trans_pd.drop(columns=['fromAccNo', 'destAccNo'], axis=1)

        # transform the values of the isFraud column into No Fraud and Fraud labels
        transaction_pd["type"] = transaction_pd["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                                             "CASH_IN": 3, "TRANSFER": 4,
                                                             "DEBIT": 5})
        transaction_pd["type"] = transaction_pd["type"].astype(float)

        transaction_pd.iloc[0]
        json_csv = transaction_pd.to_numpy()

        # splitting the data
        x = pd.np.array(self.data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        y = pd.np.array(self.data[["isFraud"]])

        # training a machine learning model
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
        dec_tree_model = DecisionTreeClassifier()
        dec_tree_model.fit(xtrain, ytrain)
        dec_tree_accuracy = dec_tree_model.score(xtest, ytest)
        print("Dec Tree Accuracy = " + str(dec_tree_accuracy))
        input_trans_pd['decisionTreeAccuracy'] = dec_tree_accuracy

        # prediction
        # features = [type, amount, oldbalanceOrg, newbalanceOrig]
        features = pd.np.array(json_csv)
        # features = pd.np.array([[4, 9000.60, 9000.60, 0.0], [4, 9000.60, 8000.60, 1000.0], [4, 9000.60, 0.60, 9000.0]])
        dec_tree_predict = dec_tree_model.predict(features)
        print("Dec Tree Prediction = " + dec_tree_predict)
        input_trans_pd['decisionTreePrediction'] = dec_tree_predict

        #
        # rand_for_model = RandomForestClassifier(class_weight='balanced')
        # rand_for_model.fit(xtrain, ytrain)
        #
        # rand_for_accuracy = rand_for_model.score(xtest, ytest)
        # print("RandomForest Accuracy = " + str(rand_for_accuracy))
        # input_trans_pd['RandomForestAccuracy'] = rand_for_accuracy
        #
        # rand_for_predict = rand_for_model.predict(features)
        # print("Random Forest = " + rand_for_predict)
        # input_trans_pd['RandomForestPrediction'] = rand_for_predict
        #
        #
        #
        #
        #
        #
        # log_regr_model = LogisticRegression()
        # log_regr_model.fit(xtrain, ytrain)
        #
        # log_regr_accuracy = log_regr_model.score(xtest, ytest)
        # print("Logistic Regression Accuracy = " + str(log_regr_accuracy))
        # input_trans_pd['LogisticRegressionAccuracy'] = log_regr_accuracy
        #
        # log_regr_predict = log_regr_model.predict(features)
        # print("Logistic Regression = " + log_regr_predict)
        # input_trans_pd['LogisticRegressionPrediction'] = log_regr_predict

        return input_trans_pd
