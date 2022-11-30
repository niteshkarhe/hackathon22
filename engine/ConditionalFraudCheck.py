import pandas as pd


class ConditionalFraudCheck:

    def __init__(self):
        self.data = pd.read_csv("../data/customer_transactions.csv")
        self.data["type"] = self.data["type"].map(
            {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
        self.data["isFraud"] = self.data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

    def check_transaction_to_new_customer(self, input_trans_df):
        input_trans_df["prediction"] = 'No Fraud'
        input_trans_df["indication"] = ''

        for index, row in input_trans_df.iterrows():
            # If previous transaction was Fraud
            if (row['type'] == 'PAYMENT' or row['type'] == 'TRANSFER' or row['type'] == 'DEBIT'):
                transfer_perc = (row['amount'] / row['oldbalanceOrg']) * 100

                prevTransaction = self.data[(self.data['nameOrig'] == row['fromAccNo']) & (
                        self.data['nameDest'] == row['destAccNo'])]

                if prevTransaction.empty and transfer_perc >= 80.0:
                    input_trans_df.at[index, 'prediction'] = "Can be Fraud"
                    input_trans_df.at[index, 'indication'] = "Huge amount transfer to new Customer"

            destAccNoTransactions = self.data[(self.data['nameDest'] == row['destAccNo'])]
            if not destAccNoTransactions.empty:
                for prevTr in prevTransaction.itertuples():
                    prevTrDict = prevTr._asdict()
                    if prevTrDict['isFraud'] == 'Fraud':
                        # if prevTr.iloc[prevTrIdx, prevTr.columns.get_loc('isFraud')] == 'Fraud':
                        # print("engine transaction")
                        input_trans_df.at[index, 'prediction'] = "Can be Fraud"
                        input_trans_df.at[index, 'indication'] = "Previous Transaction of same customer was Fraud"
                        break

                if input_trans_df.at[index, 'prediction'] != '':
                    destAccNoAvgTrans = destAccNoTransactions.groupby("nameOrig").agg({'nameOrig': ['count']}).max() / destAccNoTransactions.shape[0] * 100
                    print(destAccNoAvgTrans)
                    if destAccNoAvgTrans[0] <= 50:
                        input_trans_df.at[index, 'prediction'] = "Can be Fraud"
                        input_trans_df.at[index, 'indication'] = "Destination Customer flagged as Fraud because New " \
                                                                 "Customer Transactions are More than Average "


