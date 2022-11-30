
def validate(input_trans_pd):
    for transaction in input_trans_pd.itertuples():
        transactionDict = transaction._asdict()
        if (transactionDict['type'] == 'CASH_OUT' or transactionDict['type'] == 'PAYMENT' or transactionDict['type'] == 'TRANSFER' or transactionDict['type'] == 'DEBIT') and transactionDict["oldbalanceOrg"] < transactionDict["amount"]:
            return True


