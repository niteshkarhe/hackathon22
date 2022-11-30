import pandas as pd
from flask import Response, Flask, request

from engine.MLAlgoFraudCheck import MLAlgoFraudCheck
from engine.ConditionalFraudCheck import ConditionalFraudCheck

# initialize API
app = Flask(__name__)


@app.route('/fraud/predict', methods=['POST'])
def fraud_predict():
    input_transaction = request.get_json()
    if input_transaction:  # there is data

        # Instantiating Machine Learning Algorithm Class
        mlAlgoEngineObj = MLAlgoFraudCheck()

        if isinstance(input_transaction, dict):
            input_trans_pd = pd.DataFrame(input_transaction, index=[0])
        else:
            input_trans_pd = pd.DataFrame(input_transaction, columns=input_transaction[0].keys())

        input_trans_pd["newbalanceOrg"] = input_trans_pd["oldbalanceOrg"] - input_trans_pd["amount"]

        input_trans_pd = mlAlgoEngineObj.get_decision_tree_prediction(input_trans_pd)

        realtimeFraudCheck_obj = ConditionalFraudCheck()
        realtimeFraudCheck_obj.check_transaction_to_new_customer(input_trans_pd)

        return input_trans_pd.to_json(orient="records", date_format="iso")

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('0.0.0.0')
