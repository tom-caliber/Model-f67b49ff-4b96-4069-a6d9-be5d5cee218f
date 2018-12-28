import os
import pandas as pd
from sklearn.externals import joblib
from joblib import dump, load
from flask import Flask, jsonify, request
import dill as pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        data_json = request.get_json()
        data = pd.read_json(data_json, orient='records')

    except Exception as e:
        raise e

    if data.empty:
        return(bad_request())
    else:
        #Load the saved model
        print("Loading the model...")
        clf = None
        filename = 'model.pk'
        with open(filename,'rb') as f:
            clf = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = clf.predict(data)

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(prediction_series)

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp