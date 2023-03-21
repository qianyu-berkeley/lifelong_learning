from flask import Flask, request
import pickle
import numpy as np

local_classifier = pickle.load(open('classifier.pickle', 'rb'))
local_scaler = pickle.load(open('sc.pickle', 'rb'))

app = Flask(__name__)

@app.route('/model', method=['POST'])
def scorer():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    pred_proba = local_classifier.predict_proba(local_scaler.transform(np.array([[age,salary]])))[:,1]

    #return f"Prediction is {prediction}"
    return f"The prediction is {pred_proba}"

if __name__ == "__main__":
    app.run(port=8005, debug=True)