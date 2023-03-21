# -*- coding: utf-8 -*-
"""colab_rest_api.ipynb

Setup rest api in google colab
Original file is located at
    https://colab.research.google.com/drive/colab_id
"""
import numpy as np
import pickle


classifier_colab = pickle.load(open('classifier.pickle','rb'))
scaler_colab = pickle.load(open('sc.pickle','rb'))

# Need to use flask-ngrok module
#!pip install flask-ngrok
from flask_ngrok import run_with_ngrok
from flask import Flask , request
import requests

app = Flask(__name__)

run_with_ngrok(app)

@app.route('/predict',methods=['POST'])
def customer_behavior():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    pred_proba = classifier_colab.predict_proba(scaler_colab.transform(np.array([[age,salary]])))[:,1]
    return "The prediction is {}".format(pred_proba)

app.run()

