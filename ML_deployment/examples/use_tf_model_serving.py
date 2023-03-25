# -*- coding: utf-8 -*-
"""use_tf_model_serving.ipynb
"""

import json
import requests

import numpy as np

import pickle
scaler_colab = pickle.load(open('sc.pickle','rb'))
scaler_colab.transform(np.array([[20,40000]]))
scaler_colab.transform(np.array([[42,50000]]))

url = 'http://35.238.92.68:8501/v1/models/customer_behavior_model:predict'

request_data = json.dumps({"signature_name": "serving_default",
                   "instances": [[-1.43318661, -0.47466685],[0.2345214460208433, 0.03675871227617118]]
})

json_response = requests.post(url,request_data)
print (json_response.text)

