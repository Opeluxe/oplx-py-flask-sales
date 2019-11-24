#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
#import numpy as np
import dill as pickle
import flask
from flask import request
import os
import json

import warnings
warnings.filterwarnings("ignore")

# Constants for layout and dataframe manipulation
DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/test_k1.csv'
MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/model/model_v3.pk'
FEATURES = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']

# Data manipulation routines
def load_data(nrows):
    loaded_data = pd.read_csv(DATA_PATH, sep=',', error_bad_lines=False, low_memory=False, nrows=nrows)
    loaded_data = loaded_data[FEATURES]
    return loaded_data
    
def load_model():
    pickle._dill._reverse_typemap['ClassType'] = type
    with open(MODEL_PATH, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model
    
def predict_data(data, model):
    predicted_values = model.predict(data)
    data = data[FEATURES]
    data['Sales'] = predicted_values
    data['Id'] = data.index
    return predicted_values, data

app = flask.Flask(__name__)

#defining a route for only post requests
@app.route('/api', methods=['POST'])
def index():
    #getting an array of features from the post request's body
    nrows = request.get_json()['number']
    
    #predicting using model
    loaded_model = load_model()
    loaded_data = load_data(nrows=nrows)
    predicted_values, selected_data = predict_data(loaded_data, loaded_model)

    #returning the response object as json
    return flask.jsonify(json.loads(selected_data.to_json(orient='records')))