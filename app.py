#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import dill as pickle
import flask
from flask import request
import os
import json

import warnings
warnings.filterwarnings("ignore")

# Constants for layout and dataframe manipulation
DATA_FULL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/test_k.csv'
DATA_SAMPLE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/test_ks.csv'
MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/model/model_v3.pk'
FEATURES = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
DATA_SORT = ['Date', 'Store']

# Data manipulation routines
def load_data(nrows, sample, random):
    loaded_data = pd.read_csv(DATA_SAMPLE_PATH if sample else DATA_FULL_PATH, 
                              sep=',', 
                              error_bad_lines=False, 
                              low_memory=False)
    loaded_data = select_data(loaded_data, nrows, random)
    loaded_data.sort_values(by=DATA_SORT, inplace=True)
    return loaded_data[FEATURES]
    
def select_data(data, nrows, rndm):
    total_rows = len(data)
    if total_rows > nrows:
        drop_rows = total_rows - nrows
        if rndm == False:
            drop_indx = np.linspace(start=nrows, stop=total_rows - 1, 
                                    num=drop_rows, dtype=int)
        else:
            drop_indx = np.sort(np.random.choice(total_rows - 1, drop_rows, replace=False))
        selected_data = data.drop(drop_indx)
        selected_data.reset_index(inplace = True)
    else:
        selected_data = data
    return selected_data

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
    sample = request.get_json()['sample']
    random = request.get_json()['random']
    
    #predicting using model
    loaded_model = load_model()
    loaded_data = load_data(nrows=nrows, sample=sample, random=random)
    predicted_values, selected_data = predict_data(loaded_data, loaded_model)

    #returning the response object as json
    return flask.jsonify(json.loads(selected_data.to_json(orient='records')))