"""

    Simple Flask-based API for Serving an sklearn Model.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file instantiates a Flask webserver
    as a means to create a simple API used to deploy models trained within
    the sklearn framework.

"""

# API Dependencies
import pickle
import json
import numpy as np
from model import load_model, make_prediction
from flask import Flask, request, jsonify

# Application definition
app = Flask(__name__)

# Load our model into memory.
# Please update this path to reflect your own trained model.
static_model = load_model(path_to_model='assets/trained-models/gb2_model.pkl')

# Ensure the model is loaded successfully before starting the application
if static_model is not None:
    print('-' * 40)
    print('Model successfully loaded')
    print('-' * 40)
else:
    print('Error: Unable to load the model. Please check the model path.')

""" You may use this section (above the app routing function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

# ... (existing app route)

@app.route('/api_v0.1', methods=['POST'])
def model_prediction():
    
    # We retrieve the data payload of the POST request
    data = request.get_json(force=True)
    
    # We then preprocess our data, and use our pretrained model to make a
    # prediction.
    output = make_prediction(data, static_model)
    
    # We finally package this prediction as a JSON object to deliver a valid
    # response with our API.
    return jsonify(output)

# ... (existing server configuration)

# Configure Server Startup properties.
# Note:
# When developing your API, set `debug=True`
# This will allow Flask to automatically restart itself every time you
# update your API code.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)