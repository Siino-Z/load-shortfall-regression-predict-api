"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np
import json


# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Kaggle challenge.
test = pd.read_csv('./data/df_test.csv')


# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Print the generated JSON payload
print("Generated JSON payload:")
print(feature_vector_json)
print("*" * 50)

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://63.35.192.114:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()[0]}")
print(f"API response content: {api_response.text}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
# Check if the response is empty
if not api_response.text:
    print("API response is empty.")
else:
    try:
        # Try to parse the JSON response
        result = api_response.json()
        print(f"API prediction result: {result[0]}")
    except json.decoder.JSONDecodeError:
        print("Error decoding JSON. Check API response format.")