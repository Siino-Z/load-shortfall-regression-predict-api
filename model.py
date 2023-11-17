"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies

import numpy as np
import pandas as pd
import pickle
import json
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    train_df = feature_vector_df
    train_df = train_df.drop(['Unnamed: 0'], axis=1)
    # create copy of train_df
    train_copy_df = train_df.copy(deep = True)
    
    # Impute null values in Valencia_pressure with the column's median
    train_copy_df['Valencia_pressure'].fillna(train_copy_df['Valencia_pressure'].median(), inplace=True)
    
    # Define the training dataset
    feature_vector_df = train_copy_df [['Year','Month','Day','Hour','Madrid_wind_speed', 'Madrid_humidity', 'Madrid_clouds_all',
       'Madrid_pressure', 'Madrid_rain_1h', 'Madrid_weather_id', 'Madrid_temp',
       'Seville_humidity', 'Seville_clouds_all', 'Seville_wind_speed',
        'Seville_rain_1h', 'Seville_rain_3h',
       'Seville_weather_id', 'Seville_temp', 'Barcelona_wind_speed',
       'Barcelona_wind_deg', 'Barcelona_rain_1h', 'Barcelona_pressure',
       'Barcelona_rain_3h', 'Barcelona_weather_id', 'Barcelona_temp',
       'Valencia_wind_speed', 'Valencia_humidity',
       'Valencia_snow_3h', 'Valencia_pressure', 'Valencia_temp',
       'Bilbao_wind_speed', 'Bilbao_wind_deg', 'Bilbao_clouds_all',
       'Bilbao_pressure', 'Bilbao_rain_1h', 'Bilbao_snow_3h',
       'Bilbao_weather_id', 'Bilbao_temp']]
    
    
    predict_vector = feature_vector_df.copy()
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    print("Model Type:", type(model))
    print("Model Contents:", model)
    
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)

    # Print or log the type and contents of the prediction.
    print("Prediction Type:", type(prediction))
    print("Prediction Contents:", prediction)

    # Format as list for output standardisation.
    return prediction[0].tolist()
