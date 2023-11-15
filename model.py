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
    
    # Convert 'time' column to datetime
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])

    # Extract date and time features
    feature_vector_df['Day'] = feature_vector_df['time'].dt.day
    feature_vector_df['Month'] = feature_vector_df['time'].dt.month
    feature_vector_df['Year'] = feature_vector_df['time'].dt.year
    feature_vector_df['Hour'] = feature_vector_df['time'].dt.hour
    
    # Select specific columns
    feature_vector_df = feature_vector_df[['Year', 'Month', 'Day', 'Hour', 'Madrid_wind_speed', 'Madrid_humidity',
                                           'Madrid_clouds_all', 'Madrid_pressure', 'Madrid_rain_1h', 'Madrid_temp',
                                           'Seville_humidity', 'Seville_clouds_all', 'Seville_wind_speed',
                                           'Seville_pressure', 'Seville_rain_1h', 'Seville_rain_3h', 'Seville_temp',
                                           'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Barcelona_rain_1h',
                                           'Barcelona_pressure', 'Barcelona_rain_3h', 'Barcelona_temp',
                                           'Valencia_wind_speed', 'Valencia_wind_deg', 'Valencia_humidity',
                                           'Valencia_snow_3h', 'Valencia_pressure', 'Valencia_temp', 'Bilbao_wind_speed',
                                           'Bilbao_wind_deg', 'Bilbao_clouds_all', 'Bilbao_pressure', 'Bilbao_rain_1h',
                                           'Bilbao_snow_3h', 'Bilbao_temp']]
    
     # Winsorize to handle outliers
    for column in feature_vector_df.columns:
        winsorized_data = winsorize(feature_vector_df[column], limits=(0.05, 0.05))
        feature_vector_df[column] = winsorized_data
    
    # Feature selection using VarianceThreshold
    sel = VarianceThreshold(threshold=0.1)
    feature_vector_df = feature_vector_df[feature_vector_df.columns[sel.get_support(indices=True)]]
    
    # Add constant for intercept in statsmodels
    feature_vector_df = sm.add_constant(feature_vector_df)
    
    # Calculate VIF and exclude high VIF columns
    vif_data = pd.DataFrame()
    vif_data['Variable'] = feature_vector_df.columns
    vif_data['VIF'] = [variance_inflation_factor(feature_vector_df.values, i) for i in range(feature_vector_df.shape[1])]
    columns_to_exclude = vif_data[vif_data['VIF'] > 5]['Variable']
    feature_vector_df = feature_vector_df.drop(columns_to_exclude, axis=1)

    predict_vector = feature_vector_df
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
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
