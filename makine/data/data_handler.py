"""
data processing Module

This module reads CSV files, preprocesses the data, and
splits it into training/test sets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_csv(file_path):
    """
    Reads the CSV file and returns it as a DataFrame.
        
    Parameters:
        file_path (str): path of CSV file
        
    Transformes to the:
        pandas.DataFrame: readed data set
    """
    #  read CSV file
    dataframe = pd.read_csv(file_path)
    
    # Print first 5 row to the console for debug
    print("Data set has been successfully loaded.!")
    print("First 5 row:")
    print(dataframe.head())
    print(f"\nTotal row number: {len(dataframe)}")
    print(f"Coulumn number: {len(dataframe.columns)}")
    
    return dataframe


def get_column_info(dataframe):
    """
    Returns coulumn info in DataFrame
    
    Parameters:
        dataframe (pandas.DataFrame): data set
        
    Transformes to the:
        dict: Coulumn name and types
    """
    column_info = {}
    
    for column_name in dataframe.columns:
        column_type = str(dataframe[column_name].dtype)
        
        # Determine whether it is categorical or numerical
        if column_type == 'object':
            column_info[column_name] = 'categorical'
        else:
            column_info[column_name] = 'numerical'
    
    return column_info


def apply_one_hot_encoding(dataframe, columns_to_encode=None):
    """
    Applies One-Hot Encoding on categorical coulumns
    
    Parameters:
        dataframe (pandas.DataFrame): data set
        columns_to_encode (list): coulumns to be encoded(if it is None than finds otomatically)
        
    Transformes to the:
        pandas.DataFrame: encoded data set
    """
    # get a copy (dont change original)
    df_encoded = dataframe.copy()
    
    #  If columns are not specified, find the categorical ones
    if columns_to_encode is None:
        columns_to_encode = []
        for column_name in df_encoded.columns:
            if df_encoded[column_name].dtype == 'object':
                columns_to_encode.append(column_name)
    
    # Apply One-Hot Encoding for every categorical coulumn
    for column_name in columns_to_encode:
        # use pandas get_dummies fnc
        one_hot = pd.get_dummies(df_encoded[column_name], prefix=column_name)
        
        #remove original coulumn and add new coulumns
        df_encoded = df_encoded.drop(column_name, axis=1)
        df_encoded = pd.concat([df_encoded, one_hot], axis=1)
    
    print(f"One-Hot Encoding applied: {columns_to_encode}")
    
    return df_encoded


def apply_label_encoding(series):
    """
    Turns target coulumn to numerical values.
    
    Parameters:
        series (pandas.Series): target coulumn
        
    Transformes to the:
        numpy.ndarray: encoded values
        LabelEncoder: used encoder
    """
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(series)
    
    print(f"Label Encoding applies. Classes: {list(encoder.classes_)}")
    
    return encoded_values, encoder


def apply_normalization(X, method='standard'):
    """
    Applies normalization to numerical data.
    
    Parameters:
        X (numpy.ndarray veya pandas.DataFrame): feature matrix
        method (str): 'standard' or 'minmax'
        
    Transformes to the:
        numpy.ndarray: Normalized data
        scaler: used scalar object
    """
    # Choose scaler wrt method type
    if method == 'standard':
        scaler = StandardScaler()
        print("StandardScaler is being applied (ortalama=0, std=1)")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        print("MinMaxScaler is being applied (min=0, max=1)")
    else:
        # no normalizasyon 
        print("Normalization couldn't applied")
        return X, None
    
    # converting data
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, scaler


def split_data(X, y, train_ratio=0.7):
    """
    splits data into train and test tests.
    
    Parameters:
        X (numpy.ndarray): feature matrix
        y (numpy.ndarray): Target variable
        train_ratio (float): training set ratio (0.0 - 1.0)
        
    Transformes to the:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # calculate test ratio
    test_ratio = 1.0 - train_ratio
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_ratio, 
        random_state=42
    )
    
    print(f"\nData was split:")
    print(f"  Train set: {len(X_train)} example ({train_ratio*100:.0f}%)")
    print(f"  Test set: {len(X_test)} example ({test_ratio*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def prepare_data(dataframe, target_column, encoding=True, normalization='standard', train_ratio=0.7):
    """
    Combines all data preparation steps into a single function.
    
    Parameters:
        dataframe (pandas.DataFrame): Raw data set
        target_column (str): Target column name
        encoding (bool): Should one-hot encoding be applied?
        normalization (str): 'standard', 'minmax' or 'none'
        train_ratio (float): Test set ratio
        
    Transformes to the:
        tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    print("\n" + "="*50)
    print("DATA PREPARATION STARTED")
    print("="*50)
    
    # 1. Separate the target column
    y = dataframe[target_column]
    X = dataframe.drop(target_column, axis=1)
    
    print(f"\ntarget column: {target_column}")
    print(f"Number of features: {len(X.columns)}")
    
    # 2. encode the target column (if it is categorical)
    label_encoder = None
    if y.dtype == 'object':
        y, label_encoder = apply_label_encoding(y)
    else:
        y = y.values
    
    # 3.Apply One-Hot Encoding (if it is wanted)
    if encoding:
        X = apply_one_hot_encoding(X)
    
    # 4. Convert to numerical values
    X = X.values
    
    # 5. Apply normalization
    if normalization != 'none':
        X, scaler = apply_normalization(X, method=normalization)
    
    # 6. Seperate into Train/Test sets
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio)
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETED")
    print("="*50)
    
    return X_train, X_test, y_train, y_test, label_encoder
