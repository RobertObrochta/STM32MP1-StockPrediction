from yahoo_fin import stock_info as si
from datetime import date
from dateutil.relativedelta import *
import time

import yfinance as yfin

# gets rid of Pandas FutureWarnings
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque

import os
import numpy as np
import pandas as pd
import random


# creates these directories...
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

# set date, create a file with that date, and saves it
today_date = date.today()
three_months_ago_today = today_date + relativedelta(months = -3)
data_filename = os.path.join("data", f"STM-{today_date}.csv") 


def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def collect_data(ticker = "STM", n_steps = 50, scale = True, shuffle = True, lookup_step = 1, split_by_date = True, test_size = 0.2, feature_columns = ['volume', 'open', 'high', 'low', "close"]):

    stock_data = si.get_data(ticker, start_date = three_months_ago_today, end_date = today_date)

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in stock_data.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "date" not in stock_data.columns:
        stock_data["date"] = stock_data.index

    result = {}
    result['stock_data'] = stock_data.copy()

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            stock_data[column] = scaler.fit_transform(np.expand_dims(stock_data[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    stock_data['future'] = stock_data['close'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(stock_data[feature_columns].tail(lookup_step))
    # drop NaNs
    stock_data.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(stock_data[feature_columns + ["date"]].values, stock_data['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_stock_data"] = result["stock_data"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_stock_data"] = result["test_stock_data"][~result["test_stock_data"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    print(result)
    return result

data = collect_data()
data["stock_data"].to_csv(data_filename)

