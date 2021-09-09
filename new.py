
from datetime import date
from dateutil.relativedelta import *
import time

import yfinance as yfin

# gets rid of Panda's FutureWarnings
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
import matplotlib.pyplot as matpl

'''
Initialization Code ..................................................................................................................................................................
'''

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
stock_ticker = "STM"

day_outlook = 1
scale = True
loss_function = "mae"
window_size = 50

'''
End of Initialization Code ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''



def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def gather_data(ticker = stock_ticker): # gathers data from yfin package, copies it into a dictionary to be later stored into the .csv
    necessary_data = yfin.download(tickers = ticker, start = f'{three_months_ago_today}', end = f'{today_date}') 

    result = {}
    result = necessary_data.copy()
    return result


def prepare_data(ticker = stock_ticker, n_steps = window_size, scale = scale, shuffle = True, lookup_step = day_outlook, split_by_date = True, test_size = 0.2, feature_columns = ['Volume', 'Open', 'High', 'Low', "Close", "Adj Close"]):

    gathered_data = gather_data()

    prepped_result = {}
    prepped_result["Stock_data"] = gathered_data.copy()

    if scale:
        Column_scalar = {}
        # scale the data (prices) from 0 to 1


        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            gathered_data[column] = scaler.fit_transform(np.expand_dims(gathered_data[column].values, axis=1)
            
            )
            Column_scalar[column] = scaler
        # add the MinMaxScaler instances to the result returned
        prepped_result["Column_scalar"] = Column_scalar

    # add the target column by shifting by value of `lookup_step`
    gathered_data['Future'] = gathered_data['Close'].shift(-lookup_step)

    # # last `lookup_step` columns contains NaN in future column
    # # get them before droping NaNs
    last_sequence = np.array(gathered_data[feature_columns].tail(lookup_step))
    # # drop NaNs
    gathered_data.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(gathered_data[feature_columns].values, gathered_data['Future'].values): 
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    # # add to result
    prepped_result['last_sequence'] = last_sequence

    # # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # # # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
    #     # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 
        
        - test_size) * len(X))
        prepped_result["X_train"] = X[:train_samples]
        prepped_result["Y_train"] = y[:train_samples]
        prepped_result["X_test"]  = X[train_samples:]
        prepped_result["Y_test"]  = y[train_samples:]
        if shuffle:
    #         # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(prepped_result["X_train"], prepped_result["Y_train"])
            shuffle_in_unison(prepped_result["X_test"], prepped_result["Y_test"])
    else:    
        # split the dataset randomly
        prepped_result["X_train"], prepped_result["X_test"], prepped_result["Y_train"], prepped_result["Y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # # # retrieve test features from the original dataframe
    prepped_result["Test_stock_data"] = prepped_result["Stock_data"]

    return prepped_result


def create_model(sequence_length = window_size, n_features = 6, units = 256, cell = LSTM, n_layers = 2, dropout = 0.4, loss = loss_function, optimizer = "adam", bidirectional = False):
    model = Sequential()
    for i in range(0, n_layers):
        if i == 0: # initial layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = True), batch_input_shape = (None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences = True, batch_input_shape = (None, sequence_length, n_features)))

        elif i == n_layers - 1: # final layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences =False)))
            else:
                model.add(cell(units, return_sequences = False))

        else: # all of the hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = True)))
            else:
                model.add(cell(units, return_sequences = True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation = "linear"))
    model.compile(loss = loss, metrics = ["mean_absolute_error"], optimizer = optimizer)
    return model


def plot_graph(test_df):
    matpl.plot(test_df[f'true_adjclose_{day_outlook}'], c='g')
    matpl.plot(test_df[f'adjclose_{day_outlook}'], c='r')
    matpl.xlabel("Days")
    matpl.ylabel("Price")
    matpl.legend(["Actual Price", "Predicted Price"])
    matpl.show()


'''
def get_final_df(model, data):
    # Rewrite much of this function
    print(data)
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if scale:
        Y_test = np.squeeze(data["Column_scalar"]["Adj Close"].inverse_transform(np.expand_dims(Y_test, axis=0)))
        y_pred = np.squeeze(data["Column_scalar"]["Adj Close"].inverse_transform(y_pred))

    test_df = data["Test_stock_data"]
    # add predicted future prices to the dataframe
    test_df[f"Adjclose_{day_outlook}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"True_adjclose_{day_outlook}"] = Y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit, 
                                    final_df["Adj Close"], 
                                    final_df[f"adjclose_{day_outlook}"], 
                                    final_df[f"true_adjclose_{day_outlook}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit, 
                                    final_df["Adj Close"], 
                                    final_df[f"adjclose_{day_outlook}"], 
                                    final_df[f"true_adjclose_{day_outlook}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df
'''





def predict(model, data):
    prediction_stats = [] # every piece of data we want will be in this array
    last_sequence = data["last_sequence"][-50:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    prediction = model.predict(last_sequence)

    predicted_high = data["Column_scalar"]["High"].inverse_transform(prediction)[0][0]
    predicted_low = data["Column_scalar"]["Low"].inverse_transform(prediction)[0][0]
    predicted_open = data["Column_scalar"]["Open"].inverse_transform(prediction)[0][0]
    predicted_close = data["Column_scalar"]["Close"].inverse_transform(prediction)[0][0]
    predicted_vol = data["Column_scalar"]["Volume"].inverse_transform(prediction)[0][0]
    if scale:
        predicted_price_adjclose = data["Column_scalar"]["Adj Close"].inverse_transform(prediction)[0][0]
    else:
        predicted_price_adjclose = prediction[0][0]

    prediction_stats.append(predicted_open)
    prediction_stats.append(predicted_high)
    prediction_stats.append(predicted_low)
    prediction_stats.append(predicted_close)
    prediction_stats.append(predicted_price_adjclose)
    prediction_stats.append(predicted_vol)

    return prediction_stats


'''
Driver Code ........................................................................................................................................................................
'''
data = prepare_data()
data["Stock_data"].to_csv(data_filename)

model_descriptor = f"STM32MP1-{today_date}-{loss_function}-adam-lstm"
model_file_name = os.path.join("results", f"STM-{today_date}.hdf5") 
model = create_model()

checkpoint = ModelCheckpoint(os.path.join("results", model_descriptor + ".hdf5"), save_weights_only = True, save_best_only = True, verbose = 0)

tensorboard = TensorBoard(log_dir = os.path.join("logs", model_descriptor)) # very optional for this project, but I'll keep it
history = model.fit(data["X_train"], data["Y_train"], batch_size = 64, epochs = 7, validation_data = (data["X_test"], data["Y_test"]), callbacks = [checkpoint, tensorboard], verbose = 1)


model.load_weights(f"results/{model_descriptor}.hdf5")

loss, mae = model.evaluate(data["X_test"], data["Y_test"], verbose = 1)

if scale:
    mean_absolute_error = data["Column_scalar"]["Adj Close"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# All stats that we want presented on candlestick
future_open = predict(model, data)[0]
future_high = predict(model, data)[1]
future_low = predict(model, data)[2]
future_close = predict(model, data)[3]
future_price_adjclose = predict(model, data)[4]
future_vol = int(predict(model, data)[5])

# # we calculate the accuracy by counting the number of positive profits
# accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)


# print prediction, candlestick stats
if day_outlook == 1:
    print('.........................................................................................')
    print(f"Final outcome for {stock_ticker} in {day_outlook} day:\n")
    print(f"Predicted open: ${future_open:.2f}")
    print(f"Predicted high: ${future_high:.2f}")
    print(f"Predicted low: ${future_low:.2f}")
    print(f"Predicted close: ${future_close:.2f}")
    print(f"Predicted price (based on Adj close): ${future_price_adjclose:.2f}")
    print(f"Predicted volume: {future_vol}")
else: 
    print('.........................................................................................')
    print(f"Final outcome for {stock_ticker} in {day_outlook} days:\n")
    print(f"Predicted open: ${future_open:.2f}")
    print(f"Predicted high: ${future_high:.2f}")
    print(f"Predicted low: ${future_low:.2f}")
    print(f"Predicted close: ${future_close:.2f}")
    print(f"Predicted price (based on Adj close): ${future_price_adjclose:.2f}")
    print(f"Predicted volume: {future_vol}")


# # plot true/predicted prices graph
# final_df = get_final_df(model, data)
# plot_graph(final_df)

'''
End of Driver Code ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''
