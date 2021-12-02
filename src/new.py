
from datetime import date
from dateutil.relativedelta import *

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
import matplotlib.pyplot as matpl

'''
Initialization Code ..................................................................................................................................................................
'''

os.chdir('/home/stm32mp1/Robert/StockPrediction/src') # can now run in any directory on the terminal. This changes the pwd at runtime
basepath = os.getcwd()[:-4] # gets everything until /src

# creates these directories...
if not os.path.isdir(f"{basepath}/results"):
    os.mkdir(f"{basepath}/results")
if not os.path.isdir(f"{basepath}/logs"):
    os.mkdir(f"{basepath}/logs")
if not os.path.isdir(f"{basepath}/data"):
    os.mkdir(f"{basepath}/data")
if not os.path.isdir(f"{basepath}/deployments"):
    os.mkdir(f"{basepath}/deployments")

# set date, create a file with that date, and saves it
today_date = date.today()
three_months_ago_today = today_date + relativedelta(months = -3)
data_filename = os.path.join(f"{basepath}/data", f"STM-{today_date}.csv") 

# these variables can be altered to change the model
stock_ticker = "STM"
day_outlook = 1
scale = True
loss_function = "mae"
window_size = 50
epochs = 900
batch_size = 2
layers = 2
dropout_amount = 0.4
bidirectional_bool  = False

date_with_outlook = today_date + relativedelta(days = +day_outlook)


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


def prepare_data(n_steps = window_size, scale = scale, shuffle = True, lookup_step = day_outlook, split_by_date = True, test_size = 0.2, feature_columns = ["Volume", "Open", "High", "Low", "Close", "Adj Close"]):

    gathered_data = gather_data()
    
    
    #calculating average for defining the tensor value
    sum = 0
    count = 0
    for value in gathered_data["Adj Close"]:
        sum += value
        count += 1
    avg_adj_close = sum / count
    print("Average adj close:", avg_adj_close)

    prepped_result = {}
    prepped_result["Stock Data"] = gathered_data.copy()

    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            gathered_data[column] = scaler.fit_transform(np.expand_dims(gathered_data[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        prepped_result["Column Scaler"] = column_scaler

    gathered_data["Future"] = gathered_data["Close"].shift(-lookup_step)

    last_sequence = np.array(gathered_data[feature_columns].tail(lookup_step))

    gathered_data.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(gathered_data[feature_columns].values, gathered_data["Future"].values): 
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    prepped_result["last_sequence"] = last_sequence

    # X and Y (to be converted into numpy arrays)
    X, Y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        Y.append(target)

    X = np.array(X)
    Y = np.array(Y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        prepped_result["X_train"] = X[:train_samples]
        prepped_result["Y_train"] = Y[:train_samples]
        prepped_result["X_test"]  = X[train_samples:]
        prepped_result["Y_test"]  = Y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(prepped_result["X_train"], prepped_result["Y_train"])
            shuffle_in_unison(prepped_result["X_test"], prepped_result["Y_test"])
    else:    
        # split the dataset randomly
        prepped_result["X_train"], prepped_result["X_test"], prepped_result["Y_train"], prepped_result["Y_test"] = train_test_split(X, Y, test_size = test_size, shuffle = shuffle)

    # retrieve test features from the original dataframe
    prepped_result["Test_stock_data"] = prepped_result["Stock Data"]

    return prepped_result


def create_model(sequence_length = window_size, n_features = 6, units = 256, cell = LSTM, n_layers = layers, dropout = dropout_amount, loss = loss_function, optimizer = "adam", bidirectional = bidirectional_bool):
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


def predict(model, data):
    prediction_stats = [] # every piece of data we want will be in this array
    last_sequence = data["last_sequence"][-window_size:]
    last_sequence = np.expand_dims(last_sequence, axis = 0)

    prediction = model.predict(last_sequence)

    predicted_high = data["Column Scaler"]["High"].inverse_transform(prediction)[0][0]
    predicted_low = data["Column Scaler"]["Low"].inverse_transform(prediction)[0][0]
    predicted_open = data["Column Scaler"]["Open"].inverse_transform(prediction)[0][0]
    predicted_close = data["Column Scaler"]["Close"].inverse_transform(prediction)[0][0]
    predicted_vol = data["Column Scaler"]["Volume"].inverse_transform(prediction)[0][0]
    if scale:
        predicted_price_adjclose = data["Column Scaler"]["Adj Close"].inverse_transform(prediction)[0][0]
    else:
        predicted_price_adjclose = prediction[0][0]

    prediction_stats.append(predicted_open)
    prediction_stats.append(predicted_high)
    prediction_stats.append(predicted_low)
    prediction_stats.append(predicted_close)
    prediction_stats.append(predicted_price_adjclose)
    prediction_stats.append(predicted_vol)

    return prediction_stats


def to_tflite(keras_model):
    # convert to tflite model for deployment
    tflite_converter = tf.lite.TFLiteConverter.from_saved_model(keras_model)
    tflite_contents = tflite_converter.convert()
    with tf.io.gfile.GFile(f"{basepath}/deployments/STM-StockPrediction-{today_date}.tflite", "wb") as file: # where the .tflite deployment file will be
        file.write(tflite_contents)


'''
Driver Code ........................................................................................................................................................................
'''
data = prepare_data()
data["Stock Data"].to_csv(data_filename)

model_descriptor = f"STM32MP1-{today_date}-{loss_function}-adam-lstm"
model_file_name = os.path.join(f"{basepath}/results", f"STM-{today_date}.hdf5") 
model = create_model()
print("\n\n\n")

checkpoint = ModelCheckpoint(os.path.join(f"{basepath}/results", model_descriptor + ".hdf5"), save_weights_only = False, save_best_only = True, verbose = 0) # saves model every few checkpoints

tensorboard = TensorBoard(log_dir = os.path.join(f"{basepath}/logs", model_descriptor)) # very optional for this project, but I'll keep it
history = model.fit(data["X_train"], data["Y_train"], batch_size = batch_size, epochs = epochs, validation_data = (data["X_test"], data["Y_test"]), callbacks = [checkpoint, tensorboard], verbose = 1)

# All stats that we want presented on candlestick
future_open = predict(model, data)[0]
future_high = predict(model, data)[1]
future_low = predict(model, data)[2]
future_close = predict(model, data)[3]
future_price_adjclose = predict(model, data)[4]
future_vol = int(predict(model, data)[5])


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


saved_model_location = f"{basepath}/saved_model" 
model.save(saved_model_location, save_format="tf")
to_tflite(saved_model_location)

'''
End of Driver Code ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''
