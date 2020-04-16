import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATASET = 'dataset.xlsx'

# Config
TRAIN_PERCENT = 0.7
EPOCH = 20
LAYER = 5
n_input = 7
n_features = 1
VALIDATION = 0.3
DROPOUT = 0.3
DENSE = 1

# read data
data = pd.read_excel(DATASET)
data_end = int(np.floor(TRAIN_PERCENT * (data.shape[0])))
train = data[0:data_end]['so_don']
train = train.values.reshape(-1)
test = data[data_end:]['so_don'].values.reshape(-1)


def get_data(data_train, data_test, time_step, num_predict):
    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()

    for i in range(0, len(data_train) - time_step - num_predict):
        x_train.append(data_train[i:i + time_step])
        y_train.append(data_train[i + time_step:i + time_step + num_predict])

    for i in range(0, len(data_test) - time_step - num_predict):
        x_test.append(data_test[i:i + time_step])
        y_test.append(data_test[i + time_step:i + time_step + num_predict])

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(date_test)


x_train, y_train, x_test, y_test = get_data(train, test, n_input, 1)

# Standardize Data
scaler = MinMaxScaler()
x_train = x_train.reshape(-1, n_input)

x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

x_test = x_test.reshape(-1, n_input)

x_test = scaler.fit_transform(x_test)
y_test = scaler.fit_transform(y_test)

x_train = x_train.reshape(-1, n_input, 1)
y_train = y_train.reshape(-1, 1)

x_test = x_test.reshape(-1, n_input, 1)
y_test = y_test.reshape(-1, 1)

# create model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
model.add(Dropout(DROPOUT))

for n in range(LAYER - 2):
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(DROPOUT))

model.add(LSTM(units=50))
model.add(Dropout(DROPOUT))
model.add(Dense(DENSE))
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=EPOCH, validation_split=VALIDATION, verbose=1, batch_size=n_input)
model.save('model.h5')

# test
test_output = model.predict(x_test)
test_1 = scaler.inverse_transform(test_output)
test_2 = scaler.inverse_transform(y_test)
plt.plot(test_1[:100], color='r')
plt.plot(test_2[:100], color='b')
plt.legend(('prediction', 'reality'), loc='upper right')
plt.show()
