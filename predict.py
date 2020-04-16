from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

model = load_model('model.h5')


input = list([98784, 53455, 26354, 12345, 31975, 9999999999, 13846])
n_input = len(input)
input = np.asarray(input)

# Standardize Data
scaler = MinMaxScaler()

input = input.reshape(-1, 1)
input = scaler.fit_transform(input)

input = input.reshape(-1, n_input, 1)

output = model.predict(input)
print(scaler.inverse_transform(output))
