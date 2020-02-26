import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

# Data loading
df = pd.read_csv('afterisoforrest3.csv', index_col=None)

# Preparing input data
df = df[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC',
            'Pitch_Deg', 'NacelleOrientation_Deg', 'RotorSpeedAve', 'MeasuredYawError']]

test_data = pd.read_csv("PredictionTestData.csv", index_col=None)
# test_data = test_data.head(2592000)
time = test_data['TimeStamp']
test = test_data[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC',
            'Pitch_Deg', 'NacelleOrientation_Deg', 'RotorSpeedAve', 'MeasuredYawError']]

# Preparing label data
label = df['Power_kW']
labeltest = test_data[['Power_kW']]

dates = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f')
test_data.TimeStamp = dates
times = test_data[['TimeStamp']]

# conversion to numpy array
x, y = df.values, label.values
q = test.values
w = labeltest.values

# scaling values for model
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()

X_train = x_scale.fit_transform(x)
Y_train = y_scale.fit_transform(y.reshape(-1, 1))
X_test = x_scale.fit_transform(q)
Y_test = y_scale.fit_transform(w.reshape(-1, 1))
X_train = X_train.reshape((-1, 1, 10))
X_test = X_test.reshape((-1, 1, 10))

# time_test = timez[0:test_size]

# creating model using Keras
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(1, 10)))
model.add(LSTM(units=300, return_sequences=True))
model.add(LSTM(units=300, return_sequences=True))
model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])

model.fit(X_train, Y_train, batch_size=1000, epochs=25, validation_split=0.1, verbose=1)

score = model.evaluate(X_test, Y_test)
print('Score: {}'.format(score))

y_predicted = model.predict(X_test)
y_predicted = y_scale.inverse_transform(y_predicted)
y_test = y_scale.inverse_transform(Y_test)

plt.figure(figsize=(100, 12))
plt.plot(y_predicted, label='Predicted')
plt.plot(y_test, label='Measurements')
plt.legend()
plt.ylabel("Power, kW", size=30, weight='bold')
plt.title("LSTM Power prediction", size=30, weight='bold')
plt.savefig("LSTMP5.png")

print("Mean squared error: %.4f" % mean_squared_error(y_test, y_predicted))
print("Root mean squared error: %.4f" % sqrt(mean_squared_error(y_test, y_predicted)))
print('Variance : %.4f' % r2_score(y_test, y_predicted))
print("Mean absolute error: %.4f" % mean_absolute_error(y_test, y_predicted))
