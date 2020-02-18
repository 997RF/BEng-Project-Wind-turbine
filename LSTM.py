import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.preprocessing import StandardScaler

###  Import data  ###
df = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/afterisoforrest3.csv", index_col=None)
dfs = df[['Power_kW']] #A is windspeed and B is power
time = df['TimeStamp']
dates = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f')
print(dates[200])
df.TimeStamp = dates
timez = df['TimeStamp']

scaler = StandardScaler()
train_arr = scaler.fit_transform(dfs)

dataset = dfs.values
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = train_arr[0:train_size, :], train_arr[train_size:len(dataset), :]
time_test = timez[train_size:len(dataset)]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

look_back = 8
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

###  Fit data in LSTM model  ###
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=5, batch_size=1000,
                    verbose=1, validation_split=0.5, shuffle=False)

score = model.evaluate(testX, testY)
print('Score: {}'.format(score))


###  Generate Prediction  ###
yhat = model.predict(testX)
y_predicted = scaler.inverse_transform(yhat)
testY = testY.reshape(-1, 1)
y_test = scaler.inverse_transform(testY)

q = time_test[:15000]
w = y_predicted[:15000]
e = y_test[:15000]

###  Plot results  ###
plt.figure(figsize=(20, 8))
plt.plot(q, w,  label='Predicted values')
plt.plot(q, e,  label='Given values')
plt.legend()
plt.title(" LSTM Power prediction ")
plt.ylabel("Power, kW")
plt.savefig("LSTM.png")

### Print error results  ###
print("Mean squared error: %.3f" % mean_squared_error(testY, yhat))
print("Root mean squared error: %.3f" % sqrt(mean_squared_error(testY, yhat)))
print('Variance : %.3f' % r2_score(testY, yhat))
print("Mean absolute error: %.3f" % mean_absolute_error(testY, yhat))
