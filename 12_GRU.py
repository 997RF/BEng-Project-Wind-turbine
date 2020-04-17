import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


### Preparing input data ###
df = pd.read_csv('afterisoforrest3.csv', index_col=None)
df = df[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC',
            'Pitch_Deg', 'NacelleOrientation_Deg', 'RotorSpeedAve', 'MeasuredYawError']]
test_data = pd.read_csv("Testafterisoforrestwithaverages.csv", index_col=None)
test_data = test_data.head(864000)
time = test_data['TimeStamp']
test = test_data[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC',
            'Pitch_Deg', 'NacelleOrientation_Deg', 'RotorSpeedAve', 'MeasuredYawError']]


### Preparing label data ###
label = df['Power_kW']
labeltest = test_data[['Power_kW']]

### Conversion to a numpy array ###
x, y = df.values, label.values
q = test.values
w = labeltest.values

### Scaling values for the model ###
x_scale = MinMaxScaler()
y_scale = MinMaxScaler()
X_train = x_scale.fit_transform(x)
Y_train = y_scale.fit_transform(y.reshape(-1, 1))
X_test = x_scale.fit_transform(q)
Y_test = y_scale.fit_transform(w.reshape(-1, 1))
X_train = X_train.reshape((-1, 1, 10))
X_test = X_test.reshape((-1, 1, 10))

### Creating a Keras model ###
model = Sequential()
model.add(GRU(units=30, return_sequences=True, input_shape=(1, 10)))
model.add(GRU(units=100, return_sequences=True))
model.add(GRU(units=100, return_sequences=True))
model.add(GRU(units=30))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])

### Fitting the data to the model ###
model.fit(X_train, Y_train, batch_size=10000, epochs=5, validation_split=0.1, verbose=0)

### Scoring the model performance ###
score = model.evaluate(X_test, Y_test, verbose=0)
print('Score for 10 days: {}'.format(score))

### Calculating the R2 score ###
y_predicted = model.predict(X_test)
y_predicted = y_scale.inverse_transform(y_predicted)
y_test = y_scale.inverse_transform(Y_test)
print('Variance: %.6f' % r2_score(y_test, y_predicted))
