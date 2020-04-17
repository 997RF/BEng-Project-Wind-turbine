import numpy as np
import pandas as pd

### Import data ###
data = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/COMBI_Replaceding.csv")
x = data['WindSpeed_mps']

### Determine mean and std ###
data_std = float(np.std(x))
data_mean = float(np.mean(x))
anomaly_cut_off = float(data_std * 3)

### Determine the cut-off limits ###
lower_limit = data_mean - anomaly_cut_off
upper_limit = data_mean + anomaly_cut_off

### Remove lines that contain data outside the limits ###
indexNames = data[(data['WindSpeed_mps'] > upper_limit) | (data['WindSpeed_mps'] < lower_limit)].index
data.drop(indexNames, inplace=True)
data.to_csv("STDDEVwinspeed.csv")
