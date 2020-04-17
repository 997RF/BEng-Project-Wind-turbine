import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import eif as iso

###  Loading  dataset  ###
data = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/HalfYearFilteredNoNAN.csv")
data = data.tail(100000)
x = data['WindSpeed_mps']
y = data['Power_kW']

###  Create a two dimensional array with datatset  ###
z = np.array((list(zip(x, y))))

###  Create the dataframe  ###
new_data = pd.DataFrame(np.array(z), columns=['A', 'B'])

###  Fitting into Extended Isolation Forest Model  ###
anomalies_ratio = 0.02
eif = iso.iForest(new_data.values, ntrees=3000, sample_size=100, ExtensionLevel=0.9)
anomaly_scores = eif.compute_paths(X_in=new_data.values)
anomaly_scores_sorted = np.argsort(anomaly_scores)
indices_with_preds = anomaly_scores_sorted[-int(np.ceil(anomalies_ratio*new_data.shape[0])):]
outliers = np.zeros_like(y)
outliers[indices_with_preds] = 1

###  Getting the cleaned date from outliers  ###
x_cleaned = data[np.where(outliers != 1, True, False)]
x_cleaned.to_csv("EIF4.csv")

### Loading the created dataset  ###
data1 = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/EIF4.csv")
x1 = data1['WindSpeed_mps']
y1 = data1['Power_kW']

### Plotting the results  ###
plt.figure(figsize=(15, 12))
plt.scatter(x, y, s=3,  label="Original data")
plt.scatter(x1, y1, s=3, c='red',  label="After outlier removal")
plt.title("Power vs Wind speed ", weight='bold', size=20)
plt.xlabel("Wind speed, m/s",  weight='bold', size=16)
plt.ylabel("Power, kW",  weight='bold', size=16)
plt.savefig("EIF.png")
