import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

###  Import the data  ###
data = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/PredictionTestDataWithAverages.csv")
x = data['WindSpeed_mps']
y = data['Power_kW']

###  Create a two dimensional array with datatset  ###
z = np.array((list(zip(x, y))))

###  Create the dataframe  ###
new_data = pd.DataFrame(np.array(z), columns=['A', 'B'])

###  Isolation forrest algorithm  ###
iso_forrest = IsolationForest(n_estimators=100, contamination=0.01, )
iso_forrest.fit(new_data)
outliers = iso_forrest.predict(new_data)

###  Getting the cleaned date from outliers  ###
x_cleaned = data[np.where(outliers == 1, True, False)]
x_cleaned.to_csv("Testafterisoforrestwithaverages.csv")

data1 = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/Testafterisoforrestwithaverages.csv")
x1 = data1['WindSpeed_mps']
y1 = data1['Power_kW']

### Plotting the results  ###
plt.figure()
plt.scatter(x, y, s=2,  label="Original data")
plt.scatter(x1, y1, s=2, c='red',  label="After outlier removal")
plt.title("Power vs Wind speed ", weight='bold', size=20)
plt.xlabel("Wind speed, m/s",  weight='bold', size=16)
plt.ylabel("Power, kW",  weight='bold', size=16)
plt.show()
