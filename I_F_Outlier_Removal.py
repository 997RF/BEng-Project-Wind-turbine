import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

#import the data
data = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/Power_without_zero_and_nan.csv")
x = data['WindSpeed_mps']
y = data['Power_kW']

#Create a two dimensional array with datatset
z = np.array((list(zip(x, y))))

#create the dataframe
new_data = pd.DataFrame(np.array(z), columns=['A', 'B'])


#isolation forrest algorithm
iso_forrest = IsolationForest(n_estimators=10, contamination=0.1, behaviour="new", random_state=0)
iso_forrest.fit(new_data)
outliers = iso_forrest.predict(new_data)

#Getting the cleaned date from outliers
x_cleaned = new_data[np.where(outliers == 1, True, False)]
x_cleaned.to_csv("afterisoforrest.csv")

#Plotting the results
print(plt.rcParams.get('figure.figsize'))
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.scatter(x_cleaned.iloc[:, 0], x_cleaned.iloc[:, 1], c='blue')
plt.title("Power vs Wind speed ")
plt.xlabel("Wind speed, m/s")
plt.ylabel("Power, kW")
plt.savefig("isoforest7.jpg")

