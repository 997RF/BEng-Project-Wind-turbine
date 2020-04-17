import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

### Selecting data ###
data = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/HalfYearFilteredNoNAN.csv")
data2 = data[['Power_kW', 'WindSpeed_mps', 'WindSpeed1','WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC',
              'Pitch_Deg', 'NacelleOrientation_Deg', 'RotorSpeedAve', 'MeasuredYawError','Current1_Arms', 'Volts1_Vrms']]

### Plotting the parameter correlation heatmap ###
f, ax = plt.subplots(figsize=(10,10))
matrix = np.triu(data2.corr())
heatmap = sb.heatmap(data2.corr(), mask=matrix, annot=True, linewidths=1, cmap='coolwarm', ax=ax)
heatmap.set_title("Parameter correlation heatmap",  size=16)
plt.show()
