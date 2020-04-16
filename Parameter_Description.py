import pandas as pd
import matplotlib.pyplot as plt

### Load data ###
data = pd.read_csv("C:/Users/Reinis Fisers/PycharmProjects/TF_TEST/HalfYearFilteredNoNAN.csv") # testing data]
parameter = data[['WindSpeed_mps']]

### Print paramter desccription and histogram of values ###
print(parameter.describe())
histogramma = data.hist(column='WindSpeed_mps', bins=60, grid=False, figsize=(12, 8), rwidth=0.9)
histogramma = histogramma[0]
for x in histogramma:
    x.set_title("Wind Speed", weight='bold', size=20)
    x.set_xlabel("Speed, m/s", labelpad=20, weight='bold', size=18)
    x.set_ylabel("Samples", labelpad=20, weight='bold', size=18)
plt.show()
