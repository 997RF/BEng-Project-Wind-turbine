import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("HalfYearFilteredNoNAN.csv")
x = data['WindSpeed_mps']
y = data['Power_kW']
z = np.array((list(zip(x, y))))
new_data = pd.DataFrame(np.array(z), columns=['A', 'B'])
kmeans = KMeans(n_clusters=2)
kmeans.fit(z)
distance = kmeans.transform(z)
sorted_idx = np.argsort(distance.ravel())[::-1][:5000]
x_cleaned = np.delete(z, sorted_idx, axis=0)
for q in range(20):
    kmeans.fit(x_cleaned)
    distance = kmeans.transform(x_cleaned)
    sorted_idx = np.argsort(distance.ravel())[::-1][:5000]
    x_cleaned = np.delete(x_cleaned, sorted_idx, axis=0)

plt.figure(figsize=(15, 12))
plt.scatter(x, y, s=3,  label="Original")
plt.scatter(x_cleaned[:, 0], x_cleaned[:, 1], s=3, c='red',  label="After outlier removal")
plt.title("K-MEANS Power vs Wind speed ", weight='bold')
plt.xlabel("Wind speed, m/s",  weight='bold')
plt.ylabel("Power, kW",  weight='bold')
plt.legend()
plt.savefig("KMEANS.png")
