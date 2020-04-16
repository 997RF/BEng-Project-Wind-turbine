import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

### Loading the data ###
dfs = pd.read_csv("HalfYearFilteredNoNAN.csv", index_col=None)

### Seperating the dependent and independent variables ###
features = dfs[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC',
            'Current1_Arms', 'Current2_Arms', 'Current3_Arms',
            'Volts1_Vrms', 'Volts2_Vrms', 'Volts3_Vrms',
            'Pitch_Deg', 'NacelleOrientation_Deg', 'RotorSpeedAve', 'MeasuredYawError']]
target = dfs['Power_kW']
x, y = features.values, target.values

### Training the model ###
train_img, test_img, train_lbl, test_lbl = train_test_split(x, y, test_size=1/7.0, random_state=0)
scaler =StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

### Plotting the resuls ###
pca = PCA(.999999999)
pca.fit(train_img)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Principal Component Analysis')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()
