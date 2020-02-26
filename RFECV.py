import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

df = pd.read_csv("afterisoforrest3.csv", index_col=None)
df = df.head(1000)
dfs = df[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC', 'Pitch_Deg', 'NacelleOrientation_Deg',
          'RotorSpeedAve', 'MeasuredYawError', 'Volts1_Vrms', 'Volts2_Vrms', 'Volts3_Vrms', 'Current1_Arms',
          'Current2_Arms', 'Current3_Arms']]
y = dfs['Power_kW']
X = dfs.drop('Power_kW', axis=1)

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure(figsize=(10, 10))
plt.xlabel("Number of features selected", weight='bold', size=16)
plt.ylabel("Cross validation score", weight='bold', size=16)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.title("Recursive feature elimination with cross-validation", weight='bold', size=20)
plt.savefig("RFECV1.png")
