import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

### Loading the data ###
df = pd.read_csv("HalfYearFilteredNoNAN.csv", index_col=None)
df = df.head(1000)

### Seperating the dependent and independent variables ###
dfs = df[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC', 'Pitch_Deg', 'NacelleOrientation_Deg',
          'RotorSpeedAve', 'MeasuredYawError', 'Volts1_Vrms', 'Volts2_Vrms', 'Volts3_Vrms', 'Current1_Arms',
          'Current2_Arms', 'Current3_Arms']]
y = dfs['Power_kW']
X = dfs.drop('Power_kW', axis=1)

#### Training the model ###
rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X, y)

### Computing the importance of each feature ###
importan = rfecv.estimator_.feature_importances_

dset = pd.DataFrame()
dset['attr'] = X.columns[0:13]
dset['importance'] = 1/(rfecv.estimator_.feature_importances_)
dset = dset.sort_values(by='importance', ascending=False)

### Plotting the resuls ###
plt.figure(figsize=(10,8))
plt.bar(dset['attr'],dset['importance'])
plt.title('RFECV - Feature Importances', fontsize=16, fontweight='bold')
plt.ylabel('Importance score', fontsize=14, )
plt.xticks(rotation=90)
plt.savefig('RFECV FINAL importances',bbox_inches='tight')
plt.show()

