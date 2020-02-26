import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv("afterisoforrest3.csv", index_col=None)
df = df.head(10000)
dfs = df[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC', 'Pitch_Deg', 'NacelleOrientation_Deg',
          'RotorSpeedAve', 'MeasuredYawError', 'Volts1_Vrms', 'Volts2_Vrms', 'Volts3_Vrms', 'Current1_Arms',
          'Current2_Arms', 'Current3_Arms']]
y = dfs['Power_kW']
X = dfs.drop('Power_kW', axis=1)

extra_tree_forest = ExtraTreesClassifier(n_estimators=5, criterion='entropy', max_features=2)
extra_tree_forest.fit(X, y)
feature_importance = extra_tree_forest.feature_importances_
feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                        extra_tree_forest.estimators_],
                                       axis=0)
indices = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(10, 10))
plt.bar(X.columns, feature_importance[indices], yerr=feature_importance_normalized[indices], align="center")
plt.ylabel('Feature Importances',  weight='bold', size=16)
plt.xticks(rotation=90)
plt.title('Comparison of different Feature Importances',  weight='bold', size=16)
plt.savefig("ExtraTrees.png", bbox_inches='tight')
