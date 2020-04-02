import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

# Loading the data
df = pd.read_csv("HalfYearFilteredNoNAN.csv", index_col=None)
df = df.head(200000)
# Seperating the dependent and independent variables
dfs = df[['Power_kW', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'AmbTemp_DegC', 'Pitch_Deg', 'NacelleOrientation_Deg',
          'RotorSpeedAve', 'MeasuredYawError', 'Volts1_Vrms', 'Volts2_Vrms', 'Volts3_Vrms', 'Current1_Arms',
          'Current2_Arms', 'Current3_Arms']]
y = dfs['Power_kW']
X = dfs.drop('Power_kW', axis=1)

extra_tree_forest = ExtraTreesClassifier(n_estimators=5, criterion='entropy', max_features=2)

# Training the model
extra_tree_forest.fit(X, y)

# Computing the importance of each feature
feature_importance = extra_tree_forest.feature_importances_

# Normalizing the individual importances
feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                        extra_tree_forest.estimators_],
                                       axis=0)
indices = np.argsort(feature_importance_normalized)[::-1]

# Plotting a Bar Graph to compare the models
plt.figure(figsize=(10,8))
plt.bar(X.columns[indices], feature_importance_normalized[indices], align="center")
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Comparison of Feature Importances', weight='bold', size=16)
plt.savefig("Final Extra Trees", bbox_inches='tight')
