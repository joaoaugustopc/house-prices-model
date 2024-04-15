from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importando os dados

data_train = pd.read_csv('data/pre_data_train.csv')
data_test = pd.read_csv('data/pre_data_test.csv')

target = "SalePrice"

X_train = data_train.drop(["Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["Id"], axis=1).select_dtypes(include=["number"])

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

feature_importances = reg.feature_importances_

print("Predicted house prices: ", feature_importances)

print("Predicted house prices: ", reg.score(X_train, y_train))

#relação entre as features e a variável target
sorted_idx = np.argsort(feature_importances)
pos = np.arange(sorted_idx.shape[0]) + 0.01
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importances[sorted_idx], align="center")
plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

plt.show()