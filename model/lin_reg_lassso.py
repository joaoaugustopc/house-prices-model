import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

data_train = pd.read_csv('data/pre_data_train.csv')
data_test = pd.read_csv('data/pre_data_test.csv')

target = "SalePrice"

X_train = data_train.drop(["Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["Id"], axis=1).select_dtypes(include=["number"])

# Utilizando validação Cruzada para determinar o melhor hiperPâmetro (alpha) para o lasso 

param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]}

lasso = Lasso()

grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')

grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']

model = Lasso(alpha=best_alpha)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

padrao = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': y_pred})

padrao.to_csv('data/sample_submission_lasso.csv', index=False)







