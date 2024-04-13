import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data_train = pd.read_csv('data/pre_data_train.csv')
data_test = pd.read_csv('data/pre_data_test.csv')

target = "SalePrice"

X_train = data_train.drop(["Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["Id"], axis=1).select_dtypes(include=["number"])

from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Definindo os parâmetros para a busca em grade
param_grid = {'max_depth': np.arange(1, 21)}

# Criando o modelo
tree_reg = DecisionTreeRegressor()

# Criando o objeto de busca em grade
grid_tree = GridSearchCV(tree_reg, param_grid, cv=10, scoring='neg_mean_squared_error')

# Ajustando o modelo
grid_tree.fit(X_train, y_train)

# Imprimindo a profundidade ideal
print("Profundidade ideal: ", grid_tree.best_params_)

# Treinando o modelo com a profundidade ideal
tree_reg = DecisionTreeRegressor(max_depth=grid_tree.best_params_['max_depth'])
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)

result = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': y_pred})

result.to_csv('data/sample_submission_tree.csv', index=False)

# Exportando o modelo como um arquivo .dot
tree.export_graphviz(tree_reg, out_file='tree.dot', feature_names=X_train.columns)