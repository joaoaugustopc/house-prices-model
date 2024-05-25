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
param_grid = {'min_samples_leaf': np.arange(1, 21),
              'min_impurity_decrease': np.linspace(0, 0.5, 20),
              }

# Criando o modelo
tree_reg = DecisionTreeRegressor()

# Criando o objeto de busca em grade
grid_tree = GridSearchCV(tree_reg, param_grid, cv=10, scoring='neg_mean_squared_error')

# Ajustando o modelo
grid_tree.fit(X_train, y_train)

print(grid_tree.best_params_)


# Treinando o modelo com a profundidade ideal
tree_reg = DecisionTreeRegressor(**grid_tree.best_params_)
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)

print (tree_reg.feature_importances_)

result = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': y_pred})

result.to_csv('data/sample_submission_tree2.csv', index=False)

# Exportando o modelo como um arquivo .dot
tree.export_graphviz(tree_reg, out_file='dot/tree2.dot', feature_names=X_train.columns) 

# dot -Tpdf dot/tree4.dot -o Decision_Tree/tree4.pdf

"""
{'max_leaf_nodes': 1460, 'min_samples_leaf': 20}
[0.00000000e+00 6.78319295e-04 1.34898833e-03 7.63214504e-01
 4.46289839e-03 5.76543950e-03 2.33831601e-03 2.96548094e-04
 1.90864610e-02 0.00000000e+00 4.19949728e-04 3.52346293e-02
 1.49327183e-02 6.81113513e-04 0.00000000e+00 1.22170933e-01
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 6.91171890e-03 0.00000000e+00 0.00000000e+00
 0.00000000e+00 1.11843427e-02 1.09307951e-02 0.00000000e+00
 1.56389604e-04 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 1.85934479e-04 0.00000000e+00]
 
"""

# O que mais contribui para a tomada de decisão é o tamanho do lote 