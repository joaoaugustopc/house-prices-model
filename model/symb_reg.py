import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn import set_config
from sklearn.model_selection import GridSearchCV

data_train = pd.read_csv('data/pre_data_train.csv')
data_test = pd.read_csv('data/pre_data_test.csv')


target = "SalePrice"

X_train = data_train.drop(["Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["Id"], axis=1).select_dtypes(include=["number"])

#define operacoes protegidas
def _protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x2) > 0.00001, x1 / x2, 1.)
        result = np.where(np.isnan(result), 1., result)
        result = np.where(np.isinf(result), 1., result)
    return result

def _protected_sqrt(x1):
    with np.errstate(invalid='ignore'):
        result = np.where(x1 >= 0, np.sqrt(x1), x1)
        result = np.where(np.isnan(result), x1, result)
        result = np.where(np.isinf(result), x1, result)
    return result

def _protected_log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x1 > 0, np.log(x1), x1)
        result = np.where(np.isnan(result), x1, result)
        result = np.where(np.isinf(result), x1, result)
    return result

protected_div = make_function(function=_protected_div, name='protected_div', arity=2)
protected_sqrt = make_function(function=_protected_sqrt, name='protected_sqrt', arity=1)
protected_log = make_function(function=_protected_log, name='protected_log', arity=1)

function_set = ['add', 'sub', 'mul', 'cos', 'sin', 'tan', protected_div, protected_sqrt, protected_log]

est_gp = SymbolicRegressor(population_size=1000,generations= 50, tournament_size= 6, function_set=function_set,metric='mse', verbose= 1,p_crossover=0.9, init_depth=(5, 10),p_point_mutation=0.01, p_subtree_mutation=0.01, p_hoist_mutation=0.01,parsimony_coefficient=1000.0)

est_gp.fit(X_train, y_train)

y_pred = est_gp.predict(X_test)

result = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': y_pred})

result.to_csv('data/sample_submission_gp.csv', index=False)