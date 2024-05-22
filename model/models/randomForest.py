import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data_train = pd.read_csv('dataset/train_scaled.csv')
data_test = pd.read_csv('dataset/test_scaled.csv')

target = "SalePrice"

X_train = data_train.drop(["remainder__remainder__Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["remainder__remainder__Id"], axis=1).select_dtypes(include=["number"])

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Definindo os parâmetros para ajustar
param_grid = {
    'n_estimators': [100, 200, 400],
    #'max_features': ['None','auto', 'sqrt', 'log2'],
    #'min_samples_leaf': [1, 2, 4, 5, 10],
    #'min_impurity_decrease': np.linspace(0, 0.5, 10),
}

# Criando o modelo
rf = RandomForestRegressor(n_jobs=-1)

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)


CV_rf.fit(X_train, y_train)


print(CV_rf.best_params_)



rf = RandomForestRegressor(oob_score=True, n_jobs = -1, **CV_rf.best_params_)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(rf.oob_score_)

importances = rf.feature_importances_

feature_names = X_train.columns

feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

feature_importances = feature_importances.sort_values(by="importance", ascending=False, ignore_index=True)

print(feature_importances)

result = pd.DataFrame({'Id': data_test['remainder__remainder__Id'], 'SalePrice': y_pred})

result['Id'] = result['Id'].astype(int) 

result.to_csv('submissions/sample_submission_rf_scaled.csv', index=False)


# Submissão 1 : {'min_samples_leaf': 1, 'n_estimators': 200} ----  0.8605659089358756
"""
{'n_estimators': 400}
0.8630449709279657
          feature  importance
0     OverallQual    0.583594
1       GrLivArea    0.112302
2     TotalBsmtSF    0.043394
3        2ndFlrSF    0.039413
4      BsmtFinSF1    0.030586
5        1stFlrSF    0.025521
6      GarageCars    0.022248
7      GarageArea    0.019294
8         LotArea    0.017690
9       YearBuilt    0.013234
10    LotFrontage    0.009266
11   YearRemodAdd    0.008977
12   TotRmsAbvGrd    0.007561
13      BsmtUnfSF    0.006756
14     MasVnrArea    0.006707
15    OpenPorchSF    0.006308
16    OverallCond    0.005956
17    GarageYrBlt    0.005816
18     WoodDeckSF    0.005811
19       FullBath    0.005342
20         MoSold    0.004452
21     Fireplaces    0.004341
22     MSSubClass    0.002563
23         YrSold    0.002049
24   BedroomAbvGr    0.001964
25    ScreenPorch    0.001440
26   KitchenAbvGr    0.001306
27   BsmtFullBath    0.001250
28  EnclosedPorch    0.001158
29       HalfBath    0.001018
30     BsmtFinSF2    0.000968
31       PoolArea    0.000662
32      3SsnPorch    0.000432
33   BsmtHalfBath    0.000317
34   LowQualFinSF    0.000165
35        MiscVal    0.000140
"""