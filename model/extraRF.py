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

from sklearn.ensemble import ExtraTreesRegressor

# Definindo os parâmetros para ajustar

param_grid = {
    'n_estimators': [20,50,100, 200, 400],
    'min_samples_leaf': [1, 2, 4, 5, 10],
    'min_impurity_decrease': np.linspace(0, 0.5, 10),
}

# Criando o modelo

extra = ExtraTreesRegressor(n_jobs=-1)

CV_Extra = GridSearchCV(estimator=extra, param_grid=param_grid, cv= 5)


CV_Extra.fit(X_train, y_train)


print(CV_Extra.best_params_)

extra = ExtraTreesRegressor(n_jobs=-1,oob_score=True, **CV_Extra.best_params_, bootstrap=True)
extra.fit(X_train, y_train)

y_pred = extra.predict(X_test)

print(extra.oob_score_)

importances = extra.feature_importances_

feature_names = X_train.columns

feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

feature_importances = feature_importances.sort_values(by="importance", ascending=False, ignore_index=True)

print(feature_importances)

result = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': y_pred})

result.to_csv('data/sample_submission_extraRF.csv', index=False)


"""
0.8537408222818466
          feature  importance
0     OverallQual    0.317427
1      GarageCars    0.159103
2       GrLivArea    0.080161
3        FullBath    0.068419
4     TotalBsmtSF    0.037521
5        1stFlrSF    0.036622
6    YearRemodAdd    0.028761
7       YearBuilt    0.027152
8      Fireplaces    0.025833
9      GarageArea    0.024263
10     BsmtFinSF1    0.022649
11       2ndFlrSF    0.018321
12    GarageYrBlt    0.018117
13   TotRmsAbvGrd    0.016215
14        LotArea    0.014115
15   BsmtFullBath    0.010977
16     MasVnrArea    0.009023
17   BedroomAbvGr    0.009010
18       HalfBath    0.008569
19    LotFrontage    0.007400
20    OverallCond    0.006711
21    OpenPorchSF    0.006481
22     MSSubClass    0.006410
23     WoodDeckSF    0.006157
24         MoSold    0.005426
25      BsmtUnfSF    0.005080
26    ScreenPorch    0.004088
27         YrSold    0.003556
28   KitchenAbvGr    0.003323
29       PoolArea    0.003249
30   BsmtHalfBath    0.002526
31     BsmtFinSF2    0.002409
32  EnclosedPorch    0.002375
33      3SsnPorch    0.001437
34   LowQualFinSF    0.000578
35        MiscVal    0.000535
"""