import pandas as pd
import numpy as np

# variáveis categóricas que possui valores faltantes sem explicação : 
# MasVnrType: Masonry veneer type
# MasVnrArea: Masonry veneer area in square feet ----> Regrediu alguns valores negativos para alguns NA, levando a crer que não possui masonry veneer (MasVnrType = None)
# LotFrontage: Linear feet of street connected to property ok 
# Electrical: Electrical system
# Mszoning : Identifies the general zoning classification of the sale. ( TESTE APENAS )
# KitchenQual: Kitchen quality ( TESTE APENAS )


data_train = pd.read_csv('dataset/train_encoded.csv')
data_test = pd.read_csv('dataset/test_encoded.csv')

data_nums = data_train.select_dtypes(include=["number"], exclude=["object"])

missing_data_nums = data_nums.isnull().sum().sort_values(ascending=False) / data_train.shape[0]

print("Missing values in train data")
print(missing_data_nums[missing_data_nums != 0.0])

"""Missing values in train data
LotFrontage    0.177397
GarageYrBlt    0.055479  NA pois nao tem garagem - > substituir por 0
MasVnrArea     0.005479
dtype: float64"""

# Imputing missing values

# LotFrontage: Linear feet of street connected to property
# Como a variável é uma variável numérica, vamos preencher os valores com uma regressão linear

from sklearn.linear_model import LinearRegression

# Lista de colunas com valores faltantes
list_missing_cols = missing_data_nums[missing_data_nums != 0.0].index
list_missing_cols = list_missing_cols.drop('remainder__GarageYrBlt')

data_train['remainder__GarageYrBlt'] = data_train['remainder__GarageYrBlt'].fillna(0)

# Separar os dados em treino e teste.
# Treino: Linhas que não possuem valores faltantes em nenhuma das colunas do list_missing_cols
# Teste: Linhas que possuem valores faltantes em pelo menos uma das colunas do list_missing_cols


for col in list_missing_cols:
    train = data_train[data_train[col].notna()]
    test = data_train[data_train[col].isna()]
    
    X_train = train.drop(columns=list_missing_cols)
    y_train = train[col]

    X_test = test.drop(columns=list_missing_cols)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    data_train.loc[data_train[col].isna(), col] = y_pred 

data_nums = data_train.select_dtypes(include=["number"], exclude=["object"])

missing_data_nums = data_nums.isnull().sum().sort_values(ascending=False) / data_train.shape[0]

print("Missing values in train data")
print(missing_data_nums[missing_data_nums != 0.0])


data_train.to_csv('dataset/train_imputed.csv', index=False)

# Para o conjunto de teste, vamos fazer a mesma coisa
"""
data_nums = data_test.select_dtypes(include=["number"], exclude=["object"])

missing_data_nums = data_nums.isnull().sum().sort_values(ascending=False) / data_test.shape[0]

print("Missing values in test data")
print(missing_data_nums[missing_data_nums != 0.0])

"""
"""Missing values in test data"""
# LotFrontage    0.155586
# GarageYrBlt    0.053461  NA significa que nao tem garagem 
# MasVnrArea     0.010281
# BsmtHalfBath   0.001371  Na significa que nao tem porao
# BsmtFullBath   0.001371  Na significa que nao tem porao
# GarageArea     0.000685 
# GarageCars     0.000685
# TotalBsmtSF    0.000685 Na significa que nao tem porao
# BsmtUnfSF      0.000685 Na significa que nao tem porao
# BsmtFinSF2     0.000685  Na significa que nao tem porao
# BsmtFinSF1     0.000685  Na significa que nao tem porao 

"""
list_missing_cols = missing_data_nums[missing_data_nums != 0.0].index

list_missing_cols = list_missing_cols.drop(['remainder__GarageYrBlt', 'remainder__BsmtHalfBath', 'remainder__BsmtFullBath', 'remainder__BsmtFinSF2', 'remainder__BsmtFinSF1'])

data_train['remainder__GarageYrBlt'] = data_train['remainder__GarageYrBlt'].fillna(0)
data_train['remainder__BsmtHalfBath'] = data_train['remainder__BsmtHalfBath'].fillna(0)
data_train['remainder__BsmtFullBath'] = data_train['remainder__BsmtFullBath'].fillna(0)
data_train['remainder__BsmtFinSF2'] = data_train['remainder__BsmtFinSF2'].fillna(0)
data_train['remainder__BsmtFinSF1'] = data_train['remainder__BsmtFinSF1'].fillna(0)


for col in list_missing_cols:
    train = data_test[data_test[col].notna()]
    test = data_test[data_test[col].isna()]
    
    X_train = train.drop(columns=list_missing_cols)
    y_train = train[col]

    X_test = test.drop(columns=list_missing_cols)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    data_test.loc[data_test[col].isna(), col] = y_pred
"""
#data_test.to_csv('dataset/test_imputed.csv', index=False)



"""
train = data_train[data_train[list_missing_cols].notna().all(axis=1)]
test = data_train[data_train[list_missing_cols].isna().any(axis=1)]

for col in list_missing_cols:
    X_train = train.drop(columns=list_missing_cols)
    y_train = train[col]
    
    X_test = test.drop(columns=list_missing_cols)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    data_train.loc[data_train[col].isna(), col] = y_pred
"""



