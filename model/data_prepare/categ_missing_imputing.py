import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import sklearn
import pickle

data_train_imputed = pd.read_csv('dataset/train_imputed_KNeighbors.csv')
data_test_imputed = pd.read_csv('dataset/test_imputed_KNeighbors.csv')

data_train = pd.read_csv('dataset/train.csv')
data_test = pd.read_csv('dataset/test.csv')

data_train.loc[data_train['MasVnrType'] == "None"] = 'NoMas'
data_nums = data_train.select_dtypes(include=["number"], exclude=["object"])

missing_data_nums_train = data_nums.isnull().sum().sort_values(ascending=False) / data_nums.shape[0]

list_missing_cols = missing_data_nums_train[missing_data_nums_train != 0.0].index

print("Missing values in train data")
print(list_missing_cols)

for col in list_missing_cols:
    name = 'remainder__'+col
    data_train[col] = data_train_imputed[name]


data_nums = data_test.select_dtypes(include=["number"], exclude=["object"])

missing_data_nums_test = data_nums.isnull().sum().sort_values(ascending=False) / data_nums.shape[0]

list_missing_cols = missing_data_nums_test[missing_data_nums_test != 0.0].index

print("Missing values in test data")
print(list_missing_cols)

for col in list_missing_cols:
    name = 'remainder__'+col
    data_test[col] = data_test_imputed[name]


data_missing = data_train.isnull().sum().sort_values(ascending=False) / data_train.shape[0]

print("Missing values in train data")
print(data_missing[data_missing != 0.0])

# variaveis com valores faltantes nos dados de treinamento : Electrical, MasVnrType

# electrical 

categorias = [['Mix','FuseP','FuseF','FuseA','SBrkr']]

encoder = OrdinalEncoder(categories=categorias, handle_unknown='use_encoded_value', unknown_value=-1)

data_train['Electrical'] = encoder.fit_transform(data_train[['Electrical']])

# Proximos passos:
# 1. pegar variaveis faltantes do teste 
# 2 aplicar OrdinalEncoder 
# classificar os valores faltantes ajustado no treinamento


"""
categorias = [['Stone','NoMas','CBlock','BrkFace', 'BrkCmn']]

encoder = OrdinalEncoder(categories=categorias, handle_unknown='use_encoded_value', unknown_value=-1)

data_train['MasVnrType'] = encoder.fit_transform(data_train[['MasVnrType']])

data_train.to_csv('dataset/teste.csv', index=False)
"""

# MasVnrType : Impossivel separar NOne e NAN !!!!!!!!!!!!!!!!!!!!!!









