import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#normalizar os dados com ordem e numericos - StandardScaler, sem ser one hot encoder

data_train = pd.read_csv('dataset/train.csv')
data_test = pd.read_csv('dataset/test.csv')

X_train = data_train.drop(columns=['SalePrice','Id', ])
X_train = X_train.select_dtypes(include=["number"], exclude=["object"])

#colunas binárias
cols_to_drop = data_train.filter(regex='onehotencoder').columns
print(cols_to_drop)
data_train = data_train.drop(columns=cols_to_drop)
# Fazer o mesmo para data_test
cols_to_drop = data_test.filter(regex='onehotencoder').columns
data_test = data_test.drop(columns=cols_to_drop)

X_test = data_test.drop(columns=['Id'])
X_test = X_test.select_dtypes(include=["number"], exclude=["object"])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

data_train[X_train.columns] = X_train_scaled_df
data_test[X_test.columns] = X_test_scaled_df

print(data_train[X_train.columns].mean())
print(data_train[X_train.columns].std())


data_train.to_csv('dataset/train_scaled.csv', index=False)
data_test.to_csv('dataset/test_scaled.csv', index=False)

