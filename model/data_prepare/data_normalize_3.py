import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Normalizar os dados com ordem e numéricos - não one hot encoder

data_train = pd.read_csv('dataset/train_encoded_imputed.csv')
data_test = pd.read_csv('dataset/test_encoded_imputed.csv') # colocar o nome do arquivo que contém os dados de teste

# Colunas binárias
cols_to_drop = data_train.filter(regex='onehotencoder').columns.tolist() + ["SalePrice","remainder__remainder__Id"]
print(cols_to_drop)

# Selecionar as colunas para normalizar
cols_to_scale = data_train.columns.difference(cols_to_drop)
scaler = StandardScaler()

# Normalizar as colunas selecionadas
data_train[cols_to_scale] = scaler.fit_transform(data_train[cols_to_scale])
data_test[cols_to_scale] = scaler.transform(data_test[cols_to_scale])  # Use o mesmo scaler para os dados de teste

data_train["remainder__Id"] = data_train["remainder__remainder__Id"].astype(int)
data_test["remainder__Id"] = data_test["remainder__remainder__Id"].astype(int)

data_train.to_csv('dataset/train_scaled.csv', index=False)
data_test.to_csv('dataset/test_scaled.csv', index=False)