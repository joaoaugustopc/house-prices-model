import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

# Lista de variáveis categóricas que possuem uma ordem natural
list_lb = ['Street','ExterQual','ExterCond','BsmtQual','BsmtCond',
           'BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual',
           'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence']

pre_data_train = data_train.drop(columns=list_lb)
pre_data_train = pre_data_train.select_dtypes(include=["object"]).columns

pre_data_test = data_test.drop(columns=list_lb)
pre_data_test = pre_data_test.select_dtypes(include=["object"]).columns

# Transformar as variáveis categóricas em numéricas ordenadas
le = LabelEncoder()

for col in list_lb:
    data_train[col] = le.fit_transform(data_train[col].astype(str))
    data_test[col] = le.fit_transform(data_test[col].astype(str))


column_trans = make_column_transformer(
    (OneHotEncoder(), pre_data_train),
    remainder='passthrough'
)

column_trans_test = make_column_transformer(
    (OneHotEncoder(), pre_data_test),
    remainder='passthrough'
)

# Aplicar a transformação aos dados de treinamento
data_train_encoded = column_trans.fit_transform(data_train)
data_test_encoded = column_trans_test.fit_transform(data_test) 

#data_train_encoded_dense = data_train_encoded.toarray()

train = pd.DataFrame(data_train_encoded,columns=column_trans.get_feature_names_out())
test = pd.DataFrame(data_test_encoded,columns = column_trans_test.get_feature_names_out())


train.to_csv('data/train_encoded.csv', index=False)
test.to_csv('data/test_encoded.csv', index=False)

