import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


data_train = pd.read_csv('dataset/train_prep.csv')
data_test = pd.read_csv('dataset/test_prep.csv')
y_train = data_train['SalePrice']
data_train.drop(columns=['SalePrice'], inplace=True)

# Lista de variáveis categóricas que possuem uma ordem natural
list_ordinal = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual',
           'FireplaceQu','GarageQual','GarageCond']

list_lb = ['CentralAir'] # sim ou nao

list = list_ordinal + list_lb


pre_data_train = data_train.drop(columns=list)
pre_data_train = pre_data_train.select_dtypes(include=["object"]).columns

pre_data_test = data_test.drop(columns=list)
pre_data_test = pre_data_test.select_dtypes(include=["object"]).columns

data_train[list_ordinal] = data_train[list_ordinal].fillna('NA')
data_test[list_ordinal] = data_test[list_ordinal].fillna('NA')

# Transformar as variáveis categóricas em numéricas ordenadas
categories = [['NA','Po','Fa','TA','Gd','Ex']] * len(list_ordinal)

encoder = OrdinalEncoder(categories=categories)

data_train[list_ordinal] = encoder.fit_transform(data_train[list_ordinal])
data_test[list_ordinal] = encoder.transform(data_test[list_ordinal])


le = LabelEncoder()

for col in list_lb:
    data_train[col] = le.fit_transform(data_train[col].astype(str))
    data_test[col] = le.fit_transform(data_test[col].astype(str))


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), pre_data_train),
    remainder='passthrough'
)


# Aplicar a transformação aos dados de treinamento
data_train_encoded = column_trans.fit_transform(data_train)
data_test_encoded = column_trans.transform(data_test) 

#data_train_encoded_dense = data_train_encoded.toarray()

train = pd.DataFrame(data_train_encoded,columns=column_trans.get_feature_names_out())
test = pd.DataFrame(data_test_encoded,columns = column_trans.get_feature_names_out())

train['SalePrice'] = y_train


train.to_csv('data/train_prep_encoded.csv', index=False)
test.to_csv('data/test_prep_encoded.csv', index=False)

