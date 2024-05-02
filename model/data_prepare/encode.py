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
           'FireplaceQu','GarageQual','GarageCond', 'PoolQC']

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

#criando categorias

lotshape_mapping = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
# Substituir os valores categóricos pelos valores numéricos correspondentes
data_train['LotShape'] = data_train['LotShape'].replace(lotshape_mapping)
data_test['LotShape'] = data_test['LotShape'].replace(lotshape_mapping)

landslope_mapping = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
data_train['LandSlope'] = data_train['LandSlope'].replace(landslope_mapping)

bsmtexposure_mapping = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
data_train['BsmtExposure'] = data_train['BsmtExposure'].replace(bsmtexposure_mapping)

bsmtfintype_mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
data_train['BsmtFinType1'] = data_train['BsmtFinType1'].replace(bsmtfintype_mapping)
data_train['BsmtFinType2'] = data_train['BsmtFinType2'].replace(bsmtfintype_mapping)

garagefinish_mapping = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
data_train['GarageFinish'] = data_train['GarageFinish'].replace(garagefinish_mapping)

paveddrive_mapping = {'Y': 2, 'P': 1, 'N': 0}
data_train['PavedDrive'] = data_train['PavedDrive'].replace(paveddrive_mapping)

# Criar mapeamentos para as duas novas colunas
fence_privacy_mapping = {'GdPrv': 2, 'MnPrv': 1, 'GdWo': 0, 'MnWw': 0, 'NA': 0}
fence_wood_mapping = {'GdPrv': 0, 'MnPrv': 0, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
# Criar as novas colunas
data_train['Fence_Privacy'] = data_train['Fence'].replace(fence_privacy_mapping)
data_train['Fence_Wood'] = data_train['Fence'].replace(fence_wood_mapping)

#fim transformação ordinal 





"""
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


"""