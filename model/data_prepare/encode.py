import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


data_train = pd.read_csv('dataset/train.csv')
data_test = pd.read_csv('dataset/test.csv')


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


#criando categorias

list_categories = ['LotShape','LandSlope','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','PavedDrive','Fence']
data_train[list_categories] = data_train[list_categories].fillna('NA')
data_test[list_categories] = data_test[list_categories].fillna('NA')

from sklearn.preprocessing import OrdinalEncoder

# Criar listas de categorias
lotshape_categories = [['IR3', 'IR2', 'IR1', 'Reg']]
landslope_categories = [['Sev', 'Mod', 'Gtl']]
bsmtexposure_categories = [['NA', 'No', 'Mn', 'Av', 'Gd']]
bsmtfintype_categories = [['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']]
garagefinish_categories = [['NA', 'Unf', 'RFn', 'Fin']]
paveddrive_categories = [['N', 'P', 'Y']]
fence_privacy_categories = [['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']]
fence_wood_categories = [['NA', 'GdPrv', 'MnPrv', 'MnWw', 'GdWo']]

# Criar os encoders
lotshape_encoder = OrdinalEncoder(categories=lotshape_categories)
landslope_encoder = OrdinalEncoder(categories=landslope_categories)
bsmtexposure_encoder = OrdinalEncoder(categories=bsmtexposure_categories)
bsmtfintype_encoder = OrdinalEncoder(categories=bsmtfintype_categories)
garagefinish_encoder = OrdinalEncoder(categories=garagefinish_categories)
paveddrive_encoder = OrdinalEncoder(categories=paveddrive_categories)
fence_privacy_encoder = OrdinalEncoder(categories=fence_privacy_categories)
fence_wood_encoder = OrdinalEncoder(categories=fence_wood_categories)

# Aplicar os encoders
data_train['LotShape'] = lotshape_encoder.fit_transform(data_train[['LotShape']])
data_train['LandSlope'] = landslope_encoder.fit_transform(data_train[['LandSlope']])
data_train['BsmtExposure'] = bsmtexposure_encoder.fit_transform(data_train[['BsmtExposure']])
data_train['GarageFinish'] = garagefinish_encoder.fit_transform(data_train[['GarageFinish']])
data_train['PavedDrive'] = paveddrive_encoder.fit_transform(data_train[['PavedDrive']])
data_train['Fence_Privacy'] = fence_privacy_encoder.fit_transform(data_train[['Fence']])
data_train['Fence_Wood'] = fence_wood_encoder.fit_transform(data_train[['Fence']])

# Aplicar os mesmos encoders aos dados de teste
data_test['LotShape'] = lotshape_encoder.transform(data_test[['LotShape']])
data_test['LandSlope'] = landslope_encoder.transform(data_test[['LandSlope']])
data_test['BsmtExposure'] = bsmtexposure_encoder.transform(data_test[['BsmtExposure']])
data_test['GarageFinish'] = garagefinish_encoder.transform(data_test[['GarageFinish']])
data_test['PavedDrive'] = paveddrive_encoder.transform(data_test[['PavedDrive']])
data_test['Fence_Privacy'] = fence_privacy_encoder.transform(data_test[['Fence']])
data_test['Fence_Wood'] = fence_wood_encoder.transform(data_test[['Fence']])

# Ajustar e transformar 'BsmtFinType1'
data_train['BsmtFinType1'] = bsmtfintype_encoder.fit_transform(data_train[['BsmtFinType1']])
data_test['BsmtFinType1'] = bsmtfintype_encoder.transform(data_test[['BsmtFinType1']])

# Ajustar e transformar 'BsmtFinType2'
data_train['BsmtFinType2'] = bsmtfintype_encoder.fit_transform(data_train[['BsmtFinType2']])
data_test['BsmtFinType2'] = bsmtfintype_encoder.transform(data_test[['BsmtFinType2']])

print(data_train.head())

"""

#fim transformação ordinal 
"""


le = LabelEncoder()

for col in list_lb:
    data_train[col] = le.fit_transform(data_train[col].astype(str))
    data_test[col] = le.fit_transform(data_test[col].astype(str))

""" 

column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), pre_data_train),
    remainder='passthrough'
)


# Aplicar a transformação aos dados de treinamento
data_train_encoded = column_trans.fit_transform(data_train)
data_test_encoded = column_trans.transform(data_test) 

#data_train_encoded_dense = data_train_encoded.toarray()
"""

data_train.drop("Fence", axis=1, inplace=True)
data_test.drop("Fence", axis=1, inplace=True)

data_train.to_csv('data/train_prep_encoded.csv', index=False)
data_test.to_csv('data/test_prep_encoded.csv', index=False)