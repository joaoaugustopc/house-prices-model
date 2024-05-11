import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pickle


data_train = pd.read_csv('dataset/train.csv', na_values=['NA'])
data_test = pd.read_csv('dataset/test.csv', na_values=['NA'])
y_train = data_train['SalePrice']
data_train.drop(columns=['SalePrice'], inplace=True)

# Lista de variáveis categóricas que possuem uma ordem natural padronizada
list_ordinal = ['BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond','ExterQual','ExterCond','HeatingQC','KitchenQual'] # NA, Po, Fa, TA, Gd, Ex

#list_ordinal2 = ['ExterQual','ExterCond','HeatingQC','KitchenQual'] # Po, Fa, TA, Gd, Ex  -> Se aparecer 0 é valor faltante

list_lb = ['CentralAir','Street'] # Sim ou não / Paved ou Gravel

list = list_ordinal+ list_lb + ['LotShape','LandSlope','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','PavedDrive', 'MasVnrType','Electrical', 'MSZoning','Utilities','Exterior1st','SaleType','Functional']

""" 
list_train = list + ['MasVnrType','Electrical']

list_test = list + ['MSZoning','Utilities','Exterior1st','SaleType','Functional']
"""

pre_data_train = data_train.drop(columns=list)
pre_data_train = pre_data_train.select_dtypes(include=["object"]).columns

pre_data_test = data_test.drop(columns=list)
pre_data_test = pre_data_test.select_dtypes(include=["object"]).columns

data_train[list] = data_train[list].fillna('NA')
data_test[list] = data_test[list].fillna('NA')


# Criar listas de categorias com ordem diferentes
lotshape_categories = [['IR3', 'IR2', 'IR1', 'Reg']] #olhar
landslope_categories = [['Sev', 'Mod', 'Gtl']]
bsmtexposure_categories = [['NA', 'No', 'Mn', 'Av', 'Gd']]
bsmtfintype_categories = [['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']]
garagefinish_categories = [['NA', 'Unf', 'RFn', 'Fin']]
paveddrive_categories = [['N', 'P', 'Y']]


# Criar os encoders
lotshape_encoder = OrdinalEncoder(categories=lotshape_categories)
landslope_encoder = OrdinalEncoder(categories=landslope_categories)
bsmtexposure_encoder = OrdinalEncoder(categories=bsmtexposure_categories)
bsmtfintype_encoder = OrdinalEncoder(categories=bsmtfintype_categories)
garagefinish_encoder = OrdinalEncoder(categories=garagefinish_categories)
paveddrive_encoder = OrdinalEncoder(categories=paveddrive_categories)


# Aplicar os encoders
data_train['LotShape'] = lotshape_encoder.fit_transform(data_train[['LotShape']])
data_train['LandSlope'] = landslope_encoder.fit_transform(data_train[['LandSlope']])
data_train['BsmtExposure'] = bsmtexposure_encoder.fit_transform(data_train[['BsmtExposure']])
data_train['GarageFinish'] = garagefinish_encoder.fit_transform(data_train[['GarageFinish']])
data_train['PavedDrive'] = paveddrive_encoder.fit_transform(data_train[['PavedDrive']])


# Aplicar os mesmos encoders aos dados de teste
data_test['LotShape'] = lotshape_encoder.transform(data_test[['LotShape']])
data_test['LandSlope'] = landslope_encoder.transform(data_test[['LandSlope']])
data_test['BsmtExposure'] = bsmtexposure_encoder.transform(data_test[['BsmtExposure']])
data_test['GarageFinish'] = garagefinish_encoder.transform(data_test[['GarageFinish']])
data_test['PavedDrive'] = paveddrive_encoder.transform(data_test[['PavedDrive']])


# Ajustar e transformar 'BsmtFinType1'
data_train['BsmtFinType1'] = bsmtfintype_encoder.fit_transform(data_train[['BsmtFinType1']])
data_test['BsmtFinType1'] = bsmtfintype_encoder.transform(data_test[['BsmtFinType1']])

# Ajustar e transformar 'BsmtFinType2'
data_train['BsmtFinType2'] = bsmtfintype_encoder.fit_transform(data_train[['BsmtFinType2']])
data_test['BsmtFinType2'] = bsmtfintype_encoder.transform(data_test[['BsmtFinType2']])

# Transformar as variáveis categóricas em numéricas ordenadas
categories = [['NA','Po','Fa','TA','Gd','Ex']] * len(list_ordinal)

encoder = OrdinalEncoder(categories=categories)

data_train[list_ordinal] = encoder.fit_transform(data_train[list_ordinal])
data_test[list_ordinal] = encoder.transform(data_test[list_ordinal])


#transformando atributos categóricos faltantes em ordinal para imputing - TREINO

#altera 'masvnrtype' com base em 'masvnrarea'
data_train.loc[data_train['MasVnrArea'] == 0, 'MasVnrType'] = 'MasVnrNone'
data_train.loc[data_train['MasVnrArea'].isna(), 'MasVnrType'] = np.nan

data_test.loc[data_test['MasVnrArea'] == 0, 'MasVnrType'] = 'MasVnrNone'

# Definir a ordem das categorias
categories = [['BrkCmn', 'BrkFace', 'CBlock', 'Stone', 'MasVnrNone']]
# Criar um objeto OrdinalEncoder
encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
# Transformar 'MasVnrType' em uma coluna ordinal
data_train['MasVnrType'] = encoder.fit_transform(data_train[['MasVnrType']])

#altera electrical
# Definir a ordem das categorias para 'Electrical'
categories = [['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']]
# Criar um objeto OrdinalEncoder
encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
# Transformar 'Electrical' em uma coluna ordinal
data_train['Electrical'] = encoder.fit_transform(data_train[['Electrical']])
data_test['Electrical'] = encoder.transform(data_test[['Electrical']])

#transformando atributos categóricos faltantes em ordinal para imputing - TESTE

# Definir a ordem das categorias para cada coluna
categories_dict = {
    'MSZoning': [['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']],
    'Utilities': [['ELO', 'NoSeWa', 'NoSewr', 'AllPub']],
    'Exterior1st': [['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']],
    'SaleType': [['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'WD']],
    'Functional': [['Maj2', 'Maj1', 'Min1', 'Min2', 'Mod', 'Sev', 'Typ']]
}

# Criar um objeto OrdinalEncoder
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

for col, categories in categories_dict.items():
    encoder.categories = categories
    encoder.fit_transform(data_train[[col]])
    data_test[col] = encoder.transform(data_test[[col]])

# Transformar as variáveis categóricas em numéricas não ordenadas em binárias

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

train.fillna('NA', inplace=True)
test.fillna('NA', inplace=True)

train.to_csv('dataset/train_encoded.csv', index=False)
test.to_csv('dataset/test_encoded.csv', index=False)

#TODO: fazer as mesmas ordenações no teste e no treino