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
list_ordinal = ['BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond','ExterQual','ExterCond','HeatingQC'] # NA, Po, Fa, TA, Gd, Ex

#list_ordinal2 = ['ExterQual','ExterCond','HeatingQC','KitchenQual'] # Po, Fa, TA, Gd, Ex  -> Se aparecer 0 é valor faltante

list_lb = ['CentralAir','Street'] # Sim ou não / Paved ou Gravel

list = list_ordinal+ list_lb + ['LotShape','LandSlope','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','PavedDrive', 'MasVnrType','Electrical', 'MSZoning','Utilities','Exterior1st','SaleType','Functional','KitchenQual']

#retirando colunas que não serão binarizadas
pre_data_train = data_train.drop(columns=list)
pre_data_train = pre_data_train.select_dtypes(include=["object"]).columns
pre_data_test = data_test.drop(columns=list)
pre_data_test = pre_data_test.select_dtypes(include=["object"]).columns

data_train[list] = data_train[list].fillna('NA')
data_test[list] = data_test[list].fillna('NA')

#ordenando atributos que possuem ordens diferentes
categories_dict = {
    #atributos ordinais - ordem natural -> permanecerá nas tabelas
    'LotShape': [['IR3', 'IR2', 'IR1', 'Reg']],
    'LandSlope': [['Sev', 'Mod', 'Gtl']],
    'BsmtExposure': [['NA', 'No', 'Mn', 'Av', 'Gd']],
    'BsmtFinType1': [['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']],
    'BsmtFinType2': [['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']],
    'GarageFinish': [['NA', 'Unf', 'RFn', 'Fin']],
    'PavedDrive': [['N', 'P', 'Y']],
    #atributos categóricos (FALTANTES NO TESTE) transformados em ordinais para fazer o imputing -> será binarizado depois do imputing
    'MSZoning': [['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']],
    'Utilities': [['ELO', 'NoSeWa', 'NoSewr', 'AllPub']],
    'Exterior1st': [['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']],
    'SaleType': [['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'WD']],
    'Functional': [['Maj2', 'Maj1', 'Min1', 'Min2', 'Mod', 'Sev', 'Typ']],
    #atributo categórico (FALTANTE NO TREINO) transformado em ordinal para fazer o imputing -> será binarizado depois do imputing
    'Electrical': [['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']]
}

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

#aplicando encoder para cada item do dict
for col, categories in categories_dict.items():
    #definindo a categoria para o encoder
    encoder.categories = categories
    #transformando os dados
    data_train[col] = encoder.fit_transform(data_train[[col]])
    #transformando os dados de teste
    data_test[col] = encoder.transform(data_test[[col]])


# Transformar as variáveis categóricas em numéricas ordenadas -> ordem natural equivalente -> NA, Po, Fa, TA, Gd, Ex
categories = [['NA','Po','Fa','TA','Gd','Ex']] * len(list_ordinal)
encoder = OrdinalEncoder(categories=categories)
data_train[list_ordinal] = encoder.fit_transform(data_train[list_ordinal])
data_test[list_ordinal] = encoder.transform(data_test[list_ordinal])


#altera 'masvnrtype' com base em 'masvnrarea' -> tratando problema entre NA e None
data_train.loc[data_train['MasVnrArea'] == 0, 'MasVnrType'] = 'MasVnrNone'
#data_train.loc[data_train['MasVnrArea'].isna(), 'MasVnrType'] = np.nan ( Não faz diferença )
data_test.loc[data_test['MasVnrArea'] == 0, 'MasVnrType'] = 'MasVnrNone'

# Definir a ordem das categorias
categories = [['BrkCmn', 'BrkFace', 'CBlock', 'Stone', 'MasVnrNone']]
# Criar um objeto OrdinalEncoder
encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
# Transformar 'MasVnrType' em uma coluna ordinal
data_train['MasVnrType'] = encoder.fit_transform(data_train[['MasVnrType']])
data_test['MasVnrType'] = encoder.transform(data_test[['MasVnrType']])

categories = [['Po','Fa','TA','Gd','Ex']]
encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)

data_train['KitchenQual'] = encoder.fit_transform(data_train[['KitchenQual']])
data_test['KitchenQual'] = encoder.transform(data_test[['KitchenQual']])

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
