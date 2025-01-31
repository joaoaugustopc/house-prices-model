import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from utils.data_transformation import load_data, robust_scaler, binarize_data

def main():

    data_train, data_test = load_data('train', 'test', raw=True)
    y_train = data_train['SalePrice']
    data_train.drop(columns=['SalePrice'], inplace=True)

    #altera 'masvnrtype' com base em 'masvnrarea' -> tratando problema entre NA e None
    data_train.loc[data_train['MasVnrArea'] == 0, 'MasVnrType'] = 'MasVnrNone'
    data_test.loc[data_test['MasVnrArea'] == 0, 'MasVnrType'] = 'MasVnrNone'

    # Lista de variáveis categóricas que possuem uma ordem natural padronizada
    list_ordinal = ['BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond','ExterQual','ExterCond','HeatingQC'] # NA, Po, Fa, TA, Gd, Ex
    #['ExterQual','ExterCond','HeatingQC','KitchenQual'] # Po, Fa, TA, Gd, Ex  -> Se aparecer 0 é valor faltante

    list = list_ordinal + ['LotShape','LandSlope','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','PavedDrive', 'MasVnrType','Electrical', 'MSZoning','Utilities','Exterior1st','SaleType','Functional','KitchenQual','CentralAir','Street']

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
        'Street': [['Grvl', 'Pave']],
        'CentralAir': [['N', 'Y']],
        #atributos categóricos (FALTANTES NO TESTE) transformados em ordinais para fazer o imputing -> será binarizado depois do imputing
        'MSZoning': [['A', 'C (all)', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']],
        'Utilities': [['ELO', 'NoSeWa', 'NoSewr', 'AllPub']],
        'Exterior1st': [['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']],
        'SaleType': [['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'New', 'Oth', 'WD']],
        'Functional': [['Maj2', 'Maj1', 'Min1', 'Min2', 'Mod', 'Sev', 'Typ']],
        'KitchenQual': [['Po', 'Fa', 'TA', 'Gd', 'Ex']],
        #atributo categórico (FALTANTE NO TREINO) transformado em ordinal para fazer o imputing -> será binarizado depois do imputing
        'Electrical': [['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']],
        #atributo categórico (FALTANTE NO TREINO e TESTE) transformado em ordinal para fazer o imputing -> será binarizado depois do imputing
        'MasVnrType': [['BrkCmn', 'BrkFace', 'CBlock', 'Stone', 'MasVnrNone']]
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

    #colunas binarizada
    train, test = binarize_data(data_train, data_test, pre_data_train)

    train['SalePrice'] = y_train
    
    #usando o robust scaler
    cols_exclude_scale = ["SalePrice","remainder__Id", "remainder__MSZoning", 
                          "remainder__Utilities", "remainder__Exterior1st", "remainder__SaleType", 
                          "remainder__Functional", "remainder__Electrical", "remainder__MasVnrType", "remainder__KitchenQual"]
    train, test = robust_scaler(train, test, cols_exclude_scale)
    
    train = train.fillna('NA')
    test = test.fillna('NA')

    train.to_csv('dataset/processed/train_encoded.csv', index=False)
    test.to_csv('dataset/processed/test_encoded.csv', index=False)

if __name__ == '__main__':
    main()