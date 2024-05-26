import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

"""
    Carrega os dados de treino e teste
    :param file_train: Nome do arquivo de treino
    :param file_test: Nome do arquivo de teste
    :param raw: Se os dados são brutos ou processados
    :return: DataFrame de treino e teste
"""
def load_data(file_train, file_test, raw=False):
    dir = 'raw' if raw else 'processed'
    data_train = pd.read_csv(f'dataset/{dir}/{file_train}.csv')
    data_test = pd.read_csv(f'dataset/{dir}/{file_test}.csv')
    return data_train, data_test

""" 
    Normaliza os dados de treino e teste, função não aceita NA como string
    :param data_train: DataFrame de treino
    :param data_test: DataFrame de teste
    :param exclude_cols: Colunas a serem excluídas
    :return: DataFrames normalizados
"""
def robust_scaler(data_train, data_test, exclude_cols=[]):
    # Colunas binárias
    cols_to_drop = data_train.filter(regex='onehotencoder').columns.tolist() + exclude_cols
    print(cols_to_drop)

    # Selecionar as colunas para normalizar
    cols_to_scale = data_train.columns.difference(cols_to_drop)
    scaler = RobustScaler()

    # Normalizar as colunas selecionadas
    data_train[cols_to_scale] = scaler.fit_transform(data_train[cols_to_scale])
    data_test[cols_to_scale] = scaler.transform(data_test[cols_to_scale])

    data_train["remainder__Id"] = data_train["remainder__Id"].astype(int)
    data_test["remainder__Id"] = data_test["remainder__Id"].astype(int)

    return data_train, data_test


""" 
    Binarizando dados usando one hot encoder
"""
def binarize_data(data_train, data_test, cols_to_binarize):
    target = data_train['SalePrice']
    data_train = data_train.drop(columns=['SalePrice'])
    
    data_train.columns = data_train.columns.str.replace('onehotencoder__', '')
    data_test.columns = data_test.columns.str.replace('onehotencoder__', '')
    
    columns_train = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), cols_to_binarize),
        remainder='passthrough'
    )
    
    data_train = columns_train.fit_transform(data_train)
    data_test = columns_train.transform(data_test)
    
    train = pd.DataFrame(data_train, columns=columns_train.get_feature_names_out())
    train['SalePrice'] = target
    test = pd.DataFrame(data_test, columns=columns_train.get_feature_names_out())
    
    return train, test
    
     
def print_missing_data(data):
    missing_data = data.isnull().sum().sort_values(ascending=False) / data.shape[0]
    print(missing_data[missing_data != 0.0])

def train_model_imputing(data, col, model, cols = []):
    to_drop = ['remainder__SalePrice', 'remainder__Id'] if 'remainder__Saleprice' in data.columns else ['remainder__Id']
    data_to_train = data.drop(columns=to_drop)
    #mantendo somente as linhas que não possuem valores faltantes
    data_to_train = data_to_train[data_to_train[col].notna()].dropna()
    X = data_to_train.drop(columns=[col])
    y = data_to_train[col]
    
    if len(cols) > 0:
        X = X[cols]
    
    model.fit(X, y)
    return model



    


    