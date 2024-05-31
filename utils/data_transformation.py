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
    #colunas binárias
    cols_to_drop = data_train.filter(regex='onehotencoder').columns.tolist() + exclude_cols
    print(cols_to_drop)

    #selecionar as colunas para normalizar
    cols_to_scale = data_train.columns.difference(cols_to_drop)
    scaler = RobustScaler()

    #normalizar as colunas selecionadas
    data_train[cols_to_scale] = scaler.fit_transform(data_train[cols_to_scale])
    data_test[cols_to_scale] = scaler.transform(data_test[cols_to_scale])

    data_train["remainder__Id"] = data_train["remainder__Id"].astype(int)
    data_test["remainder__Id"] = data_test["remainder__Id"].astype(int)

    return data_train, data_test

""" 
    Binarizando dados usando one hot encoder
    A variável target está sendo manipulada para funcionar tanto com dataframes que a possuem, quanto o contrário
"""
def binarize_data(data_train, data_test, cols_to_binarize):
    data_train.columns = data_train.columns.str.replace('remainder__', '')
    data_test.columns = data_test.columns.str.replace('remainder__', '')
    
    columns_train = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), cols_to_binarize),
        remainder='passthrough'
    )
    
    data_train_encoded = columns_train.fit_transform(data_train)
    data_test_encoded = columns_train.transform(data_test)
    
    train = pd.DataFrame(data_train_encoded, columns=columns_train.get_feature_names_out())
    test = pd.DataFrame(data_test_encoded, columns=columns_train.get_feature_names_out())
    
    return train, test
    
""" 
    Imprime as colunas com valores faltantes
    :param data: DataFrame
"""
def print_missing_data(data):
    missing_data = data.isnull().sum().sort_values(ascending=False) / data.shape[0]
    print(missing_data[missing_data != 0.0])

""" 
    Retorna as colunas com valores faltantes
    :param data: DataFrame
    :return: Lista de colunas com valores faltantes
"""
def get_missing_(data):
    return data.isnull().any().tolist()
    


    