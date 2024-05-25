import pandas as pd
from sklearn.preprocessing import RobustScaler

def load_data(file_train, file_test, raw=False):
    dir = 'raw' if raw else 'processed'
    data_train = pd.read_csv(f'dataset/{dir}/{file_train}.csv')
    data_test = pd.read_csv(f'dataset/{dir}/{file_test}.csv')
    return data_train, data_test

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
