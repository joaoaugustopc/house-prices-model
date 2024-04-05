import pandas as pd
import numpy as np


data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

pre_data_train = data_train.select_dtypes(include=["number"])
pre_data_test = data_test.select_dtypes(include=["number"])

#removendo datas
cols_to_drop = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'GarageYrBlt']
pre_data_train = pre_data_train.drop(cols_to_drop, axis=1)
pre_data_test = pre_data_test.drop(cols_to_drop, axis=1) 

missing_num_train = pd.DataFrame(pre_data_train.isna().sum().sort_values(ascending=False) / pre_data_train.shape[0], columns=["%_missing_values"])
print(missing_num_train[missing_num_train["%_missing_values"] != 0.0])

missing_num_test = pd.DataFrame(pre_data_test.isna().sum().sort_values(ascending=False) / pre_data_test.shape[0], columns=["%_missing_values"])
print(missing_num_test[missing_num_test["%_missing_values"] != 0.0])

# Substituir 'NA' por np.nan
pre_data_train = pre_data_train.replace('NA', np.nan)
pre_data_test = pre_data_test.replace('NA', np.nan)

pre_data_train = pre_data_train.fillna(pre_data_train.mean())
pre_data_test = pre_data_test.fillna(pre_data_train.mean())

pre_data_train.to_csv('data/pre_data_train.csv', index=False)
pre_data_test.to_csv('data/pre_data_test.csv', index=False)
