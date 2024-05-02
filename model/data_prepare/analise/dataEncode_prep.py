import pandas as pd
import numpy as np


data_train = pd.read_csv('data/train_encoded.csv')
data_test = pd.read_csv('data/test_encoded.csv')

missing_num_train = pd.DataFrame(data_train.isna().sum().sort_values(ascending=False) / data_train.shape[0], columns=["%_missing_values"])
print(missing_num_train[missing_num_train["%_missing_values"] != 0.0])

missing_num_test = pd.DataFrame(data_test.isna().sum().sort_values(ascending=False) / data_test.shape[0], columns=["%_missing_values"])
print(missing_num_test[missing_num_test["%_missing_values"] != 0.0])

# Substituir 'NA' por np.nan
data_train = data_train.replace('NA', np.nan)
data_test = data_test.replace('NA', np.nan)

data_train = data_train.fillna(data_train.mean())
data_test = data_test.fillna(data_train.mean())

data_train.to_csv('data/pre_data_train_encoded.csv', index=False)
data_test.to_csv('data/pre_data_test_encoded.csv', index=False)
