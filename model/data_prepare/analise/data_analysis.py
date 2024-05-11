import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data_train = pd.read_csv('dataset/train.csv')
data_test = pd.read_csv('dataset/test.csv')

missing_data_train = data_train.isnull().sum().sort_values(ascending=False) / data_train.shape[0]
missing_data_test = data_test.isnull().sum().sort_values(ascending=False) / data_test.shape[0]

print("Missing values in train data")
print(missing_data_train[missing_data_train != 0.0])

print("Missing values in test data")
print(missing_data_test[missing_data_test != 0.0])

""" 
PoolQC          0.995205 // relação entre poolarea e poolQC - como não apresenta todas características, deixar só a área
MiscFeature     0.963014 // como é uma coisa extra, não precisa saber o tipo para o valor do impacto
Alley           0.937671 // separar caracteristicas, deixar
Fence           0.807534 // dividir entre tipos de: privacidade, wood e ausencia se for os dois zero
MasVnrType      0.597260 // tem relação com MasVnrArea, deixar, colocar os tipos e se nao tiver nenhum, é pq nao tem
FireplaceQu     0.472603 // colocar escala
LotFrontage     0.177397

pool e misc - NA = none -> transformar o pool NA para zero, o misc já tem 0 no preço, indica ausencia
pool - deixar os dois
misc - deixar somente o preço
"""

# Criar um boxplot da relação entre 'PoolArea' e 'PoolQC'
plt.figure(figsize=(10, 6))
sns.boxplot(x='PoolQC', y='PoolArea', data=data_train)

plt.title('Relação entre PoolArea e PoolQC')
plt.savefig('graficos/analysing_missing_values/PoolArea_PoolQC.png')

# Criar um boxplot da relação entre 'MiscFeature' e 'MiscVal'
plt.figure(figsize=(10, 6))
sns.boxplot(x='MiscFeature', y='MiscVal', data=data_train)

plt.title('Relação entre MiscFeature e MiscVal')
plt.savefig('graficos/analysing_missing_values/MiscFeature_MiscVal.png')

# Criar um boxplot da relação entre 'Alley' e 'SalePrice', pode ser que impacte no preço
plt.figure(figsize=(10, 6))
sns.boxplot(x='Alley', y='SalePrice', data=data_train)

plt.title('Relação entre Alley e SalePrice')
plt.savefig('graficos/analysing_missing_values/Alley_SalePrice.png')

# Criar um boxplot da relação entre 'MasVnrType' e 'MasVnrArea', observa-se que se tem, mostra o tipo
plt.figure(figsize=(10, 6))
sns.boxplot(x='MasVnrType', y='MasVnrArea', data=data_train)

plt.title('Relação entre MasVnrType e MasVnrArea')
plt.savefig('graficos/analysing_missing_values/MasVnrType_MasVnrArea.png')


categorical = data_train.select_dtypes(include='object').columns

print(categorical)
"""
# atualiza 'MasVnrType' com base em 'MasVnrArea'
data_train.loc[data_train['MasVnrArea'] == 0, 'MasVnrType'] = 'ausente'
data_train.loc[data_train['MasVnrArea'].isna(), 'MasVnrType'] = pd.NA

data_test.loc[data_test['MasVnrArea'] == 0, 'MasVnrType'] = 'ausente'
data_test.loc[data_test['MasVnrArea'].isna(), 'MasVnrType'] = pd.NA
"""

