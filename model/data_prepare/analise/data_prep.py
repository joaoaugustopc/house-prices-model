import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data_train = pd.read_csv('dataset/train.csv')
data_test = pd.read_csv('dataset/test.csv')

#Verificando e tratando valores faltantes

missing_data_train = data_train.isnull().sum().sort_values(ascending=False) / data_train.shape[0]
missing_data_test = data_test.isnull().sum().sort_values(ascending=False) / data_test.shape[0]

print("Missing values in train data")
print(missing_data_train[missing_data_train != 0.0])

print("Missing values in test data")
print(missing_data_test[missing_data_test != 0.0])

"""Missing values in train data
PoolQC          0.995205
MiscFeature     0.963014
Alley           0.937671
Fence           0.807534"""

"""Missing values in test data
PoolQC          0.997944
MiscFeature     0.965045
Alley           0.926662
Fence           0.801234"""

"""
# distribuição dos valores faltantes
missing_data = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
fig = plt.figure(figsize=(10, 10))

for i in range(0, len(missing_data)):
    fig.add_subplot(2, 2, i+1)
    sns.countplot(data_train[missing_data[i]])

plt.tight_layout()
plt.savefig('graficos/Dist_missing_Categ.png')

# Relação entre as variáveis com valores faltantes e a variável target
fig = plt.figure(figsize=(10, 10))

for i in range(0, len(missing_data)):
    fig.add_subplot(2, 2, i+1)
    sns.boxplot(x=missing_data[i], y='SalePrice', data=data_train)

plt.tight_layout()
plt.savefig('graficos/BoxPlot_missing_Categ.png')


#PoolQC: Muito poucos valores preenchidos
#MiscFeature: Poucos valores preenchidos, e os preenchidos não possui uma influencia clara no preço da casa
#Fence: Poucos valores preenchidos, os preenchidos não possui uma influencia clara no preço da casa além de possuir varios valores faltantes

"""

# Excluir as variáveis ['PoolQC', 'MiscFeature' e 'Fence']
data_train = data_train.drop(['PoolQC', 'MiscFeature', 'Fence','Alley'], axis=1)
data_test = data_test.drop(['PoolQC', 'MiscFeature', 'Fence','Alley'], axis=1)

# Variáveis numéricas
X_train = data_train.select_dtypes(include=["number"])
X_test = data_test.select_dtypes(include=["number"])

print("Missing values in train data")
missing_num_train = pd.DataFrame(X_train.isna().sum().sort_values(ascending=False) / X_train.shape[0], columns=["%_missing_values"])
print(missing_num_train[missing_num_train["%_missing_values"] != 0.0])

print("Missing values in test data")
missing_num_test = pd.DataFrame(X_test.isna().sum().sort_values(ascending=False) / X_test.shape[0], columns=["%_missing_values"])
print(missing_num_test[missing_num_test["%_missing_values"] != 0.0])

"""            %_missing_values
LotFrontage          0.177397
GarageYrBlt          0.055479
MasVnrArea           0.005479"""

fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
sns.histplot(X_train['LotFrontage'], kde=True)
plt.title(' LotFrontage distribuition ')

mean = X_train['LotFrontage'].mean() # 70.04995836802665
median = X_train['LotFrontage'].median() # 69.0

mean_line = plt.axvline(mean, color='r', linestyle='-')
median_line = plt.axvline(median, color='g', linestyle='-')

plt.legend([mean_line, median_line], ['Mean: {:.2f}'.format(mean), 'Median: {:.2f}'.format(median)])

fig.add_subplot(2, 2, 2)
sns.scatterplot(x='LotFrontage', y='SalePrice', data=data_train)
plt.title(' LotFrontage x SalePrice ')

plt.tight_layout()
plt.savefig('graficos/LotFrontage.png')



corr = X_train["LotFrontage"].corr(data_train['SalePrice'])

print(f"Correlation between LotFrontage and SalePrice: {corr}") # 0.35179909657067804

fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
sns.histplot(X_train['GarageYrBlt'], kde=True)
plt.title(' GarageYrBlt distribuition ')

mean = X_train['GarageYrBlt'].mean()
median = X_train['GarageYrBlt'].median() 

mean_line = plt.axvline(mean, color='r', linestyle='--')
median_line = plt.axvline(median, color='g', linestyle='-')

plt.legend([mean_line, median_line], ['Mean: {:.2f}'.format(mean), 'Median: {:.2f}'.format(median)])

fig.add_subplot(2, 2, 2)
sns.histplot(X_train['MasVnrArea'], kde=True)
plt.title(' MasVnrArea distribuition ')

mean = X_train['MasVnrArea'].mean()
median = X_train['MasVnrArea'].median() 

mean_line = plt.axvline(mean, color='r', linestyle='--')
median_line = plt.axvline(median, color='g', linestyle='-')

plt.legend([mean_line, median_line], ['Mean: {:.2f}'.format(mean), 'Median: {:.2f}'.format(median)])

plt.tight_layout()
plt.savefig('graficos/GarageYrBlt_MasVnrArea.png')

print(X_train['LotFrontage'].describe())
print(X_train['GarageYrBlt'].describe())
print(X_train['MasVnrArea'].describe())

X_train = X_train.replace('NA', np.nan)
X_test = X_test.replace('NA', np.nan)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

data_train[X_train.columns] = X_train
data_test[X_test.columns] = X_test

# Analisando a distribuição de 'SalePrice'

fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
sns.histplot(data_train['SalePrice'], kde=True)
plt.title(' SalePrice distribuition ')

mean = data_train['SalePrice'].mean()
median = data_train['SalePrice'].median()

mean_line = plt.axvline(mean, color='r', linestyle='--')
median_line = plt.axvline(median, color='g', linestyle='-')

plt.legend([mean_line, median_line], ['Mean: {:.2f}'.format(mean), 'Median: {:.2f}'.format(median)])

fig.add_subplot(2, 2, 2)
sns.boxplot(data_train['SalePrice'])
plt.title(' SalePrice boxplot ')

plt.tight_layout()
plt.savefig('graficos/SalePrice.png')

# A distribuição dos preços das casas é assimétrica, o que pode ser um problema para a regressão linear. Vamos tentar transformar os preços das casas para que eles sejam mais normalmente distribuídos.
#aplicar log nos valores de SalePrice
# Transformando a variável 'SalePrice' para uma distribuição normal, aplicando log

numeric_features = data_train.select_dtypes(include=[np.number])
numeric_features = numeric_features.drop('Id', axis=1)

numeric_features.hist(bins=50, figsize=(20, 15))

plt.tight_layout()
plt.savefig('graficos/numeric_features.png')

"""
Recursos como 1stFlrSF, TotalBsmtSF, LotFrontage e GrLiveArea
parecem ter uma distribuuição semelhante à SalePrice
"""

# Correlação entre as variáveis numéricas e SalePrice

correlation = numeric_features.corr()['SalePrice'][:-1]
variaveis_chave = correlation[abs(correlation) > 0.5].sort_values(ascending=False)

print("SalesPrice Possui", len(variaveis_chave), "correlações significativas:\n", variaveis_chave)

"""SalesPrice Possui 10 correlações significativas:
 OverallQual     0.817185
GrLivArea       0.700927
GarageCars      0.680625
GarageArea      0.650888
TotalBsmtSF     0.612134
1stFlrSF        0.596981
FullBath        0.594771
YearBuilt       0.586570
YearRemodAdd    0.565608
TotRmsAbvGrd    0.534422
Name: SalePrice, dtype: float64"""

fig = plt.figure(figsize=(15,10))

for i in range(0, len(numeric_features.columns)):
    fig.add_subplot(8,5,i+1)
    sns.scatterplot(x=numeric_features.columns[i], y='SalePrice', data=numeric_features)

plt.tight_layout()

plt.savefig('graficos/dispersão_numericos.png')

# Analisar Variaveis chave
fig = plt.figure(figsize=(15, 10))
"""
for i in range(0, len(variaveis_chave)):
    fig.add_subplot(4, 3, i+1)
    sns.boxplot(x=variaveis_chave.index[i], y='SalePrice', data=data_train)
    
plt.tight_layout()
plt.savefig('graficos/boxplot_variaveis_chave.png')
"""

for i in range(0, len(variaveis_chave)):
    fig.add_subplot(4, 3, i+1)
    sns.histplot(data_train[variaveis_chave.index[i]], kde=True)
    plt.title(f'{variaveis_chave.index[i]} distribuition ')
    mean = data_train[variaveis_chave.index[i]].mean()
    median = data_train[variaveis_chave.index[i]].median()
    mean_line = plt.axvline(mean, color='r', linestyle='--')
    median_line = plt.axvline(median, color='g', linestyle='-')
    plt.legend([mean_line, median_line], ['Mean: {:.2f}'.format(mean), 'Median: {:.2f}'.format(median)])

plt.tight_layout()
plt.savefig('graficos/hist_variaveis_chave.png')

corr = numeric_features.drop('SalePrice', axis=1).corr()

fig = plt.figure(figsize=(15, 10))

sns.heatmap(corr[abs(corr)>=0.5], annot=True, cmap='viridis', vmax=1, vmin=-1, linewidths=0.1, annot_kws={"size": 8}, square=True)

plt.tight_layout()
plt.savefig('graficos/heatmap_correlação.png')

# Remover '1stFlrSF', 'TotRmsAbvGrd' e 'GarageArea' por terem alta correlação com outras variáveis ( teste para regressao linear )

data_train = data_train.drop(['1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], axis=1)
data_test = data_test.drop(['1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], axis=1)

# dados categóricos

categorical_features = data_train.select_dtypes(include=['object'])

#Analisar a relação entre as variáveis categóricas e SalePrice

fig = plt.figure(figsize=(20, 15))

for i in range(0, len(categorical_features.columns)):
    fig.add_subplot(9, 5, i+1)
    sns.boxplot(x=categorical_features.columns[i], y='SalePrice', data=data_train)

plt.tight_layout()
plt.savefig('graficos/boxplot_categ.png')


fig = plt.figure(figsize=(20,15))

for i in range(0, len(categorical_features.columns)):
    fig.add_subplot(9,5,i+1)
    sns.countplot(x=categorical_features.columns[i], data=data_train)

plt.tight_layout()

plt.savefig('graficos/countplot_categ.png')

# Remover Utilities, condition2, Heating, Street, functional
data_train = data_train.drop(['Utilities', 'Condition2', 'Heating', 'Street', 'Functional'], axis=1)
data_test = data_test.drop(['Utilities', 'Condition2', 'Heating', 'Street', 'Functional'], axis=1)

# Aplicando log na variável 'SalePrice' e comparando a distribuição
#data_train['SalePrice'] = np.log1p(data_train['SalePrice'])

fig = plt.figure(figsize=(15, 10))

fig.add_subplot(2, 2, 1)
sns.histplot(data_train['SalePrice'], kde=True)
plt.title(' SalePrice distribuition ')

mean = data_train['SalePrice'].mean()
median = data_train['SalePrice'].median()

mean_line = plt.axvline(mean, color='r', linestyle='--')
median_line = plt.axvline(median, color='g', linestyle='-')

plt.legend([mean_line, median_line], ['Mean: {:.2f}'.format(mean), 'Median: {:.2f}'.format(median)])

fig.add_subplot(2, 2, 2)
sns.boxplot(data_train['SalePrice'])
plt.title(' SalePrice boxplot ')

plt.tight_layout()
plt.savefig('graficos/SalePrice_log.png')

# Normalizando os dados

X_train = data_train.drop(columns=['SalePrice','MSSubClass','OverallQual','OverallCond','Id'])
X_train = X_train.select_dtypes(include=["number"], exclude=["object"])

X_test = data_test.drop(columns=['MSSubClass','OverallQual','OverallCond','Id'])
X_test = X_test.select_dtypes(include=["number"], exclude=["object"])


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

data_train[X_train.columns] = X_train_scaled_df
data_test[X_test.columns] = X_test_scaled_df

print(data_train[X_train.columns].mean())
print(data_train[X_train.columns].std())


data_train.to_csv('dataset/train_prep.csv', index=False)
data_test.to_csv('dataset/test_prep.csv', index=False)
