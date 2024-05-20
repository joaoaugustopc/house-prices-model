import pandas as pd
import numpy as np

# Imputação dos dados númericos, pois os categoricos foram codificados e os NA representam uma categoria

# variáveis categóricas que possui valores faltantes sem explicação : 
# MasVnrType: Masonry veneer type ( categoria "nan" = None e NA ) -> Teste e treino
# MasVnrArea: Masonry veneer area in square feet OK ----> Regrediu alguns valores negativos para alguns NA, levando a crer que não possui masonry veneer (MasVnrType = None)
# LotFrontage: Linear feet of street connected to property OK --> Regrediu valores coerentes e positivos para os NA
# Electrical: Electrical system  --> Possui uma categoria "nan" que representa um valor faltante, mesmo o Nan nao sendo uma informação  ( TREINO APENAS)
# Mszoning : Identifies the general zoning classification of the sale. ( TESTE APENAS ) --> Valor faltante não representa uma categoria, mas esta codificado quando todas as categorias é 0
# KitchenQual: Kitchen quality ( TESTE APENAS ) ->  ((((( Valor faltante = 0 ))))), uma linha apenas
# Utilities : Type of utilities available ( TESTE APENAS ) -> ((((( Valor faltante = 0 ))))), uma linha apenas
# Functional: Home functionality ( TESTE APENAS ) -> ((((( Valor faltante = 0 ))))), uma linha apenas
# SaleType: Type of sale ( TESTE APENAS ) -> ((((( Valor faltante = 0 ))))), uma linha apenas
# Exterior1st: Exterior covering on house ( TESTE APENAS ) -> ((((( Valor faltante = 0 ))))), uma linha apenas

data_train = pd.read_csv('dataset/train_scaled.csv')
data_test = pd.read_csv('dataset/test_scaled.csv')


"""Missing values in train data
remainder__LotFrontage    0.177397
remainder__GarageYrBlt    0.055479  NA significa que nao tem garagem
remainder__MasVnrType     0.008904
remainder__MasVnrArea     0.005479
remainder__Electrical     0.000685
dtype: float64
Missing values in test data
remainder__LotFrontage     0.155586
remainder__GarageYrBlt     0.053461 NA significa que nao tem garagem
remainder__MasVnrType      0.012337
remainder__MasVnrArea      0.010281
remainder__MSZoning        0.002742
remainder__Utilities       0.001371
remainder__BsmtFullBath    0.001371 Na significa que nao tem porao
remainder__BsmtHalfBath    0.001371 Na significa que nao tem porao
remainder__Functional      0.001371
remainder__SaleType        0.000685
remainder__GarageCars      0.000685 
remainder__BsmtFinSF1      0.000685 Na significa que nao tem porao
remainder__BsmtFinSF2      0.000685 Na significa que nao tem porao
remainder__BsmtUnfSF       0.000685 Na significa que nao tem porao
remainder__TotalBsmtSF     0.000685 Na significa que nao tem porao
remainder__Exterior1st     0.000685
remainder__GarageArea      0.000685
remainder__KitchenQual     0.000685
dtype: float64"""


data_train['remainder__GarageYrBlt'] = data_train['remainder__GarageYrBlt'].fillna(0)

data_test['remainder__GarageYrBlt'] = data_test['remainder__GarageYrBlt'].fillna(0)
data_test['remainder__BsmtHalfBath'] = data_test['remainder__BsmtHalfBath'].fillna(0)
data_test['remainder__BsmtFullBath'] = data_test['remainder__BsmtFullBath'].fillna(0)
data_test['remainder__TotalBsmtSF'] = data_test['remainder__TotalBsmtSF'].fillna(0)
data_test['remainder__BsmtUnfSF'] = data_test['remainder__BsmtUnfSF'].fillna(0)
data_test['remainder__BsmtFinSF2'] = data_test['remainder__BsmtFinSF2'].fillna(0)
data_test['remainder__BsmtFinSF1'] = data_test['remainder__BsmtFinSF1'].fillna(0)

missing_data_train = data_train.isnull().sum().sort_values(ascending=False) / data_train.shape[0]
missing_data_test = data_test.isnull().sum().sort_values(ascending=False) / data_test.shape[0]

print("Missing values in train data")
print(missing_data_train[missing_data_train != 0.0])

print("Missing values in test data")
print(missing_data_test[missing_data_test != 0.0])


list_missing_cols_train = missing_data_train[missing_data_train != 0.0].index
list_missing_cols_test = missing_data_test[missing_data_test != 0.0].index

list_missing_cols = list_missing_cols_train.union(list_missing_cols_test)

list_missing_cols_categ  = ['remainder__MasVnrType', 'remainder__Electrical', 'remainder__MSZoning','remainder__Utilities','remainder__Functional','remainder__SaleType','remainder__Exterior1st','remainder__KitchenQual']

""" 
VARIÁVEIS COM INTERSECÇÃO DE VALORES FALTANTES )
remainder__MasVnrType ( treino )
remainder__MasVnrType ( teste ) 
remainder__MSZoning ( teste )
remainder__Utilities ( todas as linhas com interseção )
remainder__Functional ( teste )
"""
from sklearn.neighbors import KNeighborsClassifier

intersection = np.array([])

for col in list_missing_cols_categ:
    train = data_train[data_train[col].notna()]
    train = train.drop(columns=['remainder__Id'])
    train = train.drop(columns=['SalePrice'])
    X_train = train.dropna()
    y_train = X_train[col]
    X_train = X_train.drop(columns = [col])


    model = KNeighborsClassifier()

    model.fit(X_train,y_train)

    if col in ['remainder__MasVnrType', 'remainder__Electrical']:
        test = data_train[data_train[col].isna()]
        test = test.drop(columns=['remainder__Id','SalePrice'])
        X_test = test.drop(columns=[col])
        na_index = X_test[X_test.notna().all(axis=1)].index
        X_test = X_test.dropna()

        if(len(na_index) != len(test[col])):
            intersection = np.append(intersection,col)
        y_pred = model.predict(X_test)

        data_train.loc[na_index,col] = y_pred
    if col != 'remainder__Electrical':
        test = data_test[data_test[col].isna()]
        test = test.drop(columns=['remainder__Id'])

        X_test = test.drop(columns=[col])
        na_index = X_test[X_test.notna().all(axis=1)].index
        X_test = X_test.dropna()

        if(len(na_index) != len(test[col])):
            intersection = np.append(intersection,col)

        if(len(na_index) == 0):
            continue

        y_pred = model.predict(X_test)

        data_test.loc[na_index,col] = y_pred



intersection = np.unique(intersection)

for col in intersection:
    print(col)
    train = data_train[data_train[col].notna()]
    train = train.drop(columns=['remainder__Id'])
    train = train.drop(columns=['SalePrice'])
    X_train = train.dropna()
    y_train = X_train[col]
    X_train = X_train.drop(columns = [col])

    if col == 'remainder__MasVnrType':
        test = data_train[data_train[col].isna()]
        test = test.drop(columns=['remainder__Id','SalePrice'])
        X_test = test.drop(columns=[col])
        X_test = X_test.dropna(axis=1)
        cols_after_drop = X_test.columns
    
    
        X_train = X_train[cols_after_drop]
        
        model = KNeighborsClassifier()

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        data_train.loc[data_train[col].isna(),col] = y_pred


    test = data_test[data_test[col].isna()]
    test = test.drop(columns=['remainder__Id'])
    X_test = test.drop(columns=[col])
    X_test = X_test.dropna(axis=1)
    cols_after_drop = X_test.columns

    X_train = X_train[cols_after_drop]

    model = KNeighborsClassifier()

    model.fit(X_train,y_train)


    y_pred = model.predict(X_test)
    data_test.loc[data_test[col].isna(),col] = y_pred 


list_missing_cols = list_missing_cols.drop(list_missing_cols_categ)
print(list_missing_cols)

from sklearn.neighbors import KNeighborsRegressor

intersection = np.array([])

for col in list_missing_cols:
    train = data_train[data_train[col].notna()]
    train = train.drop(columns=['remainder__Id'])
    train = train.drop(columns=['SalePrice'])

    X_train = train.dropna()
    y_train = X_train[col]
    X_train = X_train.drop(columns = [col])

    model = KNeighborsRegressor()

    model.fit(X_train,y_train)

    if col in ['remainder__LotFrontage','remainder__MasVnrArea']:
        test = data_train[data_train[col].isna()]
        test = test.drop(columns=['remainder__Id','SalePrice'])
        X_test = test.drop(columns=[col])
        na_index = X_test[X_test.notna().all(axis=1)].index
        X_test = X_test.dropna()
        
        if(len(na_index) != len(test[col])):
            print(col, "treino com interseção")
            intersection = np.append(intersection,col)

        if(len(na_index) == 0):
            print(col,"com todas as linhas do treino com interseção")
            continue

        y_pred = model.predict(X_test)

        data_train.loc[na_index,col] = y_pred
    
    test = data_test[data_test[col].isna()]
    test = test.drop(columns=['remainder__Id'])
    X_test = test.drop(columns=[col])
    na_index = X_test[X_test.notna().all(axis=1)].index
    X_test = X_test.dropna()

    if(len(na_index) != len(test[col])):
        print(col,"teste com interseção")
        intersection = np.append(intersection,col)
    
    if(len(na_index) == 0):
        print(col,"com todas as linhas teste com interseção")
        continue

    y_pred = model.predict(X_test)

    data_test.loc[na_index,col] = y_pred

intersection = np.unique(intersection)

print(intersection)

""" 
remainder__GarageArea com todas as linhas do teste com interseção
remainder__GarageCars com todas as linhas do teste com interseção
remainder__LotFrontage ( treino e teste ) com interseção
remainder__MasVnrArea ( treino e teste ) com interseção
"""

for col in intersection:
    print(col)
    train = data_train[data_train[col].notna()]
    train = train.drop(columns=['remainder__Id'])
    train = train.drop(columns=['SalePrice'])
    X_train = train.dropna()
    y_train = X_train[col]
    X_train = X_train.drop(columns = [col])

    if col in ['remainder__LotFrontage','remainder__MasVnrArea']:
        test = data_train[data_train[col].isna()]
        test = test.drop(columns=['remainder__Id','SalePrice'])
        X_test = test.drop(columns=[col])
        X_test = X_test.dropna(axis=1)
        cols_after_drop = X_test.columns

        X_train = X_train[cols_after_drop]

        model = KNeighborsRegressor()

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        data_train.loc[data_train[col].isna(),col] = y_pred

    test = data_test[data_test[col].isna()]
    test = test.drop(columns=['remainder__Id'])
    X_test = test.drop(columns=[col])
    X_test = X_test.dropna(axis=1)
    cols_after_drop = X_test.columns

    X_train = X_train[cols_after_drop]

    model = KNeighborsRegressor()

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    data_test.loc[data_test[col].isna(),col] = y_pred

missing_data_train = data_train.isnull().sum().sort_values(ascending=False) / data_train.shape[0]

missing_data_test = data_test.isnull().sum().sort_values(ascending=False) / data_test.shape[0]

print("Missing values in train data")
print(missing_data_train[missing_data_train != 0.0])

print("Missing values in test data")
print(missing_data_test[missing_data_test != 0.0])

data_train['remainder__Id'] = data_train['remainder__Id'].astype(int)
data_test['remainder__Id'] = data_test['remainder__Id'].astype(int)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

y_train = data_train['SalePrice']
data_train = data_train.drop(columns=['SalePrice'])

# Adicionando novas colunas para binarização
columns_to_binarize = ['remainder__MSZoning', 'remainder__Utilities', 'remainder__Exterior1st', 'remainder__SaleType', 'remainder__Functional', 'remainder__Electrical', 'remainder__MasVnrType']

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

le = LabelEncoder()

"""
for col in columns_to_binarize:
    data_train[col] = le.fit_transform(data_train[col].astype(str))
    data_test[col] = le.transform(data_test[col].astype(str))
"""
    
columns_train = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), columns_to_binarize),
    remainder='passthrough'
)

data_train_encoded = columns_train.fit_transform(data_train)
data_test_encoded = columns_train.transform(data_test)

train = pd.DataFrame(data_train_encoded,columns=columns_train.get_feature_names_out())
test = pd.DataFrame(data_test_encoded,columns = columns_train.get_feature_names_out())

train['SalePrice'] = y_train

train.to_csv('dataset/train_encoded_imputed.csv',index=False)
test.to_csv('dataset/test_encoded_imputed.csv',index=False)
