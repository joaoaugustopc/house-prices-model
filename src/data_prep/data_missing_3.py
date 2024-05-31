import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from utils.data_transformation import load_data, binarize_data


def get_missing_data(data):
    missing_data = data.isnull().sum().sort_values(ascending=False) / data.shape[0]
    return missing_data[missing_data != 0.0]
     

def get_missing_cols(data):
    missing_data = data.isnull().sum().sort_values(ascending=False) / data.shape[0]
    return missing_data[missing_data != 0.0].index

def train_model(data, column, model, cols = []):
    train = data[data[column].notna()]
    X_train = train.dropna()
    y_train = X_train[column]
    X_train = X_train.drop(columns = [column])

    if len(cols) != 0:
        X_train = X_train[cols]

    model.fit(X_train,y_train)

    return model

def predict_values(data_train, data_test, column, model):
    valsToPredict = data_train[data_train[column].isna()]
    if not valsToPredict.empty:
        X_test = valsToPredict.drop(columns=[column])
        na_index = X_test[X_test.notna().all(axis=1)].index
        X_test = X_test.dropna()

        if(len(na_index) != 0):
            y_pred = model.predict(X_test)
            data_train.loc[na_index,column] = y_pred
    

    valsToPredict = data_test[data_test[column].isna()]
    if not valsToPredict.empty:
        X_test = valsToPredict.drop(columns=[column])
        na_index = X_test[X_test.notna().all(axis=1)].index
        X_test = X_test.dropna()

        if(len(na_index) != 0):
            y_pred = model.predict(X_test)
            data_test.loc[na_index,column] = y_pred

def predict_intersection(data_train, data_test, column, model):
    valsToPredict = data_train[data_train[column].isna()]
    if not valsToPredict.empty:
        X_test = valsToPredict.drop(columns=[column])
        X_test = X_test.dropna(axis=1)
        cols_after_drop = X_test.columns
        model = train_model(data_train,column,model,cols_after_drop)
        
        y_pred = model.predict(X_test)
        data_train.loc[data_train[column].isna(),column] = y_pred
    
    valsToPredict = data_test[data_test[column].isna()]
    if not valsToPredict.empty:
        X_test = valsToPredict.drop(columns=[column])
        X_test = X_test.dropna(axis=1)
        cols_after_drop = X_test.columns
        model = train_model(data_test,column,model,cols_after_drop)
        
        y_pred = model.predict(X_test)
        data_test.loc[data_test[column].isna(),column] = y_pred


def main():
    train,test = load_data('train_encoded','test_encoded')

    target = train['SalePrice']

    data_train = train.drop(columns=['remainder__Id','SalePrice'])
    data_test = test.drop(columns=['remainder__Id'])

    listFill_0 = ['remainder__GarageYrBlt','remainder__BsmtHalfBath',
                  'remainder__BsmtFullBath','remainder__TotalBsmtSF','remainder__BsmtUnfSF',
                  'remainder__BsmtFinSF2','remainder__BsmtFinSF1']
    
    data_train[listFill_0] = data_train[listFill_0].fillna(0)
    data_test[listFill_0] = data_test[listFill_0].fillna(0)

    list_missing_cols_categ  = ['remainder__MasVnrType', 'remainder__Electrical', 'remainder__MSZoning','remainder__Utilities',
                                'remainder__Functional','remainder__SaleType','remainder__Exterior1st','remainder__KitchenQual']

    model = KNeighborsClassifier()

    for col in list_missing_cols_categ:
        model = train_model(data_train,col,model)
        predict_values(data_train,data_test,col,model)


    intersection_train = get_missing_data(data_train[list_missing_cols_categ])
    intersection_test = get_missing_data(data_test[list_missing_cols_categ])
    intersection = pd.concat([intersection_train,intersection_test]).index.drop_duplicates()
    
    intersection_model = KNeighborsClassifier()

    for col in intersection:
        predict_intersection(data_train,data_test,col,intersection_model)

    list_missing_cols = get_missing_cols(data_train)
    list_missing_cols = list_missing_cols.union(get_missing_cols(data_test))

    model = KNeighborsRegressor()

    for col in list_missing_cols:
        model = train_model(data_train,col,model)
        predict_values(data_train,data_test,col,model)

    intersection_train = get_missing_data(data_train[list_missing_cols])
    intersection_test = get_missing_data(data_test[list_missing_cols])

    intersection = pd.concat([intersection_train,intersection_test]).index.drop_duplicates()
   

    intersection_model = KNeighborsRegressor()

    for col in intersection:
        predict_intersection(data_train,data_test,col,intersection_model)

    

    columns_to_binarize = ['MSZoning', 'Utilities', 'Exterior1st', 'SaleType', 'Functional', 'Electrical', 'MasVnrType']
    data_train,data_test = binarize_data(data_train,data_test,columns_to_binarize)
    
    data_train['remainder__Id'] = train['remainder__Id'].astype(int)
    data_test['remainder__Id'] = test['remainder__Id'].astype(int)

    data_train['SalePrice'] = target

    
    data_train.to_csv('dataset/processed/train_encoded_imputed.csv',index=False)
    data_test.to_csv('dataset/processed/test_encoded_imputed.csv',index=False)

if __name__ == '__main__':
    main()

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







 

