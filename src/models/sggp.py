from ..SGGP.genetic import SymbolicRegressor
from ..utils.data_transformation import load_data
import pandas as pd

def main():
    data_train, data_test = load_data('train_encoded_imputed', 'test_encoded_imputed')
    X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1)
    y_train = data_train['SalePrice']
    X_test = data_test.drop(["remainder__Id"], axis=1)
    
    est_gp = SymbolicRegressor()
    
    est_gp.fit(X_train, y_train)
        
    y_predict = est_gp.predict(X_test)
    
    result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

    result['Id'] = result['Id'].astype(int)
    result.to_csv('submissions/sample_submission_sggp.csv', index=False)

if __name__ == '__main__':
    main()