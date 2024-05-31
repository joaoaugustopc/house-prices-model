from ..SGGP.genetic import SymbolicRegressor
import numpy as np
from ..utils.data_transformation import load_data
import pandas as pd

def main():
    data_train, data_test = load_data('train_encoded_imputed', 'test_encoded_imputed')
    X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1)
    y_train = data_train['SalePrice']
    X_test = data_test.drop(["remainder__Id"], axis=1)
    
    X_train_np = X_train.to_numpy(dtype=np.float64)
    y_train_np = y_train.to_numpy(dtype=np.float64)
    X_test_np = X_test.to_numpy(dtype=np.float64)

    
    est_gp = SymbolicRegressor(population_size=100,generations=10, stopping_criteria=1e-5,
                        verbose=1, random_state=None,n_features=X_train_np.shape[1], metric='mse')

    est_gp.fit(X_train_np, y_train_np)
    
    y_predict = est_gp.predict(X_test_np)
    
    y_predict = pd.DataFrame(y_predict)
    y_predict.to_csv('submissions/sample_submission_sggp.csv', index=False)
    
""" 
    result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

    result['Id'] = result['Id'].astype(int)
    result.to_csv('submissions/sample_submission_sggp.csv', index=False)
"""

if __name__ == '__main__':
    main()