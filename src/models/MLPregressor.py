from sklearn.neural_network import MLPRegressor
import numpy as np
from utils.data_transformation import load_data
import pandas as pd

def main():
    data_train, data_test = load_data('train_encoded_imputed', 'test_encoded_imputed')
    X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1).values
    y_train = data_train['SalePrice'].values
    X_test = data_test.drop(["remainder__Id"], axis=1).values
    
    model = MLPRegressor(hidden_layer_sizes=(1024), activation='relu', solver='adam', batch_size='auto',
                          max_iter=300, random_state=None, verbose=True, learning_rate_init=0.01,warm_start=True, tol = 1e-5)

    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)

    print("Score:",model.score(X_train, y_train))
       
    result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

    result['Id'] = result['Id'].astype(int)

    result.to_csv('submissions/sample_submission_mlp.csv', index=False)

if __name__ == '__main__':
    main()
