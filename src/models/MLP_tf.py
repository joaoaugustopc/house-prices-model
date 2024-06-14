import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from utils.data_transformation import load_data

def main():
    data_train, data_test = load_data('train_encoded_imputed', 'test_encoded_imputed')
    X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1).values
    y_train = data_train['SalePrice'].values
    X_test = data_test.drop(["remainder__Id"], axis=1).values
    
    model = keras.Sequential([
        keras.layers.Dense(1024, activation="relu", input_shape=[X_train.shape[1]]),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=300)

    y_predict = model.predict(X_test)

    print(y_predict)
    print(y_predict.shape)

    y_predict = y_predict.flatten()

    print(y_predict)
    print(y_predict.shape)

    result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

    result['Id'] = result['Id'].astype(int)
    result.to_csv('submissions/sample_submission_tf.csv', index=False)


if __name__ == '__main__':
    main()
