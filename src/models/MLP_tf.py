import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import keras_tuner as kt
from utils.data_transformation import load_data

def build_model(hp, imput_shape):
    model = keras.Sequential()
    hidden_layers = hp.Int('hidden_layers', 1, 3)
    units = hp.Int('units', min_value=32, max_value=1024, step=256)
    activation = hp.Choice('activation', values=['relu','elu','selu'])
    model.add(keras.layers.Dense(units=units, activation=activation, input_shape= imput_shape))

    for i in range(hidden_layers - 1):
        units = hp.Int(f'units_{i}', min_value=32, max_value=1024, step=256)
        model.add(keras.layers.Dense(units=units, activation=activation))
    
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def main():
    data_train, data_test = load_data('train_encoded_imputed', 'test_encoded_imputed')
    X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1).values
    y_train = data_train['SalePrice'].values
    X_test = data_test.drop(["remainder__Id"], axis=1).values
    
    """
    model = keras.Sequential([
        keras.layers.Dense(1024, activation="relu", input_shape=[X_train.shape[1]]),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    """

    imput_shape = [X_train.shape[1]]

    """
     - RandomSearch -> (explora aleatoriamente o espaco de hiperparametros), 
     - Hyperband -> Combina RandomSearch com o conceito de early stopping, avaliando rapidamente muitos conjuntos de hiperparâmetros e interrompendo aqueles que não parecem promissores.
     - BayesianOptimization -> Utiliza técnicas de otimização bayesiana para selecionar os próximos hiperparâmetros a serem avaliados com base nos resultados anteriores.
        Vantagens: Geralmente mais eficiente que o RandomSearch e o Grid Search, pois foca em partes promissoras do espaço de busca    
     - gridSearch -> Explora sistematicamente todas as combinações possíveis dos hiperparâmetros fornecidos.
    """

    tuner = kt.RandomSearch( 
        lambda hp: build_model(hp, imput_shape),
        objective='val_loss',
        #max_epochs=50,
        max_trials=10,
        executions_per_trial=3,
        directory='my_dir',
        project_name='house_prices',
    )

    """
    tuner.reload()

    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Os melhores hiperparâmetros são:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")

    """

    checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss')

    early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')

    tuner.search(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[checkpoint, early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1, callbacks=[checkpoint, early_stopping])
    

    best_model = keras.models.load_model("best_model.keras")

    val_loss_history = history.history['val_loss']
    loss_history = history.history['loss']

    print("evaluate: ")
    best_model.evaluate(X_train, y_train)

    print("Menor val_loss alcançado:", min(val_loss_history))
    print("Menor loss de treinamento alcançado:", min(loss_history))

    y_predict = best_model.predict(X_test)
    y_predict = y_predict.flatten()

    
    result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

    result['Id'] = result['Id'].astype(int)
    result.to_csv('submissions/sample_submission_tf.csv', index=False)


if __name__ == '__main__':
    main()
