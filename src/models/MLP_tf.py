import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import keras_tuner as kt
from utils.data_transformation import load_data
import matplotlib.pyplot as plt

def plotHistory(history):

  fig, ax = plt.subplots(1,2,figsize=(26,10))

  # Imprime a curva de aprendizado
  ax[0].set_title('Mean Squared Percentage Error', pad=-40)
  ax[0].plot(history.history['loss'], label='train')
  ax[0].plot(history.history['val_loss'], label='valid')
  ax[0].legend(loc='best')

  # Imprime a curva de acurácia
  ax[1].set_title('Mean Squared Error', pad=-40)
  ax[1].plot(history.history['mse'], label='train')
  ax[1].plot(history.history['val_mse'], label='valid')
  ax[1].legend(loc='best')

  plt.show()


def build_model(hp, imput_shape):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units= hp.Int(f"units", min_value=64, max_value=4096, step=256),
                                 activation= hp.Choice("activation", ["relu","elu","selu","linear"]), 
                                 input_shape= imput_shape))
    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate = 0.2))

    """
    model.add(keras.layers.InputLayer(input_shape=imput_shape))
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i}", [64,256,512,1024,2048,4096,6000,8192,10000]),
                activation=hp.Choice("activation", ["relu", "elu","selu","linear"]),
            )
        )
    """
    model.add(keras.layers.Dense(1))
    
    lr = hp.Choice("learning_rate", [1e-1, 1e-2, 1e-3, 1e-4])
    opt = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
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

    tuner = kt.BayesianOptimization( 
        lambda hp: build_model(hp, imput_shape),
        objective='val_loss',
        max_trials=100,
        executions_per_trial=1,
        directory='my_dir',
        project_name='one_hidden_layer',
    )

    """
    tuner.reload()

    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Os melhores hiperparâmetros são:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")

    """

    #checkpoint = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss')

    early_stopping = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor='val_loss')

    tuner.search(X_train, y_train, epochs=1000, validation_split=0.1,callbacks=[early_stopping],batch_size=64)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1, callbacks=[early_stopping],batch_size=64)
    
    plotHistory(history)

    #best_model = keras.models.load_model("best_model.keras")

    val_loss_history = history.history['val_loss']
    loss_history = history.history['loss']

    print("evaluate: ")
    model.evaluate(X_train, y_train)

    print("Menor val_loss alcançado:", min(val_loss_history))
    print("Menor loss de treinamento alcançado:", min(loss_history))

    y_predict = model.predict(X_test)
    y_predict = y_predict.flatten()

    
    result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

    result['Id'] = result['Id'].astype(int)
    result.to_csv('submissions/sample_submission_tf.csv', index=False)


if __name__ == '__main__':
    main()
