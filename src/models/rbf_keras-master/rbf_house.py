import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import RMSprop
from .rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from utils.data_transformation import load_data

def main():
    data_train, data_test = load_data('train_encoded_imputed', 'test_encoded_imputed')
    X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1)
    y_train = data_train['SalePrice']
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    
    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=InitCentersRandom(X_train),
                        betas=2.0,
                        input_shape=(1,))
    model.add(rbflayer)
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    
    model.fit(X_train, y_train, batch_size=50, epochs=2000, verbose=1)
    
    y_pred = model.predict(X_train)
    
    print(rbflayer.get_weights())
    
    plt.plot(X_train, y_pred)
    plt.plot(X_train, y_train)
    plt.plot([-1, 1], [0, 0], color='black')
    plt.xlim([-1, 1])
    
    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    plt.scatter(centers, np.zeros(len(centers)), s=20*widths)
    plt.show()
    
if __name__ == "__main__":
    main()
    