from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('data/pre_data_train.csv')
data_test = pd.read_csv('data/pre_data_test.csv')

X_train = data_train.drop(["Id", "SalePrice"], axis=1)
y_train = data_train["SalePrice"]
X_test = data_test.drop(["Id"], axis=1)

#treinando modelo
house_price_model = linear_model.LinearRegression()

house_price_model.fit(X_train, y_train)

house_pred = house_price_model.predict(X_test)

#modelo / coeficientes
print("Predicted house prices: ", house_price_model.coef_)
