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

# Criar um DataFrame com os IDs e as previsões
result = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': house_pred})

# Escrever o DataFrame em um arquivo CSV
result.to_csv('data/sample_submission_lin.csv', index=False)
