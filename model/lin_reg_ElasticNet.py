from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
import pandas as pd


data_train = pd.read_csv('data/pre_data_train.csv')
data_test = pd.read_csv('data/pre_data_test.csv')

target = "SalePrice"

X_train = data_train.drop(["Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["Id"], axis=1).select_dtypes(include=["number"])


# Utilizando validação Cruzada para determinar o melhor hiperPâmetro (alpha) para o ElasticNet

l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

elastic = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, cv=5)

elastic.fit(X_train, y_train)

best_alpha = elastic.alpha_
best_l1_ratio = elastic.l1_ratio_

print("Melhor alpha: ", best_alpha)
print("Melhor l1_ratio: ", best_l1_ratio)

# Treinando o modelo com os melhores hiperparâmetros

model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

result = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': y_pred})

result.to_csv('data/sample_submission_elastic.csv', index=False)