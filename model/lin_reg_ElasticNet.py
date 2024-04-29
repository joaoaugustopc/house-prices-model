from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
import pandas as pd


data_train = pd.read_csv('data/train_prep_encoded.csv')
data_test = pd.read_csv('data/test_prep_encoded.csv')

target = "SalePrice"

X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["remainder__Id"], axis=1)


# Utilizando validação Cruzada para determinar o melhor hiperPâmetro (alpha) para o ElasticNet

#l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
#alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95,0.97, 0.99, 1.0, 0.4, 0.45, 0.55, 0.6,0.62,0.12,0.70, 0.58,0.59, 0.65,0.13,0.11,0.15,0.17,0.19,0.21, 0.125, 0.115,0.117,0.113,0.118,0.116]
alphas = [1,2.5,3,3.5,4,5,7,9,10,12,14,16,20,25,30,40,50]

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

result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_pred})

result['Id'] = result['Id'].astype(int)

result.to_csv('submissions/sample_submission_elastic_prep_encoded.csv', index=False)