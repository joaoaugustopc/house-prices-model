from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#importando os dados

#Melhores parâmetros: {'learning_rate': 0.05, 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}

data_train = pd.read_csv('dataset/train_scaled.csv')
data_test = pd.read_csv('dataset/test_scaled.csv')

#print(data_train.head())

target = "SalePrice"

X_train = data_train.drop(["remainder__Id", "SalePrice"], axis=1)
y_train = data_train[target]
X_test = data_test.drop(["remainder__Id"], axis=1).select_dtypes(include=["number"])

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Definindo o modelo
reg = GradientBoostingRegressor()


# Definindo os parâmetros que queremos testar
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.1],
    'max_depth': [4],
    'min_samples_split': [4, 5],
    'min_samples_leaf': [1, 2]
}

# Criando o GridSearchCV
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)

# Ajustando o GridSearchCV
grid_search.fit(X_train, y_train)

# Melhores parâmetros
print("Melhores parâmetros:", grid_search.best_params_)

# Melhor score
print("Melhor score:", grid_search.best_score_)

# Usando o melhor modelo para previsões
y_pred = grid_search.predict(X_test)

#print("Predicted house prices: ", feature_importances.ravel())

result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_pred})

result['Id'] = result['Id'].astype(int)

result.to_csv('submissions/sample_submission_grad_boost_encoded_imputed_scaled_0.1_500.csv', index=False)

#print("Predicted house prices: ", reg.score(X_train, y_train))

#relação entre as features e a variável target
# Selecionar as top N features mais importantes

#evaluating errors
mse = mean_squared_error(y_train, grid_search.predict(X_train))
caminho_file = "graficos/gradient_boost/analise.txt"
text_analise = "Mean Squared Error: " + str(mse) + "\n" + str(grid_search.best_params_)
with open(caminho_file, "a") as arquivo:
    arquivo.write(text_analise)

""" 
top_n = 30
sorted_idx = np.argsort(feature_importances)[-top_n:]

pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importances[sorted_idx], align="center")
plt.yticks(pos, np.array(X_test.columns)[sorted_idx])
plt.title("Top 30 Feature Importance (MDI)")

plt.savefig('graficos/top30_feature_importance_grad_boosting.png')
"""