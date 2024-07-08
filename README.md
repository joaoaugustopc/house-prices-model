ATENÇÃO A ALGUMAS MUDANÇAS:

EM src/data_prep/encode.py

-Todo dado ordinal que possui valor faltante terá "miss_cat\_\_ " em seu nome

EM src/utils/data_transformation.py

- Possui conjunto de funções para serem utilizadas em todo código
- Somente criar funções genéricas no contexto da competição e seus dados específicos
- Mais informações estão disponíveis como comentário em cada função
- Arquivo que continha standarScale ou normalização foi substituído por função em utils
