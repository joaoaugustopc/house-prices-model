import sys
from src.data_prep import encode_1, data_missing_3

#para a preparação de dados, primeiro argumento: data_prep
#para execução de modelo, primeiro argumeto: model, segundo arumento: nome do arquivo do modelo

if sys.argv[1] == 'data_prep':
    encode_1.main()
    data_missing_3.main()
