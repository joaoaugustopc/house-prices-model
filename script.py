import sys
from src.data_prep import encode_1, data_missing_3
from src.models import gradient_boosting_reg

#para a preparação de dados, primeiro argumento: data_prep
#para execução de modelo, primeiro argumeto: model, segundo arumento: nome do arquivo do modelo

if len(sys.argv) > 1:
    if sys.argv[1] == 'data_prep':
        encode_1.main()
        data_missing_3.main()
        if len(sys.argv) > 2:
            if sys.argv[2] == 'model':
                if sys.argv[3] == 'grad_boost':
                    print('Gradient Boosting Regressor')
                    gradient_boosting_reg.main()
                
           


