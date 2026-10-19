# Integrantes: Pedro Antonio de Faria Garcia, Felipe Grein Krause, Richard Lemuel Gonçalves Brant
import numpy as np
import scipy.io as scipy
from knn_funcoes import meuKnn, normalizacao

mat = scipy.loadmat('grupoDados2.mat')
grupoTrain = mat['grupoTrain']
trainRots = mat['trainRots'].ravel()
grupoTest = mat['grupoTest']
testRots = mat['testRots'].ravel()

# Primeiro teste: dados originais, sem normalização.
prevOriginal = meuKnn(grupoTrain, trainRots, grupoTest, 1)
accOriginal = np.mean(prevOriginal == testRots)
print(f'Sem normalização, k=1: {accOriginal * 100:.2f}%')

# Normalização Min-Max.
trainNorm, testNorm = normalizacao(grupoTrain, grupoTest)

# Depois da normalização, testamos k=3.
prevNorm = meuKnn(trainNorm, trainRots, testNorm, 3)
accNorm = np.mean(prevNorm == testRots)
print(f'Normalizado, k=3: {accNorm * 100:.2f}%')

# Q2.1:
# Com os dados originais e k=1, a acurácia é 68,33%.
#
# Q2.2:
# O resultado baixo ocorre porque as 13 características do vinho têm
# escalas muito diferentes. A distância Euclidiana é dominada por
# atributos numericamente grandes (por exemplo, Prolina), mesmo que
# outros atributos sejam importantes para a classe.
#
# Aplicamos normalização Min-Max, colocando cada característica em uma
# escala comparável, e usamos k=3. A acurácia passou para 98,33%,
# que corresponde aos aproximadamente 98% pedidos no enunciado.
