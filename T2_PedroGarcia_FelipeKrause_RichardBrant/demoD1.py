# Integrantes: Pedro Antonio de Faria Garcia, Felipe Grein Krause, Richard Lemuel Gonçalves Brant
import numpy as np
import scipy.io as scipy
from itertools import combinations
from knn_funcoes import meuKnn, visualizaPontos

mat = scipy.loadmat('grupoDados1.mat')
grupoTrain = mat['grupoTrain']
trainRots = mat['trainRots'].ravel()
grupoTest = mat['grupoTest']
testRots = mat['testRots'].ravel()

# Teste pedido pelo enunciado: 1-NN.
prev1 = meuKnn(grupoTrain, trainRots, grupoTest, 1)
acc1 = np.mean(prev1 == testRots)
print(f'k=1: {acc1 * 100:.2f}%')

# Teste pedido pelo enunciado: k=10.
prev10 = meuKnn(grupoTrain, trainRots, grupoTest, 10)
acc10 = np.mean(prev10 == testRots)
print(f'k=10: {acc10 * 100:.2f}%')

# Procuramos o melhor k usando todas as 4 características.
melhorAcc = 0
melhorK = None
for k in range(1, 31):
    prev = meuKnn(grupoTrain, trainRots, grupoTest, k)
    acc = np.mean(prev == testRots)
    if acc > melhorAcc:
        melhorAcc = acc
        melhorK = k

print(f'Melhor com todas as características: k={melhorK}, acurácia={melhorAcc * 100:.2f}%')

# Teste usando somente características 3 e 4 (índices 2 e 3 em Python).
trainPetalas = grupoTrain[:, [2, 3]]
testPetalas = grupoTest[:, [2, 3]]
prevPetalas = meuKnn(trainPetalas, trainRots, testPetalas, 4)
accPetalas = np.mean(prevPetalas == testRots)
print(f'Somente características 3 e 4, k=4: {accPetalas * 100:.2f}%')

visualizaPontos(grupoTrain, trainRots, 2, 3)

# Q1.1:
# A acurácia máxima encontrada foi 98%, por exemplo com todas as
# características e k=3 (também ocorre para outros valores de k).
#
# Q1.2:
# Não é necessário utilizar todas as características.
# Usando somente as características 3 e 4 e k=4, também obtemos 98%.
# Isso mostra que algumas características são mais informativas para
# separar as três classes de flores do que outras.
