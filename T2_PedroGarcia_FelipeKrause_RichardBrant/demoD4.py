# Integrantes: Pedro Antonio de Faria Garcia, Felipe Grein Krause, Richard Lemuel Gonçalves Brant
import numpy as np
import scipy.io as scipy
from knn_funcoes import meuKnn, normalizacao

mat = scipy.loadmat('grupoDados4.mat')

# Neste dataset os nomes das variáveis são diferentes.
grupoTest = mat['testSet']
grupoTrain = mat['trainSet']
testRots = mat['testLabs'].ravel()
trainRots = mat['trainLabs'].ravel()

# Situação inicial: todos os atributos, sem normalização, k=1.
prevOriginal = meuKnn(grupoTrain, trainRots, grupoTest, 1)
accOriginal = np.mean(prevOriginal == testRots)
print(f'Original, todas as características, k=1: {accOriginal * 100:.2f}%')

# Há dois problemas neste conjunto:
# 1) as características têm escalas muito diferentes;
# 2) as características 3 e 4 são atributos irrelevantes/ruído.
#
# Mantemos somente as duas primeiras características.
trainSelecionado = grupoTrain[:, [0, 1]]
testSelecionado = grupoTest[:, [0, 1]]

# Depois normalizamos.
trainNorm, testNorm = normalizacao(trainSelecionado, testSelecionado)

# k=3 fornece 55 acertos em 60 exemplos = 91,67%, aproximadamente 92%.
prevFinal = meuKnn(trainNorm, trainRots, testNorm, 3)
accFinal = np.mean(prevFinal == testRots)
print(f'Seleção de atributos + normalização, k=3: {accFinal * 100:.2f}%')

# Q4.1:
# Com todos os atributos, sem normalização e k=1, a acurácia é 71,67%.
#
# Q4.2:
# Desta vez existem dois problemas: diferença de escala e atributos
# irrelevantes. Retiramos as características 3 e 4, normalizamos as
# características 1 e 2 com Min-Max e usamos k=3.
# O resultado é 91,67%, que arredondado corresponde a 92%.
