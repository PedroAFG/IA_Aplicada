# Integrantes: Pedro Antonio de Faria Garcia, Felipe Grein Krause, Richard Lemuel Gonçalves Brant
import numpy as np
import scipy.io as scipy
from knn_funcoes import meuKnn, visualizaPontos

mat = scipy.loadmat('grupoDados3.mat')
grupoTrain = mat['grupoTrain']
trainRots = mat['trainRots'].ravel()
grupoTest = mat['grupoTest']
testRots = mat['testRots'].ravel()

# Q3.1: k=1, como a prof pede .
prev1 = meuKnn(grupoTrain, trainRots, grupoTest, 1)
acc1 = np.mean(prev1 == testRots)
print(f'k=1: {acc1 * 100:.2f}%')

# Ajuste de k.
prev7 = meuKnn(grupoTrain, trainRots, grupoTest, 7)
acc7 = np.mean(prev7 == testRots)
print(f'k=7: {acc7 * 100:.2f}%')

visualizaPontos(grupoTrain, trainRots, 0, 1)

# Q3.1:
# Com k=1, a acurácia é 62%.
#
# Q3.2:
# O 1-NN é muito sensível a ruído, exemplos atípicos e regiões em que
# as classes se sobrepõem. Ao aumentar k, a decisão passa a considerar
# vários vizinhos e fica menos dependente de um único ponto.
# Com k=7, a acurácia chega a 92%.
