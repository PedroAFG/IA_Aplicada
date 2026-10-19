# Integrantes: Pedro Antonio de Faria Garcia, Felipe Grein Krause, Richard Lemuel Gonçalves Brant
import numpy as np
import scipy.io as scipy
from knn_funcoes import meuKnn, normalizacao

mat = scipy.loadmat('grupoDados4.mat')

# dataset com nomes das variáveis são diferentes.
grupoTest = mat['testSet']
grupoTrain = mat['trainSet']
testRots = mat['testLabs'].ravel()
trainRots = mat['trainLabs'].ravel()

# Situação inicial: todos os atributos, sem normalização, k=1.
# Lembrando, o "mean" basicamente pega o número de acertos e divide pelo total de exemplos. Ele consulta indice por indice verificando se o número é igual, então monta um novo vetor com True e False.
prevOriginal = meuKnn(grupoTrain, trainRots, grupoTest, 1)
accOriginal = np.mean(prevOriginal == testRots)
print(f"prevOriginal (previsões): {prevOriginal}")
print(f"testRots (rótulos reais): {testRots}")

print(f'Original, todas as características, k=1: {accOriginal * 100:.2f}%')

# Há dois problemas neste conjunto:
# 1) as características têm escalas muito diferentes;
# PORQUÊ?
# Verificando os valores que existem no grupo:
# Característica 1:
# mínimo = 0,34
# máximo = 5,08
#
# Característica 2:
# mínimo = 278
# máximo = 1547
#
# Característica 3:
# mínimo ≈ 0,006
# máximo ≈ 0,989
#
# Característica 4:
# mínimo ≈ 0,007
# máximo ≈ 0,988
#
# Ou seja, a caracteristica 2 trabalha na casa das centenas e milhares, enquanto as outras perto de 0 a 5. 
# Isso faz com que a característica 2 seja muito mais relevante na hora de calcular a distância Euclidiana, e as outras características praticamente não influenciam.
# Ou seja, a característica 2 passa a decidir quase toda a noção de "proximidade".
#
# 2) as características 3 e 4 são atributos irrelevantes/ruído.
# PORQUÊ?
#
# Classe 1 → média ≈ 0,47
# Classe 2 → média ≈ 0,49
# Classe 3 → média ≈ 0,53
#
# Literalmente quase tudo na mesma região, então não há como diferenciar as classes com base nessas características.
#
#
# Mantemos somente as duas primeiras características.
trainSelecionado = grupoTrain[:, [0, 1]]
testSelecionado = grupoTest[:, [0, 1]]

# Depois normalizamos.
trainNorm, testNorm = normalizacao(trainSelecionado, testSelecionado)

# Looping pra encontrar o melhor valor de k, que se aproxime de 92%

# Procuramos o primeiro valor de k que produza
# uma acurácia aproximadamente igual a 92%.

kEscolhido = None
accEscolhida = None

for k in range(1, 21):
    previsto = meuKnn(trainNorm, trainRots, testNorm, k)
    acc = np.mean(previsto == testRots)

    print(f'k={k:2d} -> {acc * 100:.2f}%')

    # 91,67% é aproximadamente 92%.
    if abs(acc - 0.92) < 0.005:
        kEscolhido = k
        accEscolhida = acc
        break

print(f'\nk escolhido: {kEscolhido}')
print(f'Acurácia: {accEscolhida * 100:.2f}%')

# Q4.1:
# Com todos os atributos, sem normalização e k=1, a acurácia é 71,67%.
#
# Q4.2:
# Desta vez existem dois problemas: diferença de escala e atributos
# irrelevantes. Retiramos as características 3 e 4, normalizamos as
# características 1 e 2 com Min-Max e usamos k=3.
# O resultado é 91,67%, que arredondado corresponde a 92%.
