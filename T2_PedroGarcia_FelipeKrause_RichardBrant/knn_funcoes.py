# Integrantes: Pedro Antonio de Faria Garcia, Felipe Grein Krause, Richard Lemuel Gonçalves Brant
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def dist(a, b):
    """
    Calcula a distância Euclidiana entre dois vetores.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2))


def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    """
    Classificador k-Nearest Neighbour (k-NN) implementado do zero.
    """
    dadosTrain = np.asarray(dadosTrain, dtype=float)
    dadosTeste = np.asarray(dadosTeste, dtype=float)
    rotuloTrain = np.asarray(rotuloTrain).ravel()

    rotuloPrevisto = []

    # Para cada exemplo do conjunto de teste...
    for exemploTeste in dadosTeste:

        # Calcula a distância do exemplo de teste para
        # TODOS os exemplos do conjunto de treinamento.
        distancias = np.array([
            dist(exemploTeste, exemploTrain)
            for exemploTrain in dadosTrain
        ])

        # argsort() retorna os índices que colocariam
        # as distâncias em ordem crescente.
        indicesOrdenados = np.argsort(distancias)

        # Selecionamos somente os k vizinhos mais próximos.
        indicesVizinhos = indicesOrdenados[:k]

        # Descobrimos os rótulos desses k vizinhos. A partir daqui já estamos trabalhando apenas com os vizinhos mais próximos do k
        rotulosVizinhos = rotuloTrain[indicesVizinhos]

        # A classe prevista é a moda dos rótulos dos vizinhos.
        classePrevista = mode(rotulosVizinhos, keepdims=False).mode

        rotuloPrevisto.append(classePrevista)

    return np.asarray(rotuloPrevisto)


def normalizacao(dadosTrain, dadosTeste):
    """
    Normalização Min-Max.

    obs pra lembrar:
    os valores mínimo e máximo são calculados SOMENTE no conjunto
    de treinamento. Ent depois disso, os mesmos valores são usados no teste.
    Isso evita vazamento de informação (data leakage).
    """
    dadosTrain = np.asarray(dadosTrain, dtype=float)
    dadosTeste = np.asarray(dadosTeste, dtype=float)

    minimo = np.min(dadosTrain, axis=0)
    maximo = np.max(dadosTrain, axis=0)
    amplitude = maximo - minimo

    # Evita divisão por zero caso alguma característica seja constante.
    amplitude[amplitude == 0] = 1

    trainNormalizado = (dadosTrain - minimo) / amplitude
    testeNormalizado = (dadosTeste - minimo) / amplitude

    return trainNormalizado, testeNormalizado


def visualizaPontos(dados, rotulos, d1, d2):
    """
    Mostra duas características do conjunto de dados.

    Em Python os índices começam em zero:
    d1=0 -> primeira característica
    d2=1 -> segunda característica
    """
    dados = np.asarray(dados)
    rotulos = np.asarray(rotulos).ravel()

    fig, ax = plt.subplots()

    marcadores = {1: '^', 2: '+', 3: '.'}

    for classe in np.unique(rotulos):
        mascara = rotulos == classe
        ax.scatter(
            dados[mascara, d1],
            dados[mascara, d2],
            marker=marcadores.get(int(classe), 'o'),
            label=f'Classe {classe}'
        )

    ax.set_xlabel(f'Característica {d1 + 1}')
    ax.set_ylabel(f'Característica {d2 + 1}')
    ax.set_title('Visualização dos dados')
    ax.legend()
    plt.show()
