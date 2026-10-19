"""Trabalho 3 — Parte 2: Árvores de Decisão e Random Forest."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier


PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_RISCO = PASTA_PROJETO.parent / "Parte-1" / "risco_credito.pkl"
CAMINHO_CREDIT = PASTA_PROJETO / "credit.pkl"


def carregar_pickle(caminho: Path):
    """Carrega um arquivo Pickle fornecido pelo projeto."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    with caminho.open("rb") as arquivo:
        return pickle.load(arquivo)


def salvar_arvore(
    modelo,
    atributos: list[str],
    classes: list[str],
    nome_arquivo: str,
    tamanho: tuple[int, int],
) -> None:
    """Gera e salva uma representação gráfica de uma árvore."""
    figura, eixo = plt.subplots(figsize=tamanho)
    tree.plot_tree(
        modelo,
        feature_names=atributos,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=eixo,
    )
    figura.tight_layout()
    figura.savefig(PASTA_PROJETO / nome_arquivo, dpi=140, bbox_inches="tight")
    plt.close(figura)


def salvar_matriz_confusao(
    matriz: np.ndarray,
    titulo: str,
    nome_arquivo: str,
) -> None:
    """Gera e salva uma matriz de confusão."""
    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Paga (0)", "Não paga (1)"],
    )
    exibicao.plot(cmap="Blues", values_format="d")
    exibicao.ax_.set_title(titulo)
    exibicao.figure_.tight_layout()
    exibicao.figure_.savefig(
        PASTA_PROJETO / nome_arquivo,
        dpi=140,
        bbox_inches="tight",
    )
    plt.close(exibicao.figure_)


def analisar_matriz(matriz: np.ndarray) -> dict[str, int]:
    """Nomeia as quatro posições da matriz de confusão binária."""
    verdadeiros_negativos, falsos_positivos, falsos_negativos, verdadeiros_positivos = (
        matriz.ravel()
    )
    return {
        "pagantes classificados corretamente": int(verdadeiros_negativos),
        "pagantes classificados como não pagantes": int(falsos_positivos),
        "não pagantes classificados como pagantes": int(falsos_negativos),
        "não pagantes classificados corretamente": int(verdadeiros_positivos),
    }


def main() -> None:
    """Executa todas as etapas da Parte 2."""
    # 1. Árvore de Decisão na base didática de risco de crédito.
    X_risco_credito, y_risco_credito = carregar_pickle(CAMINHO_RISCO)

    arvore_risco_credito = DecisionTreeClassifier(
        criterion="entropy",
        random_state=0,
    )
    arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

    atributos_risco = ["historia", "divida", "garantias", "renda"]
    importancias_risco = pd.Series(
        arvore_risco_credito.feature_importances_,
        index=atributos_risco,
        name="importancia",
    ).sort_values(ascending=False)

    print("ÁRVORE — BASE RISCO DE CRÉDITO")
    print("\nImportância dos atributos:")
    print(importancias_risco.to_string())
    print(f"\nMaior ganho de informação: {importancias_risco.index[0]}")

    salvar_arvore(
        arvore_risco_credito,
        atributos_risco,
        [str(classe) for classe in arvore_risco_credito.classes_],
        "arvore_risco_credito.png",
        (18, 9),
    )

    novos_clientes = np.array(
        [
            [0, 0, 1, 2],  # história boa, dívida alta, sem garantia, renda > 35
            [2, 0, 0, 0],  # história ruim, dívida alta, garantia adequada, renda < 15
        ]
    )
    previsoes_risco = arvore_risco_credito.predict(novos_clientes)
    print("\nPrevisões dos clientes:")
    print(pd.DataFrame({"cliente": ["i", "ii"], "risco": previsoes_risco}).to_string(index=False))

    # 2. Árvore de Decisão na base Credit Data.
    (
        X_credit_treinamento,
        y_credit_treinamento,
        X_credit_teste,
        y_credit_teste,
    ) = carregar_pickle(CAMINHO_CREDIT)

    print("\n\nÁRVORE — BASE CREDIT DATA")
    print("Dimensões dos dados:")
    print("X treinamento:", X_credit_treinamento.shape)
    print("y treinamento:", y_credit_treinamento.shape)
    print("X teste:", X_credit_teste.shape)
    print("y teste:", y_credit_teste.shape)

    arvore_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)
    arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)
    previsoes_arvore = arvore_credit.predict(X_credit_teste)

    acuracia_arvore = accuracy_score(y_credit_teste, previsoes_arvore)
    matriz_arvore = confusion_matrix(y_credit_teste, previsoes_arvore, labels=[0, 1])
    print(f"\nAcurácia: {acuracia_arvore:.4f} ({acuracia_arvore:.2%})")
    print("\nMatriz de confusão:")
    print(matriz_arvore)
    print("\nInterpretação da matriz:")
    for descricao, quantidade in analisar_matriz(matriz_arvore).items():
        print(f"- {descricao}: {quantidade}")
    print("\nRelatório de classificação:")
    print(classification_report(y_credit_teste, previsoes_arvore, digits=4))

    salvar_matriz_confusao(
        matriz_arvore,
        "Matriz de confusão — Árvore de Decisão",
        "matriz_confusao_arvore_credit.png",
    )
    salvar_arvore(
        arvore_credit,
        ["income", "age", "loan"],
        ["Paga", "Não paga"],
        "arvore_credit.png",
        (32, 18),
    )

    # 3. Random Forest na base Credit Data.
    random_forest_credit = RandomForestClassifier(
        n_estimators=10,
        criterion="entropy",
        random_state=0,
    )
    random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)
    previsoes_forest = random_forest_credit.predict(X_credit_teste)

    acuracia_forest = accuracy_score(y_credit_teste, previsoes_forest)
    matriz_forest = confusion_matrix(y_credit_teste, previsoes_forest, labels=[0, 1])
    print("\n\nRANDOM FOREST — 10 ÁRVORES")
    print(f"Acurácia: {acuracia_forest:.4f} ({acuracia_forest:.2%})")
    print("\nMatriz de confusão:")
    print(matriz_forest)
    print("\nInterpretação da matriz:")
    for descricao, quantidade in analisar_matriz(matriz_forest).items():
        print(f"- {descricao}: {quantidade}")
    print("\nRelatório de classificação:")
    print(classification_report(y_credit_teste, previsoes_forest, digits=4))

    salvar_matriz_confusao(
        matriz_forest,
        "Matriz de confusão — Random Forest (10 árvores)",
        "matriz_confusao_random_forest.png",
    )

    # Comparação solicitada: aumento do número de árvores.
    random_forest_credit_40 = RandomForestClassifier(
        n_estimators=40,
        criterion="entropy",
        random_state=0,
    )
    random_forest_credit_40.fit(X_credit_treinamento, y_credit_treinamento)
    previsoes_forest_40 = random_forest_credit_40.predict(X_credit_teste)
    acuracia_forest_40 = accuracy_score(y_credit_teste, previsoes_forest_40)

    print("\nCOMPARAÇÃO DOS MODELOS")
    comparacao = pd.DataFrame(
        {
            "modelo": [
                "Árvore de Decisão",
                "Random Forest (10 árvores)",
                "Random Forest (40 árvores)",
            ],
            "acuracia": [
                acuracia_arvore,
                acuracia_forest,
                acuracia_forest_40,
            ],
        }
    )
    print(comparacao.to_string(index=False, formatters={"acuracia": "{:.2%}".format}))

    if acuracia_forest < acuracia_arvore:
        print(
            "\nCom 10 árvores, a Random Forest foi inferior à árvore simples. "
            "Aumentar n_estimators reduz a variância da floresta; com 40 árvores, "
            f"a acurácia passou para {acuracia_forest_40:.2%}."
        )


if __name__ == "__main__":
    main()
