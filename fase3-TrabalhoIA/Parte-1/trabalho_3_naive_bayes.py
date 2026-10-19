"""Trabalho 3 — Parte 1: classificação de risco de crédito com Naïve Bayes."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_CSV = PASTA_PROJETO / "risco_credito.csv"
CAMINHO_PKL = PASTA_PROJETO / "risco_credito.pkl"


def criar_encoder(valores: np.ndarray) -> tuple[LabelEncoder, np.ndarray]:
    """Ajusta um LabelEncoder e devolve o encoder e os valores codificados."""
    encoder = LabelEncoder()
    valores_codificados = encoder.fit_transform(valores)
    return encoder, valores_codificados


def exibir_mapeamento(nome: str, encoder: LabelEncoder) -> None:
    """Exibe o código numérico atribuído a cada categoria."""
    mapeamento = {
        categoria: int(codigo)
        for categoria, codigo in zip(
            encoder.classes_, encoder.transform(encoder.classes_)
        )
    }
    print(f"{nome}: {mapeamento}")


def main() -> None:
    """Executa todas as etapas do exercício."""
    if not CAMINHO_CSV.exists():
        raise FileNotFoundError(
            "O arquivo risco_credito.csv deve estar na mesma pasta deste script."
        )

    # Leitura e avaliação inicial da base de dados.
    dataset_risco_credito = pd.read_csv(CAMINHO_CSV)

    print("BASE DE DADOS")
    print(dataset_risco_credito.to_string(index=False))
    print(
        f"\nDimensões: {dataset_risco_credito.shape[0]} linhas e "
        f"{dataset_risco_credito.shape[1]} colunas"
    )
    print("\nValores ausentes por coluna:")
    print(dataset_risco_credito.isnull().sum().to_string())
    print("\nDistribuição da classe risco:")
    print(dataset_risco_credito["risco"].value_counts().to_string())

    # Divisão entre os atributos previsores (X) e a classe (y).
    X_risco_credito = dataset_risco_credito.iloc[:, 0:4].values
    y_risco_credito = dataset_risco_credito.iloc[:, 4].values

    print(f"\nFormato de X: {X_risco_credito.shape}")
    print(f"Formato de y: {y_risco_credito.shape}")

    # Codificação independente de cada atributo categórico.
    label_encoder_historia, X_risco_credito[:, 0] = criar_encoder(
        X_risco_credito[:, 0]
    )
    label_encoder_divida, X_risco_credito[:, 1] = criar_encoder(
        X_risco_credito[:, 1]
    )
    label_encoder_garantias, X_risco_credito[:, 2] = criar_encoder(
        X_risco_credito[:, 2]
    )
    label_encoder_renda, X_risco_credito[:, 3] = criar_encoder(
        X_risco_credito[:, 3]
    )
    X_risco_credito = X_risco_credito.astype(int)

    dados_codificados = pd.DataFrame(
        X_risco_credito,
        columns=["historia", "divida", "garantias", "renda"],
    )
    dados_codificados["risco"] = y_risco_credito

    print("\nBASE CODIFICADA")
    print(dados_codificados.to_string(index=False))
    print("\nMapeamentos utilizados:")
    exibir_mapeamento("historia", label_encoder_historia)
    exibir_mapeamento("divida", label_encoder_divida)
    exibir_mapeamento("garantias", label_encoder_garantias)
    exibir_mapeamento("renda", label_encoder_renda)

    # Salvamento dos dados pré-processados conforme solicitado no exercício.
    with CAMINHO_PKL.open("wb") as arquivo:
        pickle.dump([X_risco_credito, y_risco_credito], arquivo)
    print(f"\nArquivo pré-processado salvo em: {CAMINHO_PKL}")

    # Criação e treinamento do modelo Gaussian Naïve Bayes.
    naiveb_risco_credito = GaussianNB()
    naiveb_risco_credito.fit(X_risco_credito, y_risco_credito)

    # Codificação dos dois clientes propostos no exercício.
    novos_clientes = np.array(
        [
            [
                label_encoder_historia.transform(["boa"])[0],
                label_encoder_divida.transform(["alta"])[0],
                label_encoder_garantias.transform(["nenhuma"])[0],
                label_encoder_renda.transform(["acima_35"])[0],
            ],
            [
                label_encoder_historia.transform(["ruim"])[0],
                label_encoder_divida.transform(["alta"])[0],
                label_encoder_garantias.transform(["adequada"])[0],
                label_encoder_renda.transform(["0_15"])[0],
            ],
        ]
    )

    previsoes = naiveb_risco_credito.predict(novos_clientes)
    probabilidades = naiveb_risco_credito.predict_proba(novos_clientes)

    resultado = pd.DataFrame(
        {
            "cliente": ["i", "ii"],
            "previsao": previsoes,
        }
    )
    probabilidades_df = pd.DataFrame(
        probabilidades,
        columns=[
            f"P({classe})" for classe in naiveb_risco_credito.classes_
        ],
        index=["cliente i", "cliente ii"],
    )

    print("\nPREVISÕES")
    print(resultado.to_string(index=False))
    print("\nProbabilidades por classe:")
    print(probabilidades_df.to_string())
    print("\nClasses utilizadas:", naiveb_risco_credito.classes_)
    print("Quantidade por classe:", naiveb_risco_credito.class_count_)


if __name__ == "__main__":
    main()
