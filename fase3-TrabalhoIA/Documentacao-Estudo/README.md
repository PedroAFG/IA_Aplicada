# Documentação de Estudo — Trabalho 3

Esta pasta reúne o material de apoio para estudar e apresentar o projeto de classificação de risco de crédito com o algoritmo **Gaussian Naïve Bayes**.

## Arquivos

- [Documentação detalhada da Parte 1](Parte-1/DOCUMENTACAO_CODIGO.md): explicação técnica, bloco por bloco, do Naïve Bayes.
- [Roteiro da Parte 1](Parte-1/ROTEIRO_APRESENTACAO.md): sequência sugerida de fala, resultados e possíveis perguntas.
- [Documentação detalhada da Parte 2](Parte-2/DOCUMENTACAO_PARTE_2.md): Árvores de Decisão, matriz de confusão, métricas e Random Forest.
- [Roteiro da Parte 2](Parte-2/ROTEIRO_APRESENTACAO_PARTE_2.md): fala sugerida e respostas para perguntas sobre os novos algoritmos.

## Arquivos principais do projeto

- `Parte-1/`: notebook, script, CSV e dados pré-processados do Naïve Bayes.
- `Parte-2/`: notebook, script, Credit Data e imagens das árvores e matrizes.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o programa a partir da raiz do projeto:

```bash
python trabalho_3_naive_bayes.py
```

O script procura o CSV na mesma pasta em que ele está e gera o arquivo `risco_credito.pkl` nessa pasta.
