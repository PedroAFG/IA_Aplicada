# Trabalho 3 — Inteligência Artificial

Este repositório contém a implementação completa do Trabalho 3 de Inteligência Artificial, dividido em duas partes:

1. classificação de risco de crédito com **Gaussian Naïve Bayes**;
2. classificação com **Árvores de Decisão** e **Random Forest**.

Além dos notebooks resolvidos e executados, o projeto inclui versões completas em Python, bases pré-processadas, visualizações dos modelos e documentação de estudo para apresentação.

## Estrutura do repositório

```text
fase3-TrabalhoIA/
├── Parte-1/
│   ├── TRABALHO_3_PARTE_1_Algoritmo_Naive_Bayes_RESOLVIDO.ipynb
│   ├── trabalho_3_naive_bayes.py
│   ├── risco_credito.csv
│   └── risco_credito.pkl
├── Parte-2/
│   ├── TRABALHO_3_PARTE_2_Arvores_de_Decisao_RESOLVIDO.ipynb
│   ├── trabalho_3_parte_2_arvores_random_forest.py
│   ├── credit.pkl
│   ├── arvore_risco_credito.png
│   ├── arvore_credit.png
│   ├── matriz_confusao_arvore_credit.png
│   └── matriz_confusao_random_forest.png
├── requirements.txt
└── Documentacao-Estudo/
    ├── README.md
    ├── Parte-1/
    │   ├── DOCUMENTACAO_CODIGO.md
    │   └── ROTEIRO_APRESENTACAO.md
    └── Parte-2/
        ├── DOCUMENTACAO_PARTE_2.md
        └── ROTEIRO_APRESENTACAO_PARTE_2.md
```

## Parte 1 — Gaussian Naïve Bayes

### Objetivo

A primeira parte classifica o risco de crédito de clientes nas classes:

- `alto`;
- `baixo`;
- `moderado`.

A base `risco_credito.csv` possui 14 registros, quatro atributos de entrada e uma classe de saída.

| Coluna | Descrição |
|---|---|
| `historia` | Histórico de crédito do cliente |
| `divida` | Nível da dívida |
| `garantias` | Existência de garantias |
| `renda` | Faixa de renda |
| `risco` | Classe de risco que será prevista |

### O que foi implementado

#### 1. Leitura e inspeção dos dados

A base CSV é carregada com o pandas. O código exibe:

- todos os registros;
- dimensão da base;
- quantidade de valores ausentes por coluna;
- distribuição das classes de risco.

A base possui 14 linhas, 5 colunas e nenhum valor ausente. A distribuição é de 6 registros de risco alto, 5 de risco baixo e 3 de risco moderado.

#### 2. Separação entre atributos e classe

Os dados são divididos em:

- `X_risco_credito`: as quatro características utilizadas pelo modelo;
- `y_risco_credito`: a classe de risco conhecida.

As dimensões obtidas são `(14, 4)` para `X` e `(14,)` para `y`.

#### 3. Conversão das categorias

Como os atributos são textuais, foi aplicado um `LabelEncoder` independente em cada coluna.

| Atributo | Codificação |
|---|---|
| `historia` | boa = 0, desconhecida = 1, ruim = 2 |
| `divida` | alta = 0, baixa = 1 |
| `garantias` | adequada = 0, nenhuma = 1 |
| `renda` | 0_15 = 0, 15_35 = 1, acima_35 = 2 |

Os codificadores são preservados para que novos clientes sejam transformados com os mesmos códigos utilizados no treinamento.

#### 4. Salvamento dos dados pré-processados

Os arrays `X_risco_credito` e `y_risco_credito` são salvos no arquivo `risco_credito.pkl`. Esse arquivo é reutilizado na Parte 2.

#### 5. Treinamento do Gaussian Naïve Bayes

Foi criado um modelo `GaussianNB`, treinado com todos os 14 registros da base didática.

O Naïve Bayes utiliza o Teorema de Bayes e assume independência condicional entre os atributos. A versão gaussiana modela os valores por distribuições normais dentro de cada classe.

#### 6. Previsão de dois clientes

Foram avaliados os casos solicitados no exercício:

| Cliente | Características | Resultado |
|---|---|---|
| i | história boa, dívida alta, nenhuma garantia, renda acima de 35 | risco `baixo` |
| ii | história ruim, dívida alta, garantia adequada, renda de 0 a 15 | risco `moderado` |

O notebook também mostra as probabilidades estimadas para cada classe, as classes conhecidas pelo algoritmo e a quantidade de registros por classe.

### Arquivos da Parte 1

- `Parte-1/TRABALHO_3_PARTE_1_Algoritmo_Naive_Bayes_RESOLVIDO.ipynb`: notebook completo e executado;
- `Parte-1/trabalho_3_naive_bayes.py`: versão autônoma do código Python;
- `Parte-1/risco_credito.csv`: base de dados original;
- `Parte-1/risco_credito.pkl`: base pré-processada gerada pelo código.

## Parte 2 — Árvores de Decisão e Random Forest

### Objetivo

A segunda parte aplica:

1. uma Árvore de Decisão à base didática de risco de crédito;
2. uma Árvore de Decisão à base maior Credit Data;
3. uma Random Forest à base Credit Data;
4. métricas e comparações entre os modelos.

Na Credit Data, a classe `0` representa clientes que pagam a dívida e a classe `1` representa clientes que não pagam.

### O que foi implementado

#### 1. Carregamento dos dados pré-processados

A Parte 2 carrega:

- `risco_credito.pkl`, criado na Parte 1;
- `credit.pkl`, que contém conjuntos separados de treinamento e teste.

Dimensões da Credit Data:

| Conjunto | Dimensão |
|---|---:|
| X de treinamento | `(1500, 3)` |
| y de treinamento | `(1500,)` |
| X de teste | `(500, 3)` |
| y de teste | `(500,)` |

Os atributos da base são `income`, `age` e `loan`.

#### 2. Árvore de Decisão na base didática

Foi criado um `DecisionTreeClassifier` com:

```python
criterion="entropy"
random_state=0
```

A entropia mede a impureza de cada nó. O algoritmo escolhe as divisões que produzem maior ganho de informação.

As importâncias obtidas foram:

| Atributo | Importância aproximada |
|---|---:|
| `renda` | 48,31% |
| `historia` | 33,62% |
| `divida` | 14,18% |
| `garantias` | 3,89% |

Portanto, `renda` foi o atributo com maior ganho de informação.

A árvore classificou o cliente i como risco `baixo` e o cliente ii como risco `alto`. O resultado do segundo cliente difere do Naïve Bayes porque os dois modelos aprendem padrões de maneiras diferentes.

Também foi criada a visualização `arvore_risco_credito.png`, com nomes dos atributos, classes e cores.

#### 3. Árvore de Decisão na Credit Data

A árvore foi treinada com os 1.500 registros de treinamento e avaliada nos 500 registros de teste.

Resultado:

```text
Acurácia: 98,2%
```

Matriz de confusão:

```text
[[430,  6],
 [  3, 61]]
```

Interpretação:

- 430 clientes pagantes foram classificados corretamente;
- 6 clientes pagantes foram classificados incorretamente como não pagantes;
- 61 clientes não pagantes foram classificados corretamente;
- 3 clientes não pagantes foram classificados incorretamente como pagantes.

Para a classe dos não pagantes, o modelo obteve aproximadamente:

- precision: 91,04%;
- recall: 95,31%;
- F1-score: 93,13%.

A visualização completa foi salva como `arvore_credit.png`, e a matriz foi salva como `matriz_confusao_arvore_credit.png`.

#### 4. Random Forest com 10 árvores

Foi criado um `RandomForestClassifier` com os parâmetros solicitados:

```python
n_estimators=10
criterion="entropy"
random_state=0
```

A Random Forest combina a votação de várias árvores para produzir a classificação final.

Resultado:

```text
Acurácia: 96,8%
```

Matriz de confusão:

```text
[[432,  4],
 [ 12, 52]]
```

Interpretação:

- 432 pagantes foram classificados corretamente;
- 4 pagantes foram classificados como não pagantes;
- 52 não pagantes foram classificados corretamente;
- 12 não pagantes foram classificados como pagantes.

O recall dos não pagantes foi de 81,25%. Portanto, com apenas 10 árvores, a floresta deixou de identificar mais clientes da classe 1 do que a árvore simples.

A matriz foi salva como `matriz_confusao_random_forest.png`.

#### 5. Melhoria da Random Forest

Como o resultado com 10 árvores foi inferior, o experimento foi repetido com `n_estimators=40`.

| Modelo | Acurácia |
|---|---:|
| Árvore de Decisão | 98,2% |
| Random Forest — 10 árvores | 96,8% |
| Random Forest — 40 árvores | 98,4% |

Com 40 árvores, a votação ficou mais estável e superou a acurácia da árvore simples. Em um projeto real, o número de árvores e os demais hiperparâmetros deveriam ser escolhidos por validação cruzada.

#### 6. Análise das métricas

O projeto calcula e explica:

- **acurácia**: proporção total de acertos;
- **precision**: entre as previsões de uma classe, quantas estavam corretas;
- **recall**: entre os exemplos reais de uma classe, quantos foram encontrados;
- **F1-score**: equilíbrio entre precision e recall;
- **matriz de confusão**: quantidade de acertos e erros de cada tipo.

Em análise de crédito, o falso negativo da classe 1 é especialmente importante: representa um cliente previsto como pagante que, na realidade, não paga.

### Arquivos da Parte 2

- `Parte-2/TRABALHO_3_PARTE_2_Arvores_de_Decisao_RESOLVIDO.ipynb`: notebook completo e executado;
- `Parte-2/trabalho_3_parte_2_arvores_random_forest.py`: versão autônoma do código;
- `Parte-2/credit.pkl`: conjuntos de treinamento e teste;
- `Parte-2/arvore_risco_credito.png`: árvore da base didática;
- `Parte-2/arvore_credit.png`: árvore da Credit Data;
- `Parte-2/matriz_confusao_arvore_credit.png`: matriz da árvore;
- `Parte-2/matriz_confusao_random_forest.png`: matriz da Random Forest.

## Instalação

Recomenda-se criar um ambiente virtual dentro da pasta do projeto.

### Windows — PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No Visual Studio Code, selecione o interpretador `.venv\Scripts\python.exe` por meio do comando **Python: Select Interpreter**.

## Execução

### Parte 1

```powershell
.\.venv\Scripts\python.exe Parte-1\trabalho_3_naive_bayes.py
```

### Parte 2

```powershell
.\.venv\Scripts\python.exe Parte-2\trabalho_3_parte_2_arvores_random_forest.py
```

Também é possível abrir os arquivos `.ipynb` no Jupyter Notebook, Google Colab ou Visual Studio Code.

## Documentação de estudo

A pasta `Documentacao-Estudo` contém:

- explicações detalhadas dos códigos;
- conceitos utilizados pelos algoritmos;
- interpretação das métricas;
- limitações metodológicas;
- roteiros de apresentação;
- respostas para perguntas que podem ser feitas durante a avaliação.

## Observações metodológicas

- A base da Parte 1 possui somente 14 registros e tem finalidade didática.
- A Parte 1 não utiliza conjunto de teste, portanto não calcula desempenho de generalização.
- A Credit Data já está separada em treinamento e teste.
- A acurácia não deve ser analisada isoladamente, especialmente quando as classes são desbalanceadas.
- Aplicações reais de crédito exigem dados representativos, análise de vieses, explicabilidade, privacidade e supervisão humana.
- Arquivos Pickle devem ser carregados somente quando sua origem for confiável.
