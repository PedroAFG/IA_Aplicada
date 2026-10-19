# Documentação detalhada — Parte 2

## 1. Objetivo

A Parte 2 estuda dois algoritmos supervisionados:

- **Árvore de Decisão**, aplicada à base didática de risco e à base Credit Data;
- **Random Forest**, aplicada à base Credit Data e comparada com a árvore simples.

Na base Credit Data, a classe `0` representa o cliente que paga a dívida, e a classe `1` representa o cliente que não paga.

## 2. Importação dos dados pré-processados

```python
with open('../Parte-1/risco_credito.pkl', 'rb') as arquivo:
    X_risco_credito, y_risco_credito = pickle.load(arquivo)
```

O arquivo da Parte 1 contém `X`, com os quatro atributos já codificados numericamente, e `y`, com as classes de risco. No código final, o caminho é construído com `pathlib` para funcionar a partir da nova organização de pastas.

O arquivo `credit.pkl` contém quatro arrays:

```python
X_credit_treinamento, y_credit_treinamento,
X_credit_teste, y_credit_teste
```

Dimensões:

| Conjunto | Dimensão |
|---|---:|
| X treinamento | `(1500, 3)` |
| y treinamento | `(1500,)` |
| X teste | `(500, 3)` |
| y teste | `(500,)` |

Os três atributos da base maior são `income`, `age` e `loan`.

## 3. Árvore de Decisão

Uma Árvore de Decisão cria regras sequenciais. Cada nó contém uma pergunta sobre um atributo, cada ramo representa uma resposta e cada folha produz uma classe.

```python
arvore_credit = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0
)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)
```

- `criterion='entropy'`: escolhe divisões que mais reduzem a incerteza;
- `random_state=0`: torna o resultado reproduzível;
- `fit`: aprende as regras a partir do conjunto de treinamento.

### Entropia e ganho de informação

A entropia mede a mistura das classes em um nó. Um nó puro, contendo somente uma classe, possui entropia zero. A árvore procura a divisão com maior **ganho de informação**, isto é, a maior redução de entropia.

Na base didática, as importâncias encontradas foram:

| Atributo | Importância aproximada |
|---|---:|
| `renda` | 0,4831 |
| `historia` | 0,3362 |
| `divida` | 0,1418 |
| `garantias` | 0,0389 |

`renda` foi o atributo com maior ganho de informação. As importâncias somam 1 e indicam quanto cada atributo participou da redução de impureza ao longo da árvore.

## 4. Previsões na base didática

Os dois clientes são representados pelos mesmos códigos produzidos na Parte 1:

```python
novos_clientes = np.array([
    [0, 0, 1, 2],
    [2, 0, 0, 0]
])
```

Resultados da Árvore de Decisão:

| Cliente | Previsão |
|---|---|
| i | risco `baixo` |
| ii | risco `alto` |

O segundo resultado é diferente daquele produzido pelo Naïve Bayes na Parte 1. Isso é possível porque os algoritmos aprendem padrões de maneiras diferentes: a árvore cria regras de corte, enquanto o Naïve Bayes compara probabilidades.

## 5. Previsões na base Credit Data

```python
previsoes_arvore = arvore_credit.predict(X_credit_teste)
```

O método `predict` aplica as regras aprendidas aos 500 registros de teste. Esses registros não foram utilizados no treinamento, o que permite avaliar a capacidade do modelo de generalizar.

## 6. Acurácia

```python
acuracia = accuracy_score(y_credit_teste, previsoes_arvore)
```

A acurácia é a proporção de previsões corretas:

```text
acurácia = número de acertos / número total de exemplos
```

A Árvore de Decisão obteve **98,2%**, equivalente a 491 acertos em 500 exemplos.

## 7. Matriz de confusão

A matriz da Árvore de Decisão foi:

```text
[[430,  6],
 [  3, 61]]
```

| Posição | Interpretação | Quantidade |
|---|---|---:|
| Verdadeiro negativo | Pagou e foi classificado como pagante | 430 |
| Falso positivo | Pagou, mas foi classificado como não pagante | 6 |
| Falso negativo | Não pagou, mas foi classificado como pagante | 3 |
| Verdadeiro positivo | Não pagou e foi classificado como não pagante | 61 |

Neste problema, o falso negativo merece atenção especial: o banco poderia conceder crédito a alguém que o modelo classificou como pagante, mas que na realidade não paga.

## 8. Precision, recall e F1-score

### Precision

Responde: entre todos os exemplos previstos em determinada classe, quantos estavam corretos?

```text
precision = VP / (VP + FP)
```

### Recall

Responde: entre todos os exemplos que realmente pertenciam à classe, quantos foram encontrados?

```text
recall = VP / (VP + FN)
```

### F1-score

É a média harmônica entre precision e recall. É útil quando queremos equilíbrio entre os dois.

Na classe 1 da Árvore de Decisão:

- precision: **91,04%**;
- recall: **95,31%**;
- F1-score: **93,13%**.

O recall alto mostra que o modelo identificou 61 dos 64 clientes que não pagam.

## 9. Visualização da árvore

```python
tree.plot_tree(
    arvore_credit,
    feature_names=['income', 'age', 'loan'],
    class_names=['Paga', 'Não paga'],
    filled=True,
    rounded=True
)
```

- `feature_names`: mostra os nomes dos atributos;
- `class_names`: mostra os nomes das classes;
- `filled=True`: colore os nós conforme a classe predominante;
- `rounded=True`: melhora a apresentação visual.

O projeto salva as imagens `arvore_risco_credito.png` e `arvore_credit.png`.

## 10. Random Forest

A Random Forest combina várias árvores de decisão. Cada árvore utiliza uma amostra dos registros e dos atributos. A previsão final é determinada pela votação das árvores.

```python
random_forest_credit = RandomForestClassifier(
    n_estimators=10,
    criterion='entropy',
    random_state=0
)
```

- `n_estimators=10`: cria dez árvores;
- `criterion='entropy'`: utiliza entropia nas divisões;
- `random_state=0`: reproduz o mesmo experimento.

A combinação reduz a dependência de uma única árvore e normalmente diminui a variância, mas o resultado depende dos hiperparâmetros e dos dados.

## 11. Resultados da Random Forest com 10 árvores

Acurácia: **96,8%**.

Matriz de confusão:

```text
[[432,  4],
 [ 12, 52]]
```

Para a classe 1:

- precision: **92,86%**;
- recall: **81,25%**;
- F1-score: **86,67%**.

Embora a floresta tenha classificado corretamente mais pagantes, ela deixou de identificar 12 não pagantes. Isso explica a queda do recall da classe 1.

## 12. Aumento do número de árvores

A Random Forest com 10 árvores ficou abaixo da árvore simples. O experimento foi repetido com 40 árvores:

| Modelo | Acurácia |
|---|---:|
| Árvore de Decisão | 98,2% |
| Random Forest — 10 árvores | 96,8% |
| Random Forest — 40 árvores | 98,4% |

O aumento de `n_estimators` produziu uma votação mais estável e superou a árvore simples. Entretanto, não se deve escolher o parâmetro somente com base no conjunto de teste. Em um projeto real, a seleção seria feita com validação cruzada.

## 13. Limitações

- Uma acurácia alta não garante que todas as classes tenham bom desempenho.
- A classe de não pagantes é minoritária: 64 dos 500 registros de teste.
- Falsos negativos e falsos positivos possuem custos diferentes para o banco e para os clientes.
- Árvores sem limitação de profundidade podem sofrer sobreajuste.
- Decisões reais de crédito exigem análise de vieses, explicabilidade, privacidade e supervisão humana.
- Arquivos Pickle só devem ser carregados quando sua origem for confiável.

## 14. Conclusão

A árvore simples teve excelente desempenho e identificou 61 dos 64 não pagantes. A floresta com apenas 10 árvores foi inferior, principalmente no recall da classe 1. Com 40 árvores, a Random Forest atingiu a melhor acurácia geral. A comparação mostra por que métricas, tipos de erro e hiperparâmetros devem ser analisados em conjunto.
