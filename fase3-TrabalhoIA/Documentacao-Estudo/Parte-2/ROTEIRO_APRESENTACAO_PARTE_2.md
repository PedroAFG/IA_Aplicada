# Roteiro de apresentação — Parte 2

## Fala sugerida

### Introdução

> Nesta segunda parte aplicamos Árvore de Decisão e Random Forest. Primeiro usamos a base didática de risco de crédito e depois uma base maior, já dividida em treinamento e teste.

### Árvore na base didática

> Carregamos os dados pré-processados da Parte 1 e treinamos uma árvore usando entropia. A entropia mede a mistura das classes, e a árvore procura divisões com maior ganho de informação. O atributo mais importante foi renda, com importância aproximada de 48,3%.

> Para os dois clientes pedidos, a árvore previu risco baixo para o primeiro e risco alto para o segundo. O segundo resultado difere do Naïve Bayes porque os modelos aprendem por mecanismos diferentes.

### Base maior

> A Credit Data possui 1.500 registros de treinamento e 500 de teste, com os atributos renda, idade e empréstimo. A classe zero representa quem paga e a classe um representa quem não paga.

### Resultado da árvore

> A Árvore de Decisão atingiu 98,2% de acurácia. Na matriz de confusão, 430 pagantes e 61 não pagantes foram classificados corretamente. Houve seis pagantes classificados como não pagantes e três não pagantes classificados como pagantes.

### Precision e recall

> Precision mede a qualidade das previsões de uma classe. Recall mede quantos casos reais da classe foram encontrados. Para os não pagantes, a árvore teve aproximadamente 91% de precision e 95% de recall.

### Random Forest

> A Random Forest combina várias árvores e decide por votação. Com as dez árvores solicitadas, a acurácia foi 96,8%, abaixo da árvore simples. O principal problema foi o recall dos não pagantes, que caiu para 81,25%.

> Como melhoria, aumentamos a floresta para 40 árvores. A acurácia subiu para 98,4%, superando a árvore simples. Isso mostra que o número de estimadores influencia a estabilidade do modelo.

### Conclusão

> A acurácia não deve ser analisada sozinha. Em crédito, classificar um não pagante como pagante pode gerar prejuízo. Por isso, observamos também a matriz de confusão, precision e recall. Em um sistema real, ainda seriam necessários validação cruzada, controle de sobreajuste e análise de vieses.

## Resultados para memorizar

| Resultado | Valor |
|---|---:|
| Atributo mais importante na base didática | `renda` |
| Árvore de Decisão | 98,2% |
| Random Forest com 10 árvores | 96,8% |
| Random Forest com 40 árvores | 98,4% |
| Não pagantes corretos — árvore | 61 |
| Não pagantes perdidos — árvore | 3 |
| Não pagantes corretos — floresta com 10 | 52 |
| Não pagantes perdidos — floresta com 10 | 12 |

## Perguntas prováveis

### O que é entropia?

É uma medida de impureza. Quanto mais misturadas estiverem as classes, maior a entropia. Uma folha totalmente pura possui entropia zero.

### O que é ganho de informação?

É a redução de entropia causada por uma divisão. A árvore escolhe as perguntas que produzem os maiores ganhos.

### Por que usar `random_state=0`?

Para que as partes aleatórias do algoritmo produzam o mesmo resultado em novas execuções, facilitando a reprodução do experimento.

### Por que separar treinamento e teste?

O treinamento serve para aprender o modelo. O teste simula dados novos e permite avaliar generalização.

### Por que a Random Forest com 10 árvores foi pior?

Dez árvores podem ser insuficientes para estabilizar a votação. Neste experimento, a floresta deixou de identificar mais clientes da classe minoritária.

### Mais árvores sempre melhoram o modelo?

Não necessariamente. Normalmente estabilizam a floresta, mas aumentam o custo computacional e podem chegar a um ponto sem melhoria. O parâmetro deve ser validado experimentalmente.

### Qual erro é mais preocupante neste problema?

O falso negativo da classe 1: prever que um cliente paga quando ele realmente não paga. O custo exato depende das regras de negócio.

### Qual a diferença entre Árvore e Random Forest?

A Árvore é um único conjunto de regras e é mais fácil de visualizar. A Random Forest combina muitas árvores, normalmente é mais estável, mas perde parte da interpretabilidade.
