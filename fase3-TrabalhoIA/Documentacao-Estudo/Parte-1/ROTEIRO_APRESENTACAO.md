# Roteiro para apresentação

## Versão sugerida — aproximadamente 5 minutos

### 1. Introdução

> Este trabalho aplica o algoritmo Naïve Bayes a uma base didática de risco de crédito. O objetivo é classificar novos clientes nas classes de risco alto, baixo ou moderado a partir de quatro características: história de crédito, dívida, garantias e renda.

### 2. Conhecimento da base

> Primeiro, o código lê o arquivo CSV com o pandas. A base possui 14 registros, quatro atributos previsores e uma coluna de classe. Também verificamos que não existem valores ausentes. As classes têm 6 registros de risco alto, 5 de risco baixo e 3 de risco moderado.

### 3. Separação de X e y

> Em seguida, separamos os dados em X e y. O X contém as quatro características utilizadas na previsão, enquanto o y contém a resposta correta, que é a coluna risco. O X fica com dimensão 14 por 4 e o y possui 14 elementos.

### 4. Pré-processamento

> Como os atributos são textos, utilizamos um Label Encoder diferente para cada coluna. Por exemplo, história boa recebe código zero, desconhecida recebe um e ruim recebe dois. Guardamos os encoders porque os novos clientes precisam ser transformados exatamente com os mesmos mapeamentos.

### 5. Arquivo Pickle

> Depois da transformação, salvamos X e y no arquivo risco_credito.pkl. O Pickle preserva objetos Python em formato binário e permite reutilizar os dados pré-processados sem repetir todas as etapas.

### 6. Treinamento

> Criamos um objeto GaussianNB e chamamos o método fit com X e y. O Naïve Bayes utiliza o Teorema de Bayes e assume que os atributos são condicionalmente independentes para cada classe. A versão gaussiana modela cada atributo por uma distribuição normal dentro de cada classe.

### 7. Previsões

> O primeiro cliente possui história boa, dívida alta, nenhuma garantia e renda acima de 35. O modelo classificou esse cliente como risco baixo, com probabilidade aproximada de 93,4%. O segundo possui história ruim, dívida alta, garantia adequada e renda de até 15. Ele foi classificado como risco moderado.

### 8. Limitações e conclusão

> Esta é uma base muito pequena e didática. Não há conjunto de teste, por isso o projeto demonstra o funcionamento do algoritmo, mas não mede sua capacidade de generalização. Além disso, aplicar Gaussian Naïve Bayes a códigos de categorias é uma simplificação do exercício. Em uma aplicação real, compararíamos outros modelos, usaríamos uma base maior e avaliaríamos métricas em dados separados.

## Resultados para memorizar

| Item | Resultado |
|---|---|
| Tamanho da base | 14 linhas e 5 colunas |
| Atributos de entrada | 4 |
| Classes | `alto`, `baixo`, `moderado` |
| Contagem das classes | 6, 5 e 3 |
| Cliente i | `baixo` |
| Cliente ii | `moderado` |

## Perguntas que podem ser feitas

### Por que os textos precisam ser convertidos em números?

Porque os algoritmos matemáticos do scikit-learn trabalham com matrizes numéricas. O Label Encoder cria uma representação numérica para cada categoria.

### Por que foi criado um encoder para cada coluna?

Cada coluna representa um conceito diferente e possui suas próprias categorias. Encoders separados mantêm os mapeamentos independentes e reutilizáveis.

### Por que y não foi codificado?

O `GaussianNB` aceita diretamente nomes textuais como rótulos da classe. Mantê-los como texto torna a saída mais fácil de interpretar.

### O que o método fit faz?

Ele aprende, a partir da base, a frequência de cada classe e os parâmetros estatísticos dos atributos em cada classe.

### Qual é a diferença entre predict e predict_proba?

`predict` devolve somente a classe escolhida. `predict_proba` mostra a probabilidade estimada para cada classe.

### O que significa Naïve Bayes?

É um classificador baseado no Teorema de Bayes. O termo *naïve* se refere à suposição simplificadora de independência condicional entre os atributos.

### Por que usar a versão GaussianNB?

Ela foi indicada pelo exercício e modela os atributos numericamente por distribuições gaussianas. Para categorias puras, essa escolha é uma simplificação didática; outros modelos poderiam ser comparados em um projeto real.

### Podemos dizer que o modelo tem 100% de acurácia?

Não. Os dois resultados pedidos não formam um teste de desempenho, e não houve separação de uma amostra de teste. Seria necessário avaliar o modelo em dados que não participaram do treinamento.

### Para que serve o arquivo Pickle?

Ele armazena os arrays já pré-processados em formato binário para reutilização posterior no Python. Arquivos Pickle só devem ser abertos quando vêm de uma fonte confiável.

### Quais melhorias seriam possíveis?

Usar mais registros, separar treino e teste, aplicar validação cruzada, medir métricas, comparar algoritmos e verificar vieses e impacto das decisões.
