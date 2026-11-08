# Trabalho 3 — Regressão Logística e SVM

O notebook original desta pasta reúne a **Parte 3 (Regressão Logística)** e a **Parte 4 (SVM)**. Os dois blocos foram resolvidos em [TRABALHO_3_PARTE_3_Regressao_Logistica_SVM_RESOLVIDO.ipynb](TRABALHO_3_PARTE_3_Regressao_Logistica_SVM_RESOLVIDO.ipynb), com as células executadas, tabelas, matrizes e respostas comentadas. O arquivo original foi preservado.

## Continuidade com as partes 1 e 2

A Parte 1 codifica os quatro atributos da base didática com um `LabelEncoder` independente para cada coluna e salva `risco_credito.pkl`. A Parte 2 reutiliza esse arquivo e avalia árvores e florestas na separação de treino e teste de `credit.pkl`.

A resolução mantém essa codificação e essa divisão. Para a comparação com Naïve Bayes na Credit Data, treina um novo `GaussianNB` nessa base, pois a Parte 1 só o aplicava à base didática de 14 registros.

## Resultados registrados

Na base didática, são removidos os três registros de classe `moderado`, restando 11 exemplos. A Regressão Logística classifica o primeiro cliente como **baixo** e o segundo como **alto**.

O intercepto calculado foi `-0.80855152`, e os coeficientes foram `[-0.76706797, 0.23921064, -0.47989768, 1.12196145]`. A maior diferença absoluta em relação às referências do enunciado ficou abaixo de `0.0003`. O notebook exibe os valores efetivos e a comparação, sem substituir os resultados pelos números esperados.

Na Credit Data, precisão e recall abaixo se referem a **não paga (1)**:

| Modelo | Acurácia | Precisão (1) | Recall (1) | Falsos negativos |
|---|---:|---:|---:|---:|
| Naïve Bayes | 93,8% | 83,67% | 64,06% | 23 |
| Random Forest — 10 árvores | 96,8% | 92,86% | 81,25% | 12 |
| Random Forest — 40 árvores | 98,4% | 95,16% | 92,19% | 5 |
| Regressão Logística | 94,6% | 79,37% | 78,12% | 14 |
| SVM manual / GridSearch | 99,0% | 98,36% | 93,75% | 4 |

A logística supera Bayes em acurácia e recall, mas tem precisão menor. Fica abaixo das duas florestas. O SVM ajustado apresenta os melhores valores dessas três métricas nesta comparação; sua vantagem de acurácia sobre a floresta de 40 árvores corresponde a três casos.

Com `C=1`, os kernels linear, polinomial, sigmoide e RBF obtiveram, respectivamente, **94,6%, 96,8%, 83,8% e 98,2%** de acurácia de teste. O RBF também foi escolhido pela validação cruzada no treino. Com `C=10`, atingiu 98,8%; com `C=100`, 99,0%.

O GridSearch avaliou **52 combinações**, usando cinco dobras estratificadas. A melhor configuração foi `kernel='rbf', C=100, gamma='scale'`, com média de validação de **99,07%**. O ajuste manual chegou à mesma configuração e empatou com a busca em grade em validação e no teste. A matriz final foi `[[435, 1], [4, 60]]`.

## Execução

A partir de `C:\programas\IA_Aplicada`, com as dependências instaladas:

```powershell
.\.venv\Scripts\python.exe -m pip install -r fase3-TrabalhoIA\requirements.txt
.\.venv\Scripts\python.exe fase3-TrabalhoIA\Parte-3\trabalho_3_parte_3_regressao_logistica_svm.py
```

No VS Code ou Jupyter, selecione esse ambiente como kernel e execute todas as células do notebook resolvido. O script também pode ser executado de outro diretório, pois localiza as bases a partir de sua própria pasta.

Os arquivos `comparacao_modelos_credit.csv` e `matrizes_confusao_comparacao.png` são gerados novamente na execução. As marcações `# %%` no script permitem usar células no VS Code.

Ambiente da execução registrada: Python 3.13.6, NumPy 2.5.3, pandas 3.1.1 e scikit-learn 1.9.1. O Pickle fornecido contém um `StringArray` que requer pandas 3; o requisito foi ajustado em `../requirements.txt`.

## Limites da análise

Os hiperparâmetros são escolhidos apenas com validação no treino. Os resultados de teste dos vários experimentos são apresentados para cumprir o enunciado; como essa base já foi consultada em outras etapas do trabalho, as comparações são didáticas. Pequenas diferenças não estabelecem superioridade universal ou significância estatística.

Os atributos da Credit Data já estão padronizados, mas o código que criou o arquivo não está disponível. Assim, não é possível auditar se o pré-processamento original foi ajustado exclusivamente no treino. Uma reprodução a partir de dados brutos deveria incluir o scaler dentro da validação cruzada.

Referências oficiais: [Regressão Logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) e [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
