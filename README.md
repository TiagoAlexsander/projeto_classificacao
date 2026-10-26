# Classificação de tumores de mama — Naive Bayes × Árvore de Decisão × Random Forest

Trabalho prático de Ciência de Dados: classificação supervisionada em Python comparando três algoritmos,
cada um em versão **baseline** (parâmetros padrão) e **ajustada** (GridSearchCV), sob o mesmo protocolo.

## Problema e base

- **Problema:** prever se um tumor de mama é maligno (1) ou benigno (0) a partir de 30 medidas de imagens de PAAF.
- **Base:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository
  ([link oficial](https://archive.ics.uci.edu/dataset/17), DOI 10.24432/C5DW2B).
  Também vem no scikit-learn (`sklearn.datasets.load_breast_cancer`), por isso o notebook não precisa de download.
  Uma cópia é salva em `dados/breast_cancer_wisconsin.csv` na primeira execução.

## Estrutura

```
projeto_classificacao/
├── dados/                     # cópia da base (CSV gerado pelo notebook)
├── notebooks/
│   └── trabalho.ipynb         # fluxo completo, do início ao fim (já executado)
├── src/
│   └── avaliacao.py           # funções auxiliares: pipeline, validação cruzada, métricas
├── resultados/
│   ├── figuras/               # gráficos (EDA, matrizes de confusão, ROC, árvore, importâncias)
│   ├── tabela_final_seis_modelos.csv
│   ├── cv_resumo.csv  metricas_teste.csv  wilcoxon.csv
├── relatorio/                 # relatório (relatorio.docx e relatorio.pdf)
├── slides/                    # apresentação da defesa (apresentacao.pptx)
├── requirements.txt
└── README.md
```

## Como executar

```bash
pip install -r requirements.txt
```

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace trabalho.ipynb
```

Ou abra `notebooks/trabalho.ipynb` no Jupyter/VS Code e use *Run All*. No Google Colab, envie também
`src/avaliacao.py` para a pasta `src/` ao lado do notebook. A execução completa leva cerca de 4–5 minutos
(a busca do Random Forest é a etapa mais longa).

## Protocolo (resumo)

- Divisão treino/teste 70/30 estratificada, `random_state=42`.
- Todo pré-processamento dentro de `Pipeline` (sem vazamento de dados).
- `StratifiedKFold(k=10)` no treino; o mesmo objeto de CV para os seis modelos.
- `GridSearchCV` com F1 da classe maligno; teste usado uma única vez no final.
- Extras: tempo de treino, `plot_tree`, importância (impureza e permutação), curvas ROC juntas, teste de Wilcoxon.

## Resultado principal

| Modelo | F1 CV (média ± dp) | F1 teste | Revocação teste | FN |
|---|---|---|---|---|
| NB baseline | 0,916 ± 0,074 | 0,908 | 0,844 | 10 |
| NB ajustado | 0,917 ± 0,079 | 0,926 | 0,875 | 8 |
| AD baseline | 0,885 ± 0,063 | 0,864 | 0,844 | 10 |
| AD ajustado | 0,911 ± 0,071 | 0,908 | 0,844 | 10 |
| RF baseline | 0,939 ± 0,064 | 0,951 | 0,906 | 6 |
| **RF ajustado** | **0,949 ± 0,050** | **0,968** | **0,938** | **4** |

O Random Forest ajustado foi o melhor. Os ganhos do ajuste ficaram dentro da variação da validação cruzada
(Wilcoxon, α = 0,05). A tabela completa está em `resultados/tabela_final_seis_modelos.csv` e no Apêndice A do relatório.
