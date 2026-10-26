"""Funções auxiliares de modelagem e avaliação usadas no notebook.

Ficam aqui para que o notebook mostre o raciocínio e não fique poluído
com código repetitivo. Todas as funções recebem a semente e o objeto de
validação cruzada de fora, garantindo que os seis modelos usem
exatamente as mesmas partições.
"""
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

# Métricas calculadas em cada fold da validação cruzada.
# A classe positiva é 1 = maligno.
METRICAS = {
    "acuracia": "accuracy",
    "precisao": "precision",
    "revocacao": "recall",
    "f1": "f1",
    "auc": "roc_auc",
}


def criar_pipeline(classificador, etapas_extras=None):
    """Monta o Pipeline: imputação -> (etapas extras) -> classificador.

    A imputação pela mediana fica no pipeline mesmo sem valores ausentes
    na base atual: assim o fluxo continua correto se novos dados chegarem
    incompletos, e a mediana é sempre calculada apenas com o treino.
    """
    etapas = [("imputador", SimpleImputer(strategy="median"))]
    etapas += etapas_extras or []
    etapas.append(("modelo", classificador))
    return Pipeline(etapas)


def validar(modelo, X, y, cv):
    """Validação cruzada com todas as métricas; devolve os valores por fold."""
    res = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS, n_jobs=-1)
    por_fold = {m: res[f"test_{m}"] for m in METRICAS}
    por_fold["tempo_fit"] = res["fit_time"]
    return por_fold


def resumo_cv(folds_por_modelo):
    """Tabela 'média ± desvio' a partir dos resultados por fold."""
    linhas = {}
    for nome, folds in folds_por_modelo.items():
        linhas[nome] = {
            f"{m}_media": np.mean(v) for m, v in folds.items()
        } | {
            f"{m}_dp": np.std(v, ddof=1) for m, v in folds.items()
        }
    return pd.DataFrame(linhas).T


def treinar_cronometrado(modelo, X, y):
    """Treina o modelo e devolve o tempo de treinamento em segundos."""
    inicio = time.perf_counter()
    modelo.fit(X, y)
    return time.perf_counter() - inicio


def metricas_teste(modelo, X, y):
    """Métricas no conjunto de teste (usado uma única vez)."""
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1]
    return {
        "acuracia": accuracy_score(y, y_pred),
        "precisao": precision_score(y, y_pred),
        "revocacao": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "auc": roc_auc_score(y, y_prob),
    }


def formatar_media_dp(df, metrica, casas=3):
    """Coluna de texto 'média ± dp' para tabelas do relatório."""
    return (df[f"{metrica}_media"].round(casas).map(f"{{:.{casas}f}}".format)
            + " ± "
            + df[f"{metrica}_dp"].round(casas).map(f"{{:.{casas}f}}".format))
