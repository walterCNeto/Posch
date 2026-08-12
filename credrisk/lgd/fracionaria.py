"""Perda dada o default (LGD) — capítulo 5.

A LGD é uma fração limitada em [0,1] com massa concentrada nos extremos: muita
dívida recupera quase tudo, muita não recupera quase nada, e o meio é rarefeito.
Regressão linear ignora as duas coisas — o limite e o formato — e o preço disso
é previsão fora do intervalo admissível.

A alternativa padrão é a regressão fracionária de Papke e Wooldridge: um modelo
linear generalizado com link logit e verossimilhança binomial, estimado por
quase-máxima verossimilhança. Ele não exige que a variável seja binária nem que
a distribuição esteja correta — apenas que a média condicional esteja.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from credrisk.data.generators import SENIORIDADES


def matriz_lgd(
    dados: pd.DataFrame,
    numericas: list[str] | None = None,
    referencia: str = "Sr. Sec.",
) -> pd.DataFrame:
    """Monta a matriz de covariáveis com dummies de senioridade.

    A senioridade entra como indicadoras, deixando ``referencia`` de fora — os
    coeficientes das demais medem o efeito **frente** a ela.
    """
    numericas = ["LEV", "COB"] if numericas is None else numericas
    X = pd.DataFrame(index=dados.index)
    X["CONST"] = 1.0
    for nivel in SENIORIDADES:
        if nivel != referencia:
            X[nivel] = (dados["Senioridade"] == nivel).astype(float)
    for coluna in numericas:
        X[coluna] = dados[coluna].astype(float)
    return X


def ajustar_ols(dados: pd.DataFrame, numericas: list[str] | None = None):
    """Regressão linear da LGD em nível. Presente para servir de contraexemplo."""
    X = matriz_lgd(dados, numericas)
    return sm.OLS(dados["LGD"].astype(float), X).fit()


def ajustar_fracionaria(dados: pd.DataFrame, numericas: list[str] | None = None):
    """Regressão fracionária: GLM binomial com link logit.

    A média condicional é :math:`E[LGD \\mid x] = \\Lambda(x'\\beta)`, garantida
    dentro de (0,1) por construção. A estimação é por quase-verossimilhança, e
    os erros-padrão robustos são obtidos com ``cov_type="HC0"`` porque a
    variância binomial não descreve a dispersão real de uma variável contínua.
    """
    X = matriz_lgd(dados, numericas)
    modelo = sm.GLM(
        dados["LGD"].astype(float), X, family=sm.families.Binomial()
    )
    return modelo.fit(cov_type="HC0")


def prever_lgd(
    resultado, dados: pd.DataFrame, numericas: list[str] | None = None
) -> np.ndarray:
    """LGD prevista por qualquer um dos dois estimadores."""
    X = matriz_lgd(dados, numericas)
    return np.asarray(resultado.predict(X), dtype=float)


def fracao_fora_do_intervalo(previsto: np.ndarray) -> float:
    """Proporção de previsões fora de [0,1] — zero, num modelo bem especificado."""
    previsto = np.asarray(previsto, dtype=float)
    return float(np.mean((previsto < 0.0) | (previsto > 1.0)))


def lgd_media_por_ano(dados: pd.DataFrame) -> pd.DataFrame:
    """LGD média e taxa de default por ano, para examinar a correlação sistêmica."""
    return (
        dados.groupby("Ano")
        .agg(
            LGD_media=("LGD", "mean"),
            taxa_default=("taxa_default_ano", "first"),
            n=("LGD", "size"),
        )
        .reset_index()
    )


def lgd_downturn(
    dados: pd.DataFrame, quantil: float = 0.80, coluna_ciclo: str = "taxa_default"
) -> dict[str, float]:
    """Compara a LGD média de todos os anos com a dos anos de estresse.

    O arcabouço de capital exige LGD que reflita condições de desaceleração
    econômica quando houver dependência entre frequência de default e
    severidade. Esta função mede essa dependência da forma mais direta: separa
    os anos acima do quantil de taxa de default e compara as médias.

    Returns
    -------
    dict
        ``media_geral``, ``media_downturn``, ``diferenca`` e ``razao``.
    """
    por_ano = lgd_media_por_ano(dados)
    corte = por_ano[coluna_ciclo].quantile(quantil)
    ruins = por_ano[por_ano[coluna_ciclo] >= corte]

    ponderada = np.average(por_ano["LGD_media"], weights=por_ano["n"])
    ponderada_ruim = np.average(ruins["LGD_media"], weights=ruins["n"])

    return {
        "media_geral": float(ponderada),
        "media_downturn": float(ponderada_ruim),
        "diferenca": float(ponderada_ruim - ponderada),
        "razao": float(ponderada_ruim / ponderada),
        "anos_downturn": int(len(ruins)),
    }
