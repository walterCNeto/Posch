"""Previsão de taxas de default agregadas — capítulo 4.

Até aqui a probabilidade de default foi tratada como característica do devedor.
Este módulo trata do outro eixo: a taxa de default do sistema varia com o ciclo,
e prever essa variação é o que separa uma estimativa histórica de uma estimativa
prospectiva.

Duas dificuldades definem o capítulo. A primeira é que a taxa é uma fração
limitada em (0,1), então regressão linear não é o instrumento natural. A segunda
é que a amostra tem uma observação por ano — algumas dezenas no total — o que
torna qualquer avaliação dentro da amostra otimista demais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def logito(p: np.ndarray | float) -> np.ndarray:
    """Transformação logit, com proteção contra 0 e 1 exatos."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logito_inverso(x: np.ndarray | float) -> np.ndarray:
    """Inversa do logit."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def ajustar_taxa(
    dados: pd.DataFrame,
    preditores: list[str],
    alvo: str = "IDR",
    transformar: bool = True,
    hac_lags: int | None = None,
):
    """Regride a taxa de default (ou seu logit) sobre os fatores.

    Parameters
    ----------
    transformar
        Se ``True``, regride ``logit(taxa)`` — o que impede previsão fora de
        (0,1) e faz o efeito dos fatores ser multiplicativo na razão de chances.
        Se ``False``, regride a taxa em nível, para efeito de comparação.
    hac_lags
        Se informado, usa erros-padrão de Newey-West robustos a
        heterocedasticidade e autocorrelação. Séries de taxa de default têm
        resíduo persistente, e ignorá-lo subestima a incerteza.
    """
    X = sm.add_constant(dados.loc[:, preditores].astype(float))
    y = logito(dados[alvo]) if transformar else dados[alvo].astype(float)
    modelo = sm.OLS(y, X)
    if hac_lags is None:
        return modelo.fit()
    return modelo.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def prever_taxa(
    resultado, dados: pd.DataFrame, preditores: list[str], transformar: bool = True
) -> np.ndarray:
    """Taxa prevista, desfazendo a transformação quando houver."""
    X = sm.add_constant(dados.loc[:, preditores].astype(float), has_constant="add")
    linear = np.asarray(resultado.predict(X), dtype=float)
    return logito_inverso(linear) if transformar else linear


@dataclass
class ResultadoBacktest:
    """Saída da avaliação fora da amostra por janela expansível."""

    previsoes: pd.DataFrame
    rmse_modelo: float
    rmse_media: float
    rmse_ingenuo: float

    @property
    def ganho_sobre_media(self) -> float:
        """Redução proporcional do erro frente a prever sempre a média histórica.

        Positivo significa que o modelo ajuda; negativo, que atrapalha.
        """
        return 1.0 - self.rmse_modelo / self.rmse_media

    def resumo(self) -> pd.Series:
        return pd.Series(
            {
                "RMSE do modelo": self.rmse_modelo,
                "RMSE da média histórica": self.rmse_media,
                "RMSE do passeio aleatório": self.rmse_ingenuo,
                "ganho sobre a média": self.ganho_sobre_media,
            }
        )


def backtest_expandindo(
    dados: pd.DataFrame,
    preditores: list[str],
    alvo: str = "IDR",
    minimo_treino: int = 20,
    transformar: bool = True,
) -> ResultadoBacktest:
    """Avalia previsão um passo à frente com janela de treino expansível.

    Em cada ano, o modelo é reestimado usando **apenas** o que estaria
    disponível até ali, e prevê o ano seguinte. É a única avaliação honesta de
    capacidade preditiva: o ajuste dentro da amostra usa informação do futuro
    para estimar o passado.

    Dois comparativos acompanham o modelo, e nenhum dos dois é decorativo:

    * a **média histórica** até o ano corrente — o que uma área de risco faria
      sem modelo nenhum;
    * o **passeio aleatório**, isto é, prever que o ano que vem repete o
      corrente.

    Um modelo que não bate os dois não está pagando sua complexidade.
    """
    dados = dados.sort_values("Ano").reset_index(drop=True)
    linhas = []

    for k in range(minimo_treino, len(dados)):
        treino = dados.iloc[:k]
        teste = dados.iloc[[k]]

        ajuste = ajustar_taxa(treino, preditores, alvo, transformar=transformar)
        previsto = float(prever_taxa(ajuste, teste, preditores, transformar)[0])

        linhas.append(
            {
                "Ano": int(teste["Ano"].iloc[0]),
                "observado": float(teste[alvo].iloc[0]),
                "modelo": previsto,
                "media_historica": float(treino[alvo].mean()),
                "passeio_aleatorio": float(treino[alvo].iloc[-1]),
            }
        )

    previsoes = pd.DataFrame(linhas)

    def rmse(coluna: str) -> float:
        erro = previsoes[coluna] - previsoes["observado"]
        return float(np.sqrt(np.mean(erro**2)))

    return ResultadoBacktest(
        previsoes=previsoes,
        rmse_modelo=rmse("modelo"),
        rmse_media=rmse("media_historica"),
        rmse_ingenuo=rmse("passeio_aleatorio"),
    )


def ruido_binomial_esperado(taxa: np.ndarray, n: np.ndarray) -> float:
    """Desvio-padrão médio da frequência observada em torno da taxa verdadeira.

    Serve de piso para o erro de qualquer previsão: mesmo conhecendo a taxa
    exata, a frequência realizada de um universo finito flutua em torno dela.
    Comparar o erro do modelo com esse piso responde a uma pergunta que o
    :math:`R^2` não responde — quanto do que sobra é ruído irredutível.
    """
    taxa = np.asarray(taxa, dtype=float)
    n = np.asarray(n, dtype=float)
    return float(np.mean(np.sqrt(taxa * (1 - taxa) / n)))
