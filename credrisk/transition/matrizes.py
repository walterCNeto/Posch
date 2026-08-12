"""Matrizes de transição de rating — capítulo 3.

Dois estimadores para o mesmo objeto, com propriedades bem diferentes:

* :func:`matriz_coorte` — conta quantas empresas estavam em ``i`` no começo do
  ano e em ``j`` no fim. É o estimador do mercado, simples e transparente.
* :func:`gerador_duracao` — estima a matriz geradora em tempo contínuo usando
  o tempo de exposição em cada estado e todas as migrações observadas, inclusive
  as que acontecem e se desfazem dentro do mesmo ano.

A diferença prática aparece nas transições raras: o estimador de coorte devolve
zero para qualquer transição não observada, e zero é uma afirmação forte demais
para uma probabilidade que é apenas pequena.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from credrisk.data.generators import RATINGS


def rating_na_data(historico: pd.DataFrame, datas: np.ndarray) -> pd.DataFrame:
    """Estado de cada empresa em cada data de corte.

    Devolve um painel largo (empresas nas linhas, datas nas colunas) com o
    índice do estado, ou ``NaN`` se a empresa ainda não entrou na base.
    """
    saida = {}
    for id_, g in historico.groupby("ID", sort=True):
        tempos = g["Data"].to_numpy()
        estados = g["Estado"].to_numpy()
        posicoes = np.searchsorted(tempos, datas, side="right") - 1
        linha = np.where(posicoes >= 0, estados[np.clip(posicoes, 0, None)], np.nan)
        saida[id_] = linha
    return pd.DataFrame.from_dict(saida, orient="index", columns=[f"t{d:g}" for d in datas])


def _contar_transicoes(painel: np.ndarray) -> np.ndarray:
    """Conta transições entre colunas consecutivas de um painel de estados.

    Vetorizado com ``np.add.at``: o laço explícito sobre pares custava caro no
    bootstrap, que repete esta contagem centenas de vezes.
    """
    n = len(RATINGS)
    contagem = np.zeros((n, n))
    for k in range(painel.shape[1] - 1):
        inicio, fim = painel[:, k], painel[:, k + 1]
        valido = ~np.isnan(inicio) & ~np.isnan(fim)
        np.add.at(contagem, (inicio[valido].astype(int), fim[valido].astype(int)), 1)
    return contagem


def matriz_coorte(
    historico: pd.DataFrame, anos: float = 15.0, passo: float = 1.0
) -> pd.DataFrame:
    """Estimador de coorte da matriz de transição anual.

    Para cada par de datas separadas por ``passo``, conta as empresas que estavam
    em cada estado no início e onde estavam no fim, e divide pela contagem
    inicial. Migrações intermediárias que se revertem dentro do intervalo são
    invisíveis para este estimador — ele só enxerga as pontas.
    """
    datas = np.arange(0.0, anos + 1e-9, passo)
    painel = rating_na_data(historico, datas).to_numpy()
    n = len(RATINGS)
    contagem = _contar_transicoes(painel)
    totais = contagem.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(totais > 0, contagem / totais, 0.0)
    # Estado sem nenhuma observação inicial: mantém-se onde está, por convenção.
    for i in range(n):
        if totais[i, 0] == 0:
            P[i, i] = 1.0
    return pd.DataFrame(P, index=RATINGS, columns=RATINGS)


def exposicao_e_transicoes(
    historico: pd.DataFrame, anos: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Tempo total de exposição por estado e contagem de migrações observadas.

    Returns
    -------
    tempo : ndarray, shape (n_estados,)
        Anos-empresa acumulados em cada estado.
    N : ndarray, shape (n_estados, n_estados)
        Número de migrações de ``i`` para ``j``.
    """
    n = len(RATINGS)
    tempo = np.zeros(n)
    N = np.zeros((n, n))

    for _, g in historico.groupby("ID", sort=True):
        tempos = g["Data"].to_numpy()
        estados = g["Estado"].to_numpy()
        for k in range(len(estados)):
            inicio = tempos[k]
            fim = tempos[k + 1] if k + 1 < len(estados) else anos
            if estados[k] == n - 1:  # default é absorvente: não acumula exposição
                break
            tempo[estados[k]] += max(fim - inicio, 0.0)
            if k + 1 < len(estados):
                N[estados[k], estados[k + 1]] += 1
    return tempo, N


def gerador_duracao(historico: pd.DataFrame, anos: float = 15.0) -> pd.DataFrame:
    """Estimador de duração (ou de máxima verossimilhança) da matriz geradora.

    A intensidade estimada é

    .. math:: \\hat q_{ij} = \\frac{N_{ij}}{T_i},

    onde :math:`N_{ij}` conta migrações de ``i`` para ``j`` e :math:`T_i` é o
    tempo total que a carteira passou no estado ``i``. É o estimador de máxima
    verossimilhança da cadeia em tempo contínuo.
    """
    tempo, N = exposicao_e_transicoes(historico, anos)
    n = len(RATINGS)
    Q = np.zeros((n, n))
    for i in range(n):
        if tempo[i] > 0:
            Q[i] = N[i] / tempo[i]
        Q[i, i] = 0.0
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return pd.DataFrame(Q, index=RATINGS, columns=RATINGS)


def matriz_do_gerador(Q: pd.DataFrame | np.ndarray, t: float = 1.0) -> pd.DataFrame:
    """Matriz de transição em horizonte ``t``: :math:`P(t) = \\exp(Qt)`.

    É aqui que o estimador de duração paga: uma vez estimada a geradora, a
    matriz de **qualquer** horizonte sai por exponencial de matriz, sem
    reestimação — e sem exigir que o horizonte seja múltiplo do passo de
    observação.
    """
    valores = np.asarray(Q, dtype=float)
    P = linalg.expm(valores * t)
    # Corrige ruído numérico: probabilidades no intervalo unitário, linhas em 1.
    P = np.clip(P, 0.0, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return pd.DataFrame(P, index=RATINGS, columns=RATINGS)


def pd_por_horizonte(
    Q: pd.DataFrame | np.ndarray, horizontes: list[float]
) -> pd.DataFrame:
    """Probabilidade cumulativa de default por rating e horizonte."""
    linhas = {}
    for t in horizontes:
        P = matriz_do_gerador(Q, t)
        linhas[f"{t:g}a"] = P["D"]
    return pd.DataFrame(linhas)


def bootstrap_coorte(
    historico: pd.DataFrame,
    n_reamostras: int = 200,
    anos: float = 15.0,
    passo: float = 1.0,
    semente: int = 0,
) -> np.ndarray:
    """Reamostra empresas (não linhas) para obter a incerteza da matriz de coorte.

    A unidade de reamostragem é a empresa, porque as observações de uma mesma
    empresa não são independentes — a mesma lição do capítulo 1, agora com
    consequência prática, já que aqui há vários eventos por empresa.

    O painel de estados por data é construído uma única vez; cada reamostra
    apenas sorteia linhas dele. Reamostrar linhas do painel é o mesmo que
    reamostrar empresas, e é duas ordens de grandeza mais rápido que
    reconstruir o histórico a cada passo.

    Returns
    -------
    ndarray, shape (n_reamostras, n_estados, n_estados)
    """
    rng = np.random.default_rng(semente)
    datas = np.arange(0.0, anos + 1e-9, passo)
    painel = rating_na_data(historico, datas).to_numpy()
    n_empresas = painel.shape[0]
    n = len(RATINGS)
    saida = np.empty((n_reamostras, n, n))

    for b in range(n_reamostras):
        linhas = rng.integers(0, n_empresas, size=n_empresas)
        contagem = _contar_transicoes(painel[linhas])
        totais = contagem.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            P = np.where(totais > 0, contagem / totais, 0.0)
        for i in range(n):
            if totais[i, 0] == 0:
                P[i, i] = 1.0
        saida[b] = P
    return saida
