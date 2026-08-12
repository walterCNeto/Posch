"""Escore de crédito com regressão logística — capítulo 1.

Três estimadores para o mesmo modelo, em ordem crescente de honestidade
estatística:

* :func:`ajustar` — máxima verossimilhança padrão, erros-padrão que supõem
  observações independentes;
* :func:`ajustar` com ``cluster="ID"`` — mesmos coeficientes, erros-padrão que
  reconhecem que observações da mesma empresa não são independentes;
* :func:`ajustar_firth` — verossimilhança penalizada de Firth, que corrige o
  viés de amostra pequena que aparece quando há poucos defaults.

A distinção importa: o primeiro subestima a incerteza, o terceiro corrige o
ponto estimado. Um validador que só olha o coeficiente não vê nem um nem outro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def matriz_de_desenho(
    dados: pd.DataFrame, preditores: list[str], constante: bool = True
) -> pd.DataFrame:
    """Monta a matriz de covariáveis, com coluna de constante nomeada ``CONST``."""
    X = dados.loc[:, preditores].astype(float).copy()
    if constante:
        X.insert(0, "CONST", 1.0)
    return X


def ajustar(
    dados: pd.DataFrame,
    preditores: list[str],
    alvo: str = "Default",
    cluster: str | None = None,
):
    """Ajusta o logit por máxima verossimilhança.

    Parameters
    ----------
    dados
        Painel com o alvo binário e as covariáveis.
    preditores
        Nomes das colunas explicativas.
    alvo
        Nome da coluna binária de default.
    cluster
        Se informado (por exemplo ``"ID"``), calcula erros-padrão robustos
        agrupados por essa coluna. Os coeficientes não mudam; a incerteza sim.

    Returns
    -------
    statsmodels.discrete.discrete_model.BinaryResultsWrapper
    """
    X = matriz_de_desenho(dados, preditores)
    y = dados[alvo].astype(float)
    modelo = sm.Logit(y, X)
    if cluster is None:
        return modelo.fit(disp=0)
    grupos = dados[cluster].to_numpy()
    return modelo.fit(disp=0, cov_type="cluster", cov_kwds={"groups": grupos})


@dataclass
class ResultadoFirth:
    """Saída do logit penalizado de Firth."""

    params: pd.Series
    bse: pd.Series
    n_obs: int
    n_eventos: int
    iteracoes: int
    convergiu: bool
    nomes: list[str] = field(default_factory=list)

    @property
    def z(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2 * stats.norm.sf(np.abs(self.z)), index=self.params.index)

    def resumo(self) -> pd.DataFrame:
        """Tabela de coeficientes no formato usual."""
        inferior = self.params - 1.96 * self.bse
        superior = self.params + 1.96 * self.bse
        return pd.DataFrame(
            {
                "coef": self.params,
                "ep": self.bse,
                "z": self.z,
                "p": self.pvalues,
                "ic_inf": inferior,
                "ic_sup": superior,
            }
        )


def ajustar_firth(
    dados: pd.DataFrame,
    preditores: list[str],
    alvo: str = "Default",
    max_iter: int = 100,
    tolerancia: float = 1e-8,
) -> ResultadoFirth:
    """Ajusta o logit com a penalização de Firth (viés de amostra pequena).

    A máxima verossimilhança do logit é consistente, mas enviesada em amostra
    finita, e o viés cresce quando os eventos são raros. Firth (1993) resolve
    penalizando a verossimilhança pelo *prior* de Jeffreys, o que equivale a
    somar :math:`h_i/2` ao escore, onde :math:`h_i` é a alavancagem da
    observação :math:`i`. O estimador resultante tem viés de ordem menor e
    existe mesmo sob separação completa, situação em que a MLE diverge.

    Implementado por Newton-Raphson sobre o escore modificado.

    Returns
    -------
    ResultadoFirth
    """
    X = matriz_de_desenho(dados, preditores)
    nomes = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    y = dados[alvo].to_numpy(dtype=float)

    beta = np.zeros(Xv.shape[1])
    convergiu = False
    iteracao = 0

    for passo_n in range(1, max_iter + 1):
        iteracao = passo_n
        eta = Xv @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)

        XtWX = Xv.T @ (Xv * w[:, None])
        inversa = np.linalg.pinv(XtWX)

        # Alavancagem: diagonal da matriz-chapéu ponderada.
        raiz_w = np.sqrt(w)
        Xw = Xv * raiz_w[:, None]
        h = np.einsum("ij,jk,ik->i", Xw, inversa, Xw)

        escore = Xv.T @ (y - p + h * (0.5 - p))
        passo = inversa @ escore

        # Amortecimento: evita passo explosivo nas primeiras iterações.
        norma = np.max(np.abs(passo))
        if norma > 5.0:
            passo *= 5.0 / norma

        beta = beta + passo
        if np.max(np.abs(passo)) < tolerancia:
            convergiu = True
            break

    eta = Xv @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1.0 - p)
    cov = np.linalg.pinv(Xv.T @ (Xv * w[:, None]))
    erros = np.sqrt(np.diag(cov))

    return ResultadoFirth(
        params=pd.Series(beta, index=nomes),
        bse=pd.Series(erros, index=nomes),
        n_obs=len(y),
        n_eventos=int(y.sum()),
        iteracoes=iteracao,
        convergiu=convergiu,
        nomes=nomes,
    )


def prever_pd(resultado, dados: pd.DataFrame, preditores: list[str]) -> np.ndarray:
    """Probabilidade de default prevista, para qualquer um dos estimadores."""
    X = matriz_de_desenho(dados, preditores).to_numpy(dtype=float)
    beta = np.asarray(resultado.params, dtype=float)
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


def comparar_com_verdadeiro(
    resultado, verdadeiros: dict[str, float]
) -> pd.DataFrame:
    """Confronta o estimado com o parâmetro plantado no gerador sintético.

    Devolve, por coeficiente, o valor verdadeiro, o estimado, o erro-padrão, o
    desvio em número de erros-padrão e se o intervalo de 95% cobre a verdade.
    Esta tabela é o instrumento central do curso: ela responde "o estimador
    acertou?", pergunta que dado real nunca permite fazer.
    """
    params = pd.Series(resultado.params)
    erros = pd.Series(resultado.bse)
    verdade = pd.Series({nome: verdadeiros[nome] for nome in params.index})

    desvio_ep = (params - verdade) / erros
    tabela = pd.DataFrame(
        {
            "verdadeiro": verdade,
            "estimado": params,
            "ep": erros,
            "desvio_em_ep": desvio_ep,
            "ic95_cobre": (verdade >= params - 1.96 * erros)
            & (verdade <= params + 1.96 * erros),
        }
    )
    return tabela


def auc(escore: np.ndarray, alvo: np.ndarray) -> float:
    """Área sob a curva ROC, pela identidade com a estatística de Mann-Whitney.

    Um escore *alto* deve indicar *maior* risco. Ao usar a PD prevista como
    escore, isso já vale por construção.
    """
    escore = np.asarray(escore, dtype=float)
    alvo = np.asarray(alvo)
    maus = escore[alvo == 1]
    bons = escore[alvo == 0]
    if len(maus) == 0 or len(bons) == 0:
        raise ValueError("É preciso ter ao menos um default e um não-default.")
    postos = stats.rankdata(np.concatenate([maus, bons]))
    soma_maus = postos[: len(maus)].sum()
    return (soma_maus - len(maus) * (len(maus) + 1) / 2) / (len(maus) * len(bons))


def razao_de_acuracia(escore: np.ndarray, alvo: np.ndarray) -> float:
    """Razão de acurácia (AR, ou Gini), pela identidade ``AR = 2 * AUC - 1``."""
    return 2.0 * auc(escore, alvo) - 1.0
