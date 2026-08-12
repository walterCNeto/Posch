"""Risco de carteira pela abordagem de valor do ativo — capítulo 7.

Os capítulos anteriores produziram parâmetros por devedor: PD, LGD, exposição, e
a correlação que os liga. Este capítulo os agrega em uma **distribuição de
perda** — e é dessa distribuição, não da perda média, que saem capital econômico
e capital regulatório.

Três instrumentos, em ordem de generalidade decrescente e precisão crescente:

* :func:`simular_perdas` — Monte Carlo, que aceita qualquer carteira;
* :func:`quantil_vasicek` — fórmula fechada, exata só no limite de carteira
  homogênea e infinitamente granular, que é a hipótese por trás do IRB;
* :func:`simular_com_is` — Monte Carlo com amostragem por importância, que
  entrega a mesma resposta do primeiro com uma fração das simulações.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ResultadoSimulacao:
    """Distribuição de perda simulada e suas estatísticas de cauda."""

    perdas: np.ndarray
    pesos: np.ndarray | None = None

    @property
    def perda_esperada(self) -> float:
        if self.pesos is None:
            return float(self.perdas.mean())
        return float(np.average(self.perdas, weights=self.pesos))

    def quantil(self, nivel: float) -> float:
        """Percentil da distribuição de perda, respeitando pesos se houver."""
        if self.pesos is None:
            return float(np.quantile(self.perdas, nivel))
        ordem = np.argsort(self.perdas)
        perdas = self.perdas[ordem]
        acumulado = np.cumsum(self.pesos[ordem]) / self.pesos.sum()
        return float(np.interp(nivel, acumulado, perdas))

    def var(self, nivel: float = 0.999) -> float:
        """Perda no percentil ``nivel`` (valor em risco de crédito)."""
        return self.quantil(nivel)

    def capital(self, nivel: float = 0.999) -> float:
        """Perda inesperada: quantil menos a perda esperada.

        É esta a quantidade que o capital cobre. A perda esperada é coberta por
        provisão — confundir as duas leva a dupla contagem ou a buraco.
        """
        return self.var(nivel) - self.perda_esperada

    def expected_shortfall(self, nivel: float = 0.999) -> float:
        """Perda média condicional a ultrapassar o quantil."""
        corte = self.quantil(nivel)
        acima = self.perdas >= corte
        if not acima.any():
            return corte
        if self.pesos is None:
            return float(self.perdas[acima].mean())
        return float(np.average(self.perdas[acima], weights=self.pesos[acima]))


def simular_perdas(
    carteira: pd.DataFrame,
    n_simulacoes: int = 50_000,
    semente: int = 42,
    bloco: int = 5_000,
) -> ResultadoSimulacao:
    """Simula a distribuição de perda por Monte Carlo no modelo de fator único.

    Para cada cenário sorteia-se um fator sistêmico :math:`X`, depois os ruídos
    idiossincráticos de cada devedor; há default onde o valor do ativo cai
    abaixo do limiar, e a perda é ``EAD × LGD`` somada sobre os que quebraram.

    A simulação é feita em blocos para não materializar uma matriz de
    ``n_simulacoes × n_obrigados`` na memória.
    """
    rng = np.random.default_rng(semente)
    ead = carteira["EAD"].to_numpy(float)
    lgd = carteira["LGD"].to_numpy(float)
    rho = carteira["RHO"].to_numpy(float)
    limiar = stats.norm.ppf(carteira["PD"].to_numpy(float))

    raiz_rho = np.sqrt(rho)
    raiz_resto = np.sqrt(1 - rho)
    severidade = ead * lgd

    perdas = np.empty(n_simulacoes)
    feito = 0
    while feito < n_simulacoes:
        m = min(bloco, n_simulacoes - feito)
        X = rng.normal(0.0, 1.0, size=(m, 1))
        eps = rng.normal(0.0, 1.0, size=(m, len(ead)))
        Z = raiz_rho * X + raiz_resto * eps
        perdas[feito : feito + m] = ((Z < limiar) * severidade).sum(axis=1)
        feito += m

    return ResultadoSimulacao(perdas=perdas)


def taxa_condicional_vasicek(
    x: np.ndarray | float, pd_inc: float, rho: float
) -> np.ndarray:
    """Taxa de default condicional ao fator, no modelo de Vasicek."""
    limiar = stats.norm.ppf(pd_inc)
    return stats.norm.cdf((limiar - np.sqrt(rho) * np.asarray(x, float)) / np.sqrt(1 - rho))


def quantil_vasicek(pd_inc: float, rho: float, nivel: float = 0.999) -> float:
    """Taxa de default no percentil ``nivel``, em carteira homogênea granular.

    .. math::
       q(\\alpha) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) + \\sqrt{\\rho}\\,\\Phi^{-1}(\\alpha)}{\\sqrt{1-\\rho}}\\right)

    É a fórmula por trás do requerimento de capital do IRB. Ela vale
    exatamente no limite em que a carteira tem infinitos devedores idênticos —
    situação em que o risco idiossincrático desaparece por diversificação e só
    resta o fator sistêmico.
    """
    limiar = stats.norm.ppf(pd_inc)
    return float(
        stats.norm.cdf(
            (limiar + np.sqrt(rho) * stats.norm.ppf(nivel)) / np.sqrt(1 - rho)
        )
    )


def perda_analitica(
    pd_inc: float, rho: float, lgd: float, ead_total: float, nivel: float = 0.999
) -> dict[str, float]:
    """Perda esperada, perda no percentil e capital pela fórmula fechada."""
    taxa_estresse = quantil_vasicek(pd_inc, rho, nivel)
    esperada = pd_inc * lgd * ead_total
    estresse = taxa_estresse * lgd * ead_total
    return {
        "taxa_estresse": taxa_estresse,
        "perda_esperada": esperada,
        "perda_estresse": estresse,
        "capital": estresse - esperada,
    }


def simular_com_is(
    carteira: pd.DataFrame,
    n_simulacoes: int = 50_000,
    deslocamento: float = -1.5,
    semente: int = 42,
    bloco: int = 5_000,
) -> ResultadoSimulacao:
    """Monte Carlo com amostragem por importância no fator sistêmico.

    O problema do Monte Carlo direto na cauda é aritmético: para estimar o
    percentil 99,9% com precisão são necessários muitos cenários **além** dele,
    e por definição só 0,1% dos sorteios chega lá.

    A correção mais eficaz aqui é deslocar a média do fator sistêmico para o
    território ruim, já que quase toda a massa de perda extrema vem de
    realizações ruins de :math:`X`. Sorteia-se :math:`X \\sim N(\\mu, 1)` com
    :math:`\\mu < 0` e corrige-se cada cenário pela razão de verossimilhanças

    .. math:: w = \\frac{\\phi(x)}{\\phi_\\mu(x)} = \\exp\\!\\left(-\\mu x + \\tfrac{\\mu^2}{2}\\right),

    o que mantém o estimador não-enviesado enquanto concentra o esforço onde a
    perda mora.

    O deslocamento ótimo é menor em magnitude do que a intuição sugere. Para
    estimar o **quantil** 99,9% não basta ter cenários além dele: é preciso
    massa em torno dele. Deslocar demais joga quase tudo para além do ponto de
    interesse e a variância volta a subir. O valor padrão foi calibrado por
    medição direta do erro-padrão — ver :func:`erro_padrao_quantil`.
    """
    rng = np.random.default_rng(semente)
    ead = carteira["EAD"].to_numpy(float)
    lgd = carteira["LGD"].to_numpy(float)
    rho = carteira["RHO"].to_numpy(float)
    limiar = stats.norm.ppf(carteira["PD"].to_numpy(float))

    raiz_rho = np.sqrt(rho)
    raiz_resto = np.sqrt(1 - rho)
    severidade = ead * lgd

    perdas = np.empty(n_simulacoes)
    pesos = np.empty(n_simulacoes)
    feito = 0
    while feito < n_simulacoes:
        m = min(bloco, n_simulacoes - feito)
        X = rng.normal(deslocamento, 1.0, size=(m, 1))
        eps = rng.normal(0.0, 1.0, size=(m, len(ead)))
        Z = raiz_rho * X + raiz_resto * eps
        perdas[feito : feito + m] = ((Z < limiar) * severidade).sum(axis=1)
        pesos[feito : feito + m] = np.exp(
            -deslocamento * X.ravel() + 0.5 * deslocamento**2
        )
        feito += m

    return ResultadoSimulacao(perdas=perdas, pesos=pesos)


def erro_padrao_quantil(
    carteira: pd.DataFrame,
    n_simulacoes: int,
    nivel: float,
    n_repeticoes: int = 20,
    com_is: bool = False,
    deslocamento: float = -1.5,
    semente_base: int = 1000,
) -> dict[str, float]:
    """Mede a dispersão do quantil estimado repetindo a simulação inteira.

    É a única forma honesta de comparar dois estimadores de cauda: rodar cada
    um várias vezes com sementes diferentes e olhar o desvio-padrão da
    estimativa. Um estimador que devolve valores muito diferentes a cada
    execução não é utilizável, por mais elegante que seja.
    """
    estimativas = []
    for k in range(n_repeticoes):
        if com_is:
            r = simular_com_is(
                carteira, n_simulacoes, deslocamento, semente=semente_base + k
            )
        else:
            r = simular_perdas(carteira, n_simulacoes, semente=semente_base + k)
        estimativas.append(r.quantil(nivel))
    estimativas = np.array(estimativas)
    return {
        "media": float(estimativas.mean()),
        "erro_padrao": float(estimativas.std(ddof=1)),
        "cv": float(estimativas.std(ddof=1) / estimativas.mean()),
    }


def contribuicao_por_posicao(
    carteira: pd.DataFrame,
    n_simulacoes: int = 50_000,
    nivel: float = 0.999,
    semente: int = 42,
    bloco: int = 5_000,
) -> pd.DataFrame:
    """Contribuição de cada posição à perda nos cenários de cauda.

    A soma das contribuições é a perda no nível escolhido — é a decomposição
    que responde "de onde vem o capital", pergunta que a perda esperada por
    posição não responde, porque ignora concentração.
    """
    rng = np.random.default_rng(semente)
    ead = carteira["EAD"].to_numpy(float)
    lgd = carteira["LGD"].to_numpy(float)
    rho = carteira["RHO"].to_numpy(float)
    limiar = stats.norm.ppf(carteira["PD"].to_numpy(float))
    severidade = ead * lgd

    raiz_rho = np.sqrt(rho)
    raiz_resto = np.sqrt(1 - rho)

    perdas = np.empty(n_simulacoes)
    defaults = np.zeros((n_simulacoes, len(ead)), dtype=bool)
    feito = 0
    while feito < n_simulacoes:
        m = min(bloco, n_simulacoes - feito)
        X = rng.normal(0.0, 1.0, size=(m, 1))
        eps = rng.normal(0.0, 1.0, size=(m, len(ead)))
        Z = raiz_rho * X + raiz_resto * eps
        d = Z < limiar
        defaults[feito : feito + m] = d
        perdas[feito : feito + m] = (d * severidade).sum(axis=1)
        feito += m

    corte = np.quantile(perdas, nivel)
    cauda = perdas >= corte
    contrib = (defaults[cauda] * severidade).mean(axis=0)

    return pd.DataFrame(
        {
            "ID": carteira["ID"].to_numpy(),
            "EAD": ead,
            "PD": carteira["PD"].to_numpy(float),
            "perda_esperada": severidade * carteira["PD"].to_numpy(float),
            "contribuicao_cauda": contrib,
        }
    )


def numero_efetivo_posicoes(carteira: pd.DataFrame) -> dict[str, float]:
    """Índice de Herfindahl das exposições e o número efetivo que ele implica.

    O inverso do Herfindahl responde: **a quantas posições de tamanho igual
    esta carteira equivale?** Uma carteira de quinhentas posições com exposições
    muito desiguais pode se comportar como algumas dezenas — e é o número
    efetivo, não a contagem, que governa quanto risco idiossincrático sobrevive
    à diversificação.

    É a medida que torna operacional a distância entre a carteira real e a
    hipótese de granularidade infinita por trás da fórmula do IRB.
    """
    ead = carteira["EAD"].to_numpy(float)
    hhi = float((ead**2).sum() / ead.sum() ** 2)
    return {
        "posicoes": int(len(ead)),
        "herfindahl": hhi,
        "numero_efetivo": float(1.0 / hhi),
        "participacao_top1pct": float(
            np.sort(ead)[::-1][: max(1, len(ead) // 100)].sum() / ead.sum()
        ),
    }
