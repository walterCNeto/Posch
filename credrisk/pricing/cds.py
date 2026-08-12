"""CDS e probabilidades de default neutras ao risco — capítulo 10.

Todos os capítulos anteriores estimaram probabilidades a partir de **frequências
observadas**: quantos devedores quebraram, quantas vezes. Este capítulo extrai
probabilidades de **preços** — do que o mercado cobra hoje para assumir risco de
crédito.

As duas coisas não são iguais, e a diferença não é erro de medição. Um investidor
que assume risco de default exige retorno acima da taxa livre de risco. Essa
exigência aparece, quando se desconta tudo à taxa livre de risco, como uma
probabilidade inflada. É a probabilidade neutra ao risco: não é uma previsão de
frequência, é um preço.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


def desconto(t: np.ndarray | float, taxa: float) -> np.ndarray:
    """Fator de desconto contínuo à taxa livre de risco."""
    return np.exp(-taxa * np.asarray(t, dtype=float))


def sobrevivencia(
    t: np.ndarray | float, hazards: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Probabilidade de sobreviver até ``t`` com intensidade constante por trecho.

    Com intensidade de default :math:`\\lambda` constante em cada intervalo,

    .. math:: S(t) = \\exp\\left(-\\int_0^t \\lambda(u)\\,du\\right),

    e a integral é a soma das intensidades pelos comprimentos dos trechos
    percorridos. Modelar por trechos — em vez de uma intensidade única — é o que
    permite ajustar simultaneamente CDS de vários prazos.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    hazards = np.asarray(hazards, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    bordas = np.concatenate([[0.0], vertices])
    integral = np.zeros_like(t)
    for i, h in enumerate(hazards):
        inicio, fim = bordas[i], bordas[i + 1]
        integral += h * np.clip(t - inicio, 0.0, fim - inicio)
    return np.exp(-integral)


def pd_acumulada(
    t: np.ndarray | float, hazards: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Probabilidade neutra ao risco de default até ``t``."""
    return 1.0 - sobrevivencia(t, hazards, vertices)


def _grade(prazo: float, por_ano: int = 4) -> np.ndarray:
    """Datas de pagamento do prêmio, trimestrais por padrão."""
    return np.arange(1, int(round(prazo * por_ano)) + 1) / por_ano


def perna_premio(
    prazo: float,
    hazards: np.ndarray,
    vertices: np.ndarray,
    taxa: float,
    por_ano: int = 4,
) -> float:
    """Valor presente de um prêmio unitário, incluindo o acruado até o default.

    O comprador do CDS paga o prêmio enquanto o emissor sobrevive. Se o default
    ocorre entre duas datas, paga-se a fração acruada — aproximada aqui por meio
    período, que é a convenção de mercado.
    """
    datas = _grade(prazo, por_ano)
    dt = 1.0 / por_ano
    S = sobrevivencia(datas, hazards, vertices)
    S_anterior = sobrevivencia(datas - dt, hazards, vertices)

    regular = float(np.sum(dt * S * desconto(datas, taxa)))
    acruado = float(np.sum(0.5 * dt * (S_anterior - S) * desconto(datas - dt / 2, taxa)))
    return regular + acruado


def perna_protecao(
    prazo: float,
    hazards: np.ndarray,
    vertices: np.ndarray,
    taxa: float,
    recuperacao: float = 0.40,
    passos_por_ano: int = 12,
) -> float:
    """Valor presente do pagamento de proteção em caso de default.

    O vendedor paga ``1 - recuperacao`` no momento do default. Integra-se a
    densidade de default ao longo do prazo, discretizada mensalmente.
    """
    n = int(round(prazo * passos_por_ano))
    bordas = np.linspace(0.0, prazo, n + 1)
    S = sobrevivencia(bordas, hazards, vertices)
    prob_default = S[:-1] - S[1:]
    meio = 0.5 * (bordas[:-1] + bordas[1:])
    return float((1 - recuperacao) * np.sum(prob_default * desconto(meio, taxa)))


def spread_justo(
    prazo: float,
    hazards: np.ndarray,
    vertices: np.ndarray,
    taxa: float,
    recuperacao: float = 0.40,
    por_ano: int = 4,
) -> float:
    """Spread que zera o valor do contrato: proteção dividida por prêmio unitário."""
    return perna_protecao(prazo, hazards, vertices, taxa, recuperacao) / perna_premio(
        prazo, hazards, vertices, taxa, por_ano
    )


def spread_aproximado(hazard: float, recuperacao: float = 0.40) -> float:
    """Aproximação de mercado: ``spread ≈ hazard × (1 - recuperação)``.

    Vale bem para spreads pequenos e serve de sanidade — mas ignora desconto e
    o efeito da sobrevivência sobre a perna de prêmio, então se degrada quando o
    crédito piora. O capítulo mede o erro.
    """
    return hazard * (1 - recuperacao)


@dataclass
class CurvaHazard:
    """Curva de intensidade de default extraída de spreads de mercado."""

    vertices: np.ndarray
    hazards: np.ndarray
    recuperacao: float
    taxa: float

    def sobrevivencia(self, t):
        return sobrevivencia(t, self.hazards, self.vertices)

    def pd_acumulada(self, t):
        return pd_acumulada(t, self.hazards, self.vertices)

    def spread(self, prazo: float) -> float:
        return spread_justo(prazo, self.hazards, self.vertices, self.taxa,
                            self.recuperacao)

    def como_tabela(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vértice (anos)": self.vertices,
                "hazard": self.hazards,
                "PD acumulada": self.pd_acumulada(self.vertices),
            }
        )


def bootstrap_hazards(
    prazos: np.ndarray,
    spreads: np.ndarray,
    taxa: float,
    recuperacao: float = 0.40,
    por_ano: int = 4,
) -> CurvaHazard:
    """Extrai a curva de hazard de spreads de CDS de vários prazos.

    Procede em cascata: o CDS de menor prazo determina a primeira intensidade;
    com ela fixa, o CDS seguinte determina a segunda; e assim por diante. Cada
    passo resolve uma equação escalar — o spread teórico tem de igualar o
    observado.

    É o mesmo raciocínio do bootstrap de curva de juros, com a diferença de que
    aqui a incógnita é a intensidade de default e não a taxa a termo.
    """
    prazos = np.asarray(prazos, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if len(prazos) != len(spreads):
        raise ValueError("um spread por prazo")
    if not np.all(np.diff(prazos) > 0):
        raise ValueError("os prazos devem estar em ordem crescente")

    hazards = np.zeros(len(prazos))
    for i, (prazo, alvo) in enumerate(zip(prazos, spreads, strict=True)):
        def erro(h: float, i=i, prazo=prazo, alvo=alvo) -> float:
            tentativa = hazards.copy()
            tentativa[i] = h
            return spread_justo(prazo, tentativa[: i + 1], prazos[: i + 1],
                                taxa, recuperacao, por_ano) - alvo

        hazards[i] = optimize.brentq(erro, 1e-10, 5.0, xtol=1e-12)
    return CurvaHazard(vertices=prazos, hazards=hazards,
                       recuperacao=recuperacao, taxa=taxa)


def pd_fisica_de_neutra(
    pd_neutra: float, premio_de_risco: float, prazo: float = 1.0
) -> float:
    """Converte PD neutra ao risco em PD física, dado um prêmio de risco.

    Sob o modelo de fator único, a passagem entre as duas medidas desloca o
    limiar de default:

    .. math:: PD^{\\mathbb{P}} = \\Phi\\big(\\Phi^{-1}(PD^{\\mathbb{Q}}) - \\eta\\sqrt{T}\\big),

    onde :math:`\\eta` é o prêmio de risco (preço de mercado do risco). Com
    :math:`\\eta > 0`, a PD física é menor — como deve ser.
    """
    from scipy import stats

    return float(
        stats.norm.cdf(stats.norm.ppf(pd_neutra) - premio_de_risco * np.sqrt(prazo))
    )


def premio_implicito(pd_neutra: float, pd_fisica: float, prazo: float = 1.0) -> float:
    """Prêmio de risco implícito no par de probabilidades."""
    from scipy import stats

    return float(
        (stats.norm.ppf(pd_neutra) - stats.norm.ppf(pd_fisica)) / np.sqrt(prazo)
    )
