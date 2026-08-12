"""Abordagem estrutural de default — capítulo 2.

Merton (1974) enxerga o capital próprio de uma empresa alavancada como uma
opção de compra sobre o valor do ativo, com preço de exercício igual à dívida.
O default acontece se, no vencimento, o ativo valer menos que a dívida.

A dificuldade prática é que nem o valor do ativo nem sua volatilidade são
observáveis. Observa-se o valor de mercado do capital próprio. O procedimento
padrão — devido a KMV — inverte a fórmula de Black-Scholes dia a dia e itera
até que a volatilidade usada na inversão coincida com a volatilidade dos
retornos do ativo assim obtidos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


def _d1_d2(
    V: np.ndarray | float, L: float, r: float, T: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    V = np.asarray(V, dtype=float)
    raiz = sigma * np.sqrt(T)
    d1 = (np.log(V / L) + (r + 0.5 * sigma**2) * T) / raiz
    return d1, d1 - raiz


def preco_call(
    V: np.ndarray | float, L: float, r: float, T: float, sigma: float
) -> np.ndarray:
    """Preço de Black-Scholes de uma call europeia sobre o ativo.

    No modelo de Merton, é o valor de mercado do capital próprio:

    .. math:: E = V\\,\\Phi(d_1) - L e^{-rT}\\,\\Phi(d_2).
    """
    d1, d2 = _d1_d2(V, L, r, T, sigma)
    V = np.asarray(V, dtype=float)
    return V * stats.norm.cdf(d1) - L * np.exp(-r * T) * stats.norm.cdf(d2)


def delta_call(
    V: np.ndarray | float, L: float, r: float, T: float, sigma: float
) -> np.ndarray:
    """Sensibilidade do capital próprio ao valor do ativo, :math:`\\Phi(d_1)`."""
    d1, _ = _d1_d2(V, L, r, T, sigma)
    return stats.norm.cdf(d1)


def valor_ativo_implicito(
    E: np.ndarray | float, L: float, r: float, T: float, sigma: float
) -> np.ndarray:
    """Inverte Black-Scholes: dado o capital próprio, devolve o valor do ativo.

    O preço da call é estritamente crescente em ``V`` e sua derivada é
    :math:`\\Phi(d_1)`, conhecida em forma fechada — então Newton vetorizado
    resolve a série inteira de uma vez, em poucos passos.

    Newton pode falhar quando a opção está muito fora do dinheiro e
    :math:`\\Phi(d_1)` fica próximo de zero. Para os pontos que não convergirem,
    caímos em Brent, que é mais lento mas não falha por ser um método de
    intervalo.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))

    # Chute inicial: valor do ativo se a opção estivesse profundamente dentro
    # do dinheiro, quando E ≈ V - L e^{-rT}.
    V = E + L * np.exp(-r * T)

    for _ in range(60):
        residuo = preco_call(V, L, r, T, sigma) - E
        derivada = delta_call(V, L, r, T, sigma)
        derivada = np.where(derivada < 1e-12, 1e-12, derivada)
        passo = residuo / derivada
        V = np.maximum(V - passo, 1e-12)
        if np.max(np.abs(residuo)) < 1e-10 * np.maximum(1.0, np.max(E)):
            break

    # Garantia: qualquer ponto que Newton não tenha resolvido vai para Brent.
    residuo = np.abs(preco_call(V, L, r, T, sigma) - E)
    ruins = np.flatnonzero(residuo > 1e-6 * np.maximum(1.0, E))
    for i in ruins:
        e = float(E[i])
        V[i] = optimize.brentq(
            lambda v, e=e: preco_call(v, L, r, T, sigma) - e,
            max(e, 1e-8),
            e + L * 5.0,
            xtol=1e-10,
            rtol=1e-12,
        )
    return V


@dataclass
class ResultadoKMV:
    """Saída do procedimento iterativo de estimação do ativo."""

    V: np.ndarray
    sigma_V: float
    iteracoes: int
    convergiu: bool
    historico_sigma: list[float]


def estimar_iterativo(
    E: np.ndarray,
    L: float,
    r: float,
    T: float = 1.0,
    dias_ano: int = 260,
    sigma_inicial: float | None = None,
    max_iter: int = 200,
    tolerancia: float = 1e-8,
) -> ResultadoKMV:
    """Estima valor e volatilidade do ativo pelo procedimento iterativo (KMV).

    O algoritmo:

    1. parte de um chute para :math:`\\sigma_V` (por omissão, a volatilidade
       dos retornos do capital próprio, que é um limite superior);
    2. inverte Black-Scholes em cada dia para obter a série :math:`V_t`;
    3. calcula a volatilidade anualizada dos log-retornos dessa série;
    4. repete até que a volatilidade de entrada e a de saída coincidam.

    O ponto fixo existe porque a volatilidade do capital próprio e a do ativo
    se relacionam pela alavancagem da opção,
    :math:`\\sigma_E = \\frac{V}{E}\\Phi(d_1)\\,\\sigma_V`, e a alavancagem cai
    quando :math:`\\sigma_V` sobe.
    """
    E = np.asarray(E, dtype=float)

    if sigma_inicial is None:
        ret_E = np.diff(np.log(E))
        sigma_inicial = float(np.std(ret_E, ddof=1) * np.sqrt(dias_ano))

    sigma = float(sigma_inicial)
    historico = [sigma]
    convergiu = False
    iteracao = 0

    V = np.array([])
    for passo in range(1, max_iter + 1):
        iteracao = passo
        V = valor_ativo_implicito(E, L, r, T, sigma)
        retornos = np.diff(np.log(V))
        nova = float(np.std(retornos, ddof=1) * np.sqrt(dias_ano))
        historico.append(nova)
        if abs(nova - sigma) < tolerancia:
            sigma = nova
            convergiu = True
            break
        sigma = nova

    return ResultadoKMV(
        V=V,
        sigma_V=sigma,
        iteracoes=iteracao,
        convergiu=convergiu,
        historico_sigma=historico,
    )


def distancia_ao_default(
    V: float | np.ndarray,
    L: float | np.ndarray,
    mu: float,
    sigma_V: float | np.ndarray,
    T: float = 1.0,
) -> np.ndarray:
    """Distância ao default, em desvios-padrão do log do ativo.

    .. math::
       DD = \\frac{\\ln(V/L) + (\\mu - \\tfrac12\\sigma_V^2)T}{\\sigma_V\\sqrt{T}}

    Usa a deriva **física** :math:`\\mu`. Trocá-la pela taxa livre de risco
    devolve a distância neutra ao risco, que é outra coisa — ver
    :func:`pd_merton`.
    """
    V = np.asarray(V, dtype=float)
    L = np.asarray(L, dtype=float)
    sigma_V = np.asarray(sigma_V, dtype=float)
    return (np.log(V / L) + (mu - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))


def pd_merton(
    V: float | np.ndarray,
    L: float | np.ndarray,
    mu: float,
    sigma_V: float | np.ndarray,
    T: float = 1.0,
) -> np.ndarray:
    """Probabilidade de default do modelo: :math:`\\Phi(-DD)`.

    Com :math:`\\mu` igual à deriva física, é a PD do mundo real. Passando a
    taxa livre de risco no lugar de :math:`\\mu`, obtém-se a PD neutra ao risco,
    sistematicamente maior — a diferença é o prêmio de risco, e é o assunto do
    capítulo 10.
    """
    return stats.norm.cdf(-distancia_ao_default(V, L, mu, sigma_V, T))


def volatilidade_do_capital(
    V: float, E: float, L: float, r: float, T: float, sigma_V: float
) -> float:
    """Relação entre volatilidade do ativo e do capital próprio via alavancagem."""
    return (V / E) * float(delta_call(V, L, r, T, sigma_V)) * sigma_V
