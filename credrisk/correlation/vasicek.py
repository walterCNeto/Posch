"""Correlação de ativos no modelo de fator único — capítulo 6.

A correlação de ativos é o parâmetro que transforma um conjunto de PDs
individuais em uma distribuição de perda de carteira. Sem ela, a perda de uma
carteira grande seria praticamente determinística e não haveria capital
econômico a discutir.

Ela também é, de longe, o parâmetro mais difícil de estimar de todo o curso. A
informação sobre :math:`\\rho` vem da **variação da taxa de default entre anos**,
e uma carteira observada por vinte anos fornece vinte observações do fator
sistêmico — por mais devedores que tenha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


def taxa_condicional(x: np.ndarray | float, pd_inc: float, rho: float) -> np.ndarray:
    """Taxa de default condicional ao fator sistêmico.

    .. math:: p(x) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) - \\sqrt{\\rho}\\,x}{\\sqrt{1-\\rho}}\\right)

    Valores altos de ``x`` correspondem a bons estados da economia.
    """
    limiar = stats.norm.ppf(pd_inc)
    x = np.asarray(x, dtype=float)
    return stats.norm.cdf((limiar - np.sqrt(rho) * x) / np.sqrt(1 - rho))


def densidade_vasicek(taxa: np.ndarray, pd_inc: float, rho: float) -> np.ndarray:
    """Densidade da taxa de default no limite de carteira infinitamente granular.

    Com infinitos devedores, a taxa observada converge para :math:`p(X)` e sua
    densidade tem forma fechada. É a distribuição de Vasicek — assimétrica à
    direita, o que explica por que a perda média de uma carteira é menor que a
    perda típica de um ano ruim.
    """
    taxa = np.clip(np.asarray(taxa, dtype=float), 1e-12, 1 - 1e-12)
    a = stats.norm.ppf(taxa)
    b = stats.norm.ppf(pd_inc)
    termo = np.sqrt((1 - rho) / rho) * np.exp(
        0.5 * a**2 - 0.5 * (np.sqrt(1 - rho) * a - b) ** 2 / rho
    )
    return termo


def estimar_momentos(taxas: np.ndarray) -> dict[str, float]:
    """Estimador de momentos: casa média e variância das taxas observadas.

    No limite granular, :math:`E[p(X)] = PD` e a variância da taxa é

    .. math:: \\operatorname{Var}[p(X)] = \\Phi_2(\\Phi^{-1}(PD), \\Phi^{-1}(PD); \\rho) - PD^2,

    onde :math:`\\Phi_2` é a normal bivariada acumulada. Invertendo
    numericamente essa relação obtém-se :math:`\\rho`.

    É simples, rápido e não usa toda a informação da amostra — serve de valor
    inicial e de checagem para o estimador de máxima verossimilhança.
    """
    taxas = np.asarray(taxas, dtype=float)
    pd_est = float(taxas.mean())
    var_obs = float(taxas.var(ddof=1))

    limiar = stats.norm.ppf(pd_est)

    def var_teorica(rho: float) -> float:
        if rho <= 1e-8:
            return 0.0
        conjunta = stats.multivariate_normal.cdf(
            [limiar, limiar],
            mean=[0.0, 0.0],
            cov=[[1.0, rho], [rho, 1.0]],
        )
        return float(conjunta) - pd_est**2

    alvo = lambda r: var_teorica(r) - var_obs  # noqa: E731
    if alvo(1e-6) > 0:
        rho_est = 0.0
    elif alvo(0.95) < 0:
        rho_est = 0.95
    else:
        rho_est = float(optimize.brentq(alvo, 1e-6, 0.95, xtol=1e-10))

    return {"PD": pd_est, "RHO": rho_est}


def _nos_gauss_hermite(n_nos: int) -> tuple[np.ndarray, np.ndarray]:
    """Nós e pesos para integrar contra a densidade normal padrão.

    ``numpy.polynomial.hermite.hermgauss`` integra contra :math:`e^{-t^2}`; a
    mudança de variável :math:`x=\\sqrt{2}\\,t` e a divisão por :math:`\\sqrt{\\pi}`
    convertem para integração contra :math:`\\phi(x)`.
    """
    t, w = np.polynomial.hermite.hermgauss(n_nos)
    return np.sqrt(2.0) * t, w / np.sqrt(np.pi)


def _log_binomial(defaults: np.ndarray, n_obrigados: np.ndarray, p: np.ndarray) -> np.ndarray:
    """log da massa binomial, com p podendo ser matriz (anos x nós)."""
    p = np.clip(p, 1e-300, 1 - 1e-15)
    log_comb = (
        special.gammaln(n_obrigados + 1)
        - special.gammaln(defaults + 1)
        - special.gammaln(n_obrigados - defaults + 1)
    )
    return log_comb + defaults * np.log(p) + (n_obrigados - defaults) * np.log1p(-p)


def log_verossimilhanca_simples(
    defaults: np.ndarray,
    n_obrigados: np.ndarray,
    pd_inc: float,
    rho: float,
    n_nos: int = 64,
) -> float:
    """Quadratura de Gauss-Hermite padrão, com nós fixos.

    Presente para efeito de comparação. Os nós ficam espalhados sobre toda a
    normal padrão, enquanto o integrando é um pico estreito em torno do fator
    que explica os defaults daquele ano. Quando a carteira é grande ou a
    correlação é alta, o pico fica mais estreito que o espaçamento dos nós e a
    quadratura devolve um número errado **sem sinalizar erro algum**.

    Use :func:`log_verossimilhanca`, que desloca os nós para o pico.
    """
    if not (1e-8 < pd_inc < 1 - 1e-8) or not (1e-8 < rho < 1 - 1e-8):
        return -np.inf

    x, w = _nos_gauss_hermite(n_nos)
    p = taxa_condicional(x, pd_inc, rho)

    defaults = np.asarray(defaults, dtype=float)[:, None]
    n_obrigados = np.asarray(n_obrigados, dtype=float)[:, None]

    log_termo = _log_binomial(defaults, n_obrigados, p[None, :]) + np.log(
        np.clip(w, 1e-300, None)
    )[None, :]
    return float(special.logsumexp(log_termo, axis=1).sum())


def log_verossimilhanca(
    defaults: np.ndarray,
    n_obrigados: np.ndarray,
    pd_inc: float,
    rho: float,
    n_nos: int = 30,
) -> float:
    """Log-verossimilhança por quadratura de Gauss-Hermite **adaptativa**.

    Cada ano contribui com

    .. math::
       \\Pr(D_t = d) = \\int \\binom{N_t}{d} p(x)^d (1-p(x))^{N_t-d}\\,\\phi(x)\\,dx.

    O integrando é concentrado: com mil devedores, apenas uma faixa estreita de
    :math:`x` é compatível com o número de defaults observado. A quadratura
    adaptativa localiza o modo :math:`\\hat x_t` do integrando, mede sua
    curvatura e desloca os nós para lá — o mesmo procedimento usado em modelos
    mistos generalizados.

    Com isso, trinta nós entregam mais precisão que centenas de nós fixos.
    """
    if not (1e-8 < pd_inc < 1 - 1e-8) or not (1e-8 < rho < 1 - 1e-8):
        return -np.inf

    t_nos, w = np.polynomial.hermite.hermgauss(n_nos)
    log_w = np.log(np.clip(w, 1e-300, None))

    defaults = np.asarray(defaults, dtype=float)
    n_obrigados = np.asarray(n_obrigados, dtype=float)

    def g(x: np.ndarray, d: np.ndarray, n: np.ndarray) -> np.ndarray:
        """log do integrando: binomial vezes densidade normal."""
        p = taxa_condicional(x, pd_inc, rho)
        return (
            _log_binomial(d, n, p)
            - 0.5 * x**2
            - 0.5 * np.log(2 * np.pi)
        )

    # Modo do integrando, um por ano, achado em paralelo por Newton.
    # O chute inicial é o fator que reproduziria exatamente a taxa observada,
    # que já cai muito perto do pico.
    taxa = np.clip(defaults / n_obrigados, 1e-9, 1 - 1e-9)
    modo = np.clip(
        (stats.norm.ppf(pd_inc) - np.sqrt(1 - rho) * stats.norm.ppf(taxa)) / np.sqrt(rho),
        -8.0,
        8.0,
    )

    h = 1e-4
    for _ in range(30):
        g_mais = g(modo + h, defaults, n_obrigados)
        g_menos = g(modo - h, defaults, n_obrigados)
        g_centro = g(modo, defaults, n_obrigados)
        primeira = (g_mais - g_menos) / (2 * h)
        segunda = (g_mais - 2 * g_centro + g_menos) / h**2
        segunda = np.where(segunda > -1e-10, -1e-10, segunda)
        passo = np.clip(primeira / segunda, -1.0, 1.0)
        modo = np.clip(modo - passo, -9.0, 9.0)
        if np.max(np.abs(passo)) < 1e-10:
            break

    g_mais = g(modo + h, defaults, n_obrigados)
    g_menos = g(modo - h, defaults, n_obrigados)
    segunda = (g_mais - 2 * g(modo, defaults, n_obrigados) + g_menos) / h**2
    escala = np.sqrt(1.0 / np.maximum(-segunda, 1e-12))

    # Nós deslocados e reescalados para o entorno de cada modo: (anos, nós)
    z = modo[:, None] + np.sqrt(2.0) * escala[:, None] * t_nos[None, :]
    log_termo = (
        log_w[None, :]
        + (t_nos**2)[None, :]
        + g(z, defaults[:, None], n_obrigados[:, None])
    )
    log_ano = special.logsumexp(log_termo, axis=1) + np.log(np.sqrt(2.0) * escala)
    return float(log_ano.sum())


@dataclass
class ResultadoML:
    """Saída da estimação por máxima verossimilhança."""

    PD: float
    RHO: float
    log_lik: float
    convergiu: bool

    def como_serie(self):
        import pandas as pd

        return pd.Series({"PD": self.PD, "RHO": self.RHO, "log-verossimilhança": self.log_lik})


def estimar_ml(
    defaults: np.ndarray,
    n_obrigados: np.ndarray,
    n_nos: int = 30,
    chute: dict[str, float] | None = None,
) -> ResultadoML:
    """Estima PD e rho por máxima verossimilhança.

    Otimiza na escala logit dos dois parâmetros, o que remove as restrições de
    intervalo e melhora muito o condicionamento — o otimizador irrestrito nunca
    propõe um :math:`\\rho` negativo.
    """
    defaults = np.asarray(defaults, dtype=float)
    n_obrigados = np.asarray(n_obrigados, dtype=float)

    if chute is None:
        chute = estimar_momentos(defaults / n_obrigados)
    inicio = np.array(
        [
            np.log(chute["PD"] / (1 - chute["PD"])),
            np.log(max(chute["RHO"], 1e-3) / (1 - max(chute["RHO"], 1e-3))),
        ]
    )

    def negativo(theta: np.ndarray) -> float:
        pd_i = 1.0 / (1.0 + np.exp(-theta[0]))
        rho = 1.0 / (1.0 + np.exp(-theta[1]))
        return -log_verossimilhanca(defaults, n_obrigados, pd_i, rho, n_nos)

    saida = optimize.minimize(negativo, inicio, method="Nelder-Mead",
                              options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 4000})

    pd_est = float(1.0 / (1.0 + np.exp(-saida.x[0])))
    rho_est = float(1.0 / (1.0 + np.exp(-saida.x[1])))
    return ResultadoML(PD=pd_est, RHO=rho_est, log_lik=float(-saida.fun),
                       convergiu=bool(saida.success))


def intervalo_perfil(
    defaults: np.ndarray,
    n_obrigados: np.ndarray,
    nivel: float = 0.95,
    n_nos: int = 30,
    grade: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Intervalo de confiança para rho por verossimilhança perfilada.

    Para cada :math:`\\rho` da grade, maximiza a verossimilhança sobre ``PD`` e
    guarda o valor. O intervalo é o conjunto de :math:`\\rho` cuja perda de
    log-verossimilhança em relação ao máximo é menor que metade do quantil
    qui-quadrado com um grau de liberdade.

    Perfilar é preferível ao erro-padrão assintótico aqui porque a
    verossimilhança em :math:`\\rho` é bastante assimétrica com poucos anos —
    e um intervalo simétrico chegaria a incluir valores negativos.

    Returns
    -------
    (inferior, superior, grade, log_lik_perfilada)
    """
    defaults = np.asarray(defaults, dtype=float)
    n_obrigados = np.asarray(n_obrigados, dtype=float)
    if grade is None:
        grade = np.linspace(0.005, 0.45, 45)

    media = float(np.mean(defaults / n_obrigados))
    centro = np.log(media / (1 - media))

    perfil = np.empty(len(grade))
    for i, rho in enumerate(grade):
        # Otimização escalar em PD (na escala logit), com o parâmetro de
        # interesse fixo. Busca em intervalo é bem mais barata que Nelder-Mead
        # e o perfil é chamado uma vez por ponto da grade.
        saida = optimize.minimize_scalar(
            lambda u, rho=rho: -log_verossimilhanca(
                defaults, n_obrigados, 1.0 / (1.0 + np.exp(-u)), rho, n_nos
            ),
            bounds=(centro - 2.5, centro + 2.5),
            method="bounded",
            options={"xatol": 1e-8},
        )
        perfil[i] = -float(saida.fun)

    corte = perfil.max() - 0.5 * stats.chi2.ppf(nivel, df=1)
    dentro = grade[perfil >= corte]
    if len(dentro) == 0:
        return float("nan"), float("nan"), grade, perfil
    return float(dentro.min()), float(dentro.max()), grade, perfil
