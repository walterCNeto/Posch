"""Validação de modelos de carteira — capítulo 9.

Validar um modelo de carteira é mais difícil que validar um sistema de rating,
por uma razão aritmética simples: o objeto a validar é uma **distribuição
inteira**, e dela se observa uma realização por ano.

Um sistema de rating com dez mil devedores gera dez mil observações por ano para
testar calibração. Um modelo de carteira gera **uma**: a perda daquele ano. Vinte
anos de histórico dão vinte pontos para julgar se a cauda de 99,9% está certa.

O instrumento padrão é a transformada integral de probabilidade: se o modelo
está certo, a posição percentual da perda realizada dentro da distribuição
prevista é uniforme em (0,1) e independente entre anos. Testar o modelo vira
testar uniformidade e independência de uma amostra pequena.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


def transformada_pit(
    perda_observada: np.ndarray, cenarios_por_ano: list[np.ndarray]
) -> np.ndarray:
    """Posição percentual de cada perda realizada na distribuição prevista.

    .. math:: u_t = \\hat F_t(L_t)

    Se o modelo está correto, os :math:`u_t` são uniformes em (0,1) e
    independentes entre si — qualquer que seja o formato da distribuição
    prevista, e mesmo que ela mude de ano para ano.

    É essa invariância que torna a transformada útil: ela reduz "a distribuição
    está certa?" a "esta amostra é uniforme?".
    """
    perda_observada = np.asarray(perda_observada, dtype=float)
    if len(perda_observada) != len(cenarios_por_ano):
        raise ValueError("um conjunto de cenários por ano observado")

    u = np.empty(len(perda_observada))
    for t, (perda, cenarios) in enumerate(
        zip(perda_observada, cenarios_por_ano, strict=True)
    ):
        cenarios = np.asarray(cenarios, dtype=float)
        u[t] = float((cenarios <= perda).mean())
    # Evita 0 e 1 exatos, que quebram a transformação para a normal.
    n = max(len(cenarios_por_ano[0]), 2)
    return np.clip(u, 1.0 / (2 * n), 1.0 - 1.0 / (2 * n))


def teste_uniformidade(u: np.ndarray) -> dict[str, float]:
    """Kolmogorov-Smirnov contra a uniforme (0,1)."""
    estat, p = stats.kstest(np.asarray(u, float), "uniform")
    return {"estatistica": float(estat), "p_valor": float(p),
            "rejeita": bool(p < 0.05)}


@dataclass
class ResultadoBerkowitz:
    """Saída do teste de Berkowitz."""

    estatistica: float
    p_valor: float
    graus_liberdade: int
    media: float
    desvio: float
    autocorrelacao: float

    @property
    def rejeita(self) -> bool:
        return self.p_valor < 0.05

    def como_serie(self) -> pd.Series:
        return pd.Series(
            {
                "estatística LR": self.estatistica,
                "p-valor": self.p_valor,
                "média (0 se ok)": self.media,
                "desvio (1 se ok)": self.desvio,
                "autocorrelação (0 se ok)": self.autocorrelacao,
            }
        )


def teste_berkowitz(u: np.ndarray) -> ResultadoBerkowitz:
    """Teste de Berkowitz sobre a transformada integral de probabilidade.

    Aplica :math:`z_t = \\Phi^{-1}(u_t)`, que sob o modelo correto é normal
    padrão independente, e testa por razão de verossimilhanças a hipótese
    conjunta

    .. math:: \\mu = 0,\\quad \\sigma^2 = 1,\\quad \\varphi = 0

    em um AR(1) ajustado a :math:`z_t`. Testar na escala normal em vez de testar
    uniformidade diretamente dá mais poder contra erros de cauda, que é onde um
    modelo de capital erra e importa.

    Os três parâmetros têm leitura direta:

    * :math:`\\mu \\ne 0` — o modelo erra o nível da perda;
    * :math:`\\sigma \\ne 1` — erra a dispersão (tipicamente, subestima a cauda);
    * :math:`\\varphi \\ne 0` — sobra dependência temporal que o modelo não captura.
    """
    z = stats.norm.ppf(np.clip(np.asarray(u, float), 1e-8, 1 - 1e-8))
    n = len(z)
    if n < 4:
        raise ValueError("são necessários pelo menos quatro anos")

    def log_lik(mu: float, sigma: float, phi: float) -> float:
        if sigma <= 1e-8 or abs(phi) >= 0.999:
            return -np.inf
        # primeira observação pela distribuição estacionária
        var0 = sigma**2 / (1 - phi**2)
        ll = -0.5 * np.log(2 * np.pi * var0) - (z[0] - mu / (1 - phi)) ** 2 / (2 * var0)
        resid = z[1:] - mu - phi * z[:-1]
        ll += np.sum(
            -0.5 * np.log(2 * np.pi * sigma**2) - resid**2 / (2 * sigma**2)
        )
        return float(ll)

    restrito = log_lik(0.0, 1.0, 0.0)

    saida = optimize.minimize(
        lambda th: -log_lik(th[0], np.exp(th[1]), np.tanh(th[2])),
        x0=np.array([float(z.mean()), float(np.log(max(z.std(ddof=1), 1e-3))), 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )
    irrestrito = -float(saida.fun)
    mu, sigma, phi = saida.x[0], float(np.exp(saida.x[1])), float(np.tanh(saida.x[2]))

    estat = max(2.0 * (irrestrito - restrito), 0.0)
    return ResultadoBerkowitz(
        estatistica=float(estat),
        p_valor=float(stats.chi2.sf(estat, 3)),
        graus_liberdade=3,
        media=float(mu),
        desvio=sigma,
        autocorrelacao=phi,
    )


def teste_excedencias(u: np.ndarray, nivel: float = 0.99) -> dict[str, float]:
    """Conta violações do quantil e testa a frequência por binomial.

    É o teste mais simples e o mais usado — e o de menor poder. Com vinte anos
    e nível de 99%, o número esperado de violações é 0,2: observar zero é
    perfeitamente compatível com um modelo bom **e** com um modelo péssimo.
    """
    u = np.asarray(u, float)
    n = len(u)
    violacoes = int((u > nivel).sum())
    esperadas = n * (1 - nivel)
    p_valor = float(stats.binom.sf(violacoes - 1, n, 1 - nivel))
    return {
        "anos": n,
        "violacoes": violacoes,
        "esperadas": float(esperadas),
        "p_valor": p_valor,
        "rejeita": bool(p_valor < 0.05),
    }


def poder_do_teste(
    gerar_u,
    n_repeticoes: int = 200,
    teste: str = "berkowitz",
) -> float:
    """Fração de vezes em que o teste rejeita, para uma dada forma de gerar u.

    Chamada com um gerador que produz dados sob o modelo correto, devolve a
    **taxa de erro tipo I** (deveria ficar perto de 5%). Chamada com um gerador
    sob modelo errado, devolve o **poder** — a chance de detectar o erro.

    Poder baixo não significa que o modelo está certo. Significa que o teste não
    consegue distinguir, o que é uma informação bem diferente e frequentemente
    reportada como se fosse aprovação.
    """
    rejeicoes = 0
    for k in range(n_repeticoes):
        u = gerar_u(k)
        if teste == "berkowitz":
            rejeicoes += teste_berkowitz(u).rejeita
        elif teste == "ks":
            rejeicoes += teste_uniformidade(u)["rejeita"]
        elif teste == "excedencias":
            rejeicoes += teste_excedencias(u)["rejeita"]
        else:
            raise ValueError(f"teste desconhecido: {teste}")
    return rejeicoes / n_repeticoes
