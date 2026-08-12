"""Risco de crédito estruturado: CDOs — capítulo 11.

Uma tranche de CDO redistribui a perda de uma carteira. Quem compra a tranche
mais júnior absorve os primeiros prejuízos; quem compra a sênior só perde depois
que os subordinados forem consumidos.

A promessa da estrutura é criar títulos de alta qualidade a partir de uma
carteira de qualidade média. A promessa funciona — sob a hipótese de que a
correlação é a que se supôs. Este módulo permite medir o que acontece quando não
é.

A ferramenta é a mesma dos capítulos 6 e 7: o modelo de fator único. A diferença
é que aqui a saída não é o quantil da perda, é o valor esperado de uma função
não-linear dela — e não-linearidade com incerteza de parâmetro é onde as coisas
quebram.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats


def perda_tranche(
    perda_carteira: np.ndarray, attach: float, detach: float
) -> np.ndarray:
    """Perda da tranche, como fração de seu próprio tamanho.

    A tranche entre ``attach`` e ``detach`` (ambos em fração da carteira) absorve
    a parte da perda que cai nessa faixa:

    .. math::
       L_{\\text{tranche}} = \\frac{\\min(L, D) - \\min(L, A)}{D - A}.
    """
    if not 0.0 <= attach < detach <= 1.0:
        raise ValueError("é preciso 0 <= attach < detach <= 1")
    L = np.asarray(perda_carteira, dtype=float)
    return (np.minimum(L, detach) - np.minimum(L, attach)) / (detach - attach)


def taxa_condicional(x: np.ndarray | float, pd_inc: float, rho: float) -> np.ndarray:
    """Taxa de default condicional ao fator sistêmico."""
    limiar = stats.norm.ppf(pd_inc)
    x = np.asarray(x, dtype=float)
    return stats.norm.cdf((limiar - np.sqrt(rho) * x) / np.sqrt(1 - rho))


def el_tranche_lhp(
    attach: float,
    detach: float,
    pd_inc: float,
    rho: float,
    lgd: float = 0.60,
    limite: float = 8.0,
) -> float:
    """Perda esperada da tranche no limite de carteira homogênea infinita (LHP).

    Com infinitos devedores idênticos, a perda da carteira condicional ao fator
    é determinística e vale ``LGD × p(x)``. A perda esperada da tranche é então
    uma integral unidimensional sobre o fator:

    .. math::
       E[L_{\\text{tranche}}] = \\int L_{\\text{tranche}}\\big(LGD \\cdot p(x)\\big)\\,\\phi(x)\\,dx.

    É a aproximação padrão de mercado e permite ver o efeito da correlação sem
    nenhum ruído de simulação.
    """
    def integrando(x: float) -> float:
        perda = lgd * float(taxa_condicional(x, pd_inc, rho))
        return float(perda_tranche(np.array([perda]), attach, detach)[0]) * stats.norm.pdf(x)

    valor, _ = integrate.quad(integrando, -limite, limite, limit=300)
    return float(valor)


def simular_perdas_carteira(
    n_obrigados: int,
    pd_inc: float,
    rho: float,
    lgd: float = 0.60,
    n_simulacoes: int = 100_000,
    semente: int = 42,
    bloco: int = 20_000,
) -> np.ndarray:
    """Perda da carteira (fração do nocional) por Monte Carlo no fator único."""
    rng = np.random.default_rng(semente)
    limiar = stats.norm.ppf(pd_inc)
    perdas = np.empty(n_simulacoes)

    # A matriz de ruído tem bloco x n_obrigados entradas. Sem este teto, uma
    # carteira grande com bloco grande estoura a memória antes de simular nada.
    bloco = max(1, min(bloco, 20_000_000 // max(n_obrigados, 1)))

    feito = 0
    while feito < n_simulacoes:
        m = min(bloco, n_simulacoes - feito)
        X = rng.normal(0.0, 1.0, size=(m, 1))
        eps = rng.normal(0.0, 1.0, size=(m, n_obrigados))
        Z = np.sqrt(rho) * X + np.sqrt(1 - rho) * eps
        perdas[feito : feito + m] = lgd * (Z < limiar).mean(axis=1)
        feito += m
    return perdas


def el_tranche_mc(
    perdas_carteira: np.ndarray, attach: float, detach: float
) -> float:
    """Perda esperada da tranche a partir de perdas simuladas da carteira."""
    return float(perda_tranche(perdas_carteira, attach, detach).mean())


def spread_tranche(
    el: float, prazo: float = 5.0, taxa: float = 0.04, recuperacao: float = 0.0
) -> float:
    """Spread anual aproximado que compensa a perda esperada da tranche.

    Aproximação deliberadamente simples — perda esperada anualizada dividida
    pelo fator de anuidade. Serve para converter perda em preço e comparar
    tranches, não para precificar de verdade.
    """
    datas = np.arange(1, int(prazo) + 1)
    anuidade = float(np.sum(np.exp(-taxa * datas)))
    return float(el * (1 - recuperacao) * np.exp(-taxa * prazo) / anuidade)


def estrutura_padrao() -> pd.DataFrame:
    """Estrutura de tranches no padrão dos índices sintéticos de crédito."""
    return pd.DataFrame(
        {
            "tranche": ["equity", "júnior mezanino", "mezanino", "sênior", "supersênior"],
            "attach": [0.00, 0.03, 0.07, 0.10, 0.15],
            "detach": [0.03, 0.07, 0.10, 0.15, 0.30],
        }
    )


def tabela_tranches(
    pd_inc: float,
    rho: float,
    lgd: float = 0.60,
    estrutura: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Perda esperada de cada tranche da estrutura, pela aproximação LHP."""
    estrutura = estrutura_padrao() if estrutura is None else estrutura
    saida = estrutura.copy()
    saida["EL"] = [
        el_tranche_lhp(a, d, pd_inc, rho, lgd)
        for a, d in zip(estrutura["attach"], estrutura["detach"], strict=True)
    ]
    saida["spread (bps)"] = [spread_tranche(el) * 1e4 for el in saida["EL"]]
    return saida


def sensibilidade_a_correlacao(
    attach: float,
    detach: float,
    pd_inc: float,
    lgd: float = 0.60,
    rhos: np.ndarray | None = None,
) -> pd.DataFrame:
    """Perda esperada da tranche em função da correlação suposta.

    O comportamento é oposto nas pontas da estrutura, e é a característica mais
    importante do produto: correlação alta **reduz** a perda da tranche júnior e
    **aumenta** a da sênior. A intuição é que correlação alta torna os extremos
    mais prováveis — muitos defaults juntos ou quase nenhum. O júnior, que já
    perde tudo em qualquer cenário mediano, se beneficia da chance de nada
    acontecer; o sênior, que só é atingido em catástrofe, sofre com o aumento da
    chance de catástrofe.
    """
    rhos = np.linspace(0.01, 0.60, 40) if rhos is None else np.asarray(rhos)
    return pd.DataFrame(
        {
            "rho": rhos,
            "EL": [el_tranche_lhp(attach, detach, pd_inc, r, lgd) for r in rhos],
        }
    )


def correlacao_implicita(
    el_alvo: float,
    attach: float,
    detach: float,
    pd_inc: float,
    lgd: float = 0.60,
) -> float:
    """Correlação que reproduz uma perda esperada observada.

    É o análogo da volatilidade implícita: em vez de acreditar num parâmetro,
    pergunta-se qual valor de :math:`\\rho` o preço de mercado embute. Quando
    tranches distintas da **mesma** carteira exigem correlações diferentes, o
    modelo está rejeitado pelos próprios preços que deveria explicar.
    """
    from scipy import optimize

    def erro(r: float) -> float:
        return el_tranche_lhp(attach, detach, pd_inc, r, lgd) - el_alvo

    baixo, alto = 1e-4, 0.98
    if erro(baixo) * erro(alto) > 0:
        return float("nan")
    return float(optimize.brentq(erro, baixo, alto, xtol=1e-10))
