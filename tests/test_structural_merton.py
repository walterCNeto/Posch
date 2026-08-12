"""Testes do capítulo 2.

O modelo de Merton tem várias identidades fechadas, e todas viram teste: paridade
entre call e put, limites da call, a relação de alavancagem entre volatilidades,
e — o mais importante — a recuperação dos parâmetros verdadeiros do gerador.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from credrisk.data.generators import (
    DIAS_UTEIS_ANO,
    PARAMS_MERTON,
    gerar_carteira_merton,
    gerar_serie_merton,
)
from credrisk.structural.merton import (
    delta_call,
    distancia_ao_default,
    estimar_iterativo,
    pd_merton,
    preco_call,
    valor_ativo_implicito,
    volatilidade_do_capital,
)

P = PARAMS_MERTON


def test_call_respeita_os_limites_de_nao_arbitragem() -> None:
    V, L, r, T, s = 100.0, 60.0, 0.05, 1.0, 0.28
    c = float(preco_call(V, L, r, T, s))
    assert max(V - L * np.exp(-r * T), 0.0) <= c <= V


def test_paridade_call_put() -> None:
    """C - P = V - L e^{-rT}, com a put obtida da fórmula fechada."""
    V, L, r, T, s = 120.0, 80.0, 0.04, 2.0, 0.35
    d1 = (np.log(V / L) + (r + 0.5 * s**2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    put = L * np.exp(-r * T) * stats.norm.cdf(-d2) - V * stats.norm.cdf(-d1)
    call = float(preco_call(V, L, r, T, s))
    assert call - put == pytest.approx(V - L * np.exp(-r * T), rel=1e-10)


def test_delta_e_a_derivada_da_call() -> None:
    V, L, r, T, s = 100.0, 60.0, 0.05, 1.0, 0.28
    h = 1e-5
    numerica = (preco_call(V + h, L, r, T, s) - preco_call(V - h, L, r, T, s)) / (2 * h)
    assert float(delta_call(V, L, r, T, s)) == pytest.approx(float(numerica), rel=1e-6)


def test_inversao_recupera_o_valor_do_ativo() -> None:
    V = np.array([80.0, 100.0, 250.0])
    L, r, T, s = 60.0, 0.05, 1.0, 0.28
    E = preco_call(V, L, r, T, s)
    np.testing.assert_allclose(valor_ativo_implicito(E, L, r, T, s), V, rtol=1e-9)


def test_gerador_e_consistente_com_black_scholes() -> None:
    """O E do gerador tem de ser exatamente a call sobre o V verdadeiro."""
    serie = gerar_serie_merton()
    esperado = preco_call(
        serie["V_verdadeiro"].to_numpy(), P["L"], P["r"], P["T"], P["sigma_V"]
    )
    np.testing.assert_allclose(serie["E"].to_numpy(), esperado, rtol=1e-12)


def test_kmv_recupera_a_volatilidade_verdadeira() -> None:
    """O teste central do capítulo: partindo só de E, o algoritmo acha sigma_V."""
    serie = gerar_serie_merton()
    r = estimar_iterativo(
        serie["E"].to_numpy(), P["L"], P["r"], P["T"], dias_ano=DIAS_UTEIS_ANO
    )
    assert r.convergiu, f"não convergiu em {r.iteracoes} iterações"
    # A volatilidade amostral de uma série de 260 dias tem erro de ~1/sqrt(2n).
    erro_amostral = P["sigma_V"] / np.sqrt(2 * (DIAS_UTEIS_ANO - 1))
    assert abs(r.sigma_V - P["sigma_V"]) < 4 * erro_amostral


def test_kmv_recupera_a_serie_do_ativo() -> None:
    serie = gerar_serie_merton()
    r = estimar_iterativo(serie["E"].to_numpy(), P["L"], P["r"], P["T"])
    erro_relativo = np.abs(r.V - serie["V_verdadeiro"].to_numpy()) / serie["V_verdadeiro"]
    assert erro_relativo.max() < 0.02


def test_kmv_independe_do_chute_inicial() -> None:
    serie = gerar_serie_merton()
    a = estimar_iterativo(serie["E"].to_numpy(), P["L"], P["r"], P["T"], sigma_inicial=0.10)
    b = estimar_iterativo(serie["E"].to_numpy(), P["L"], P["r"], P["T"], sigma_inicial=0.90)
    assert a.sigma_V == pytest.approx(b.sigma_V, rel=1e-5)


def test_relacao_de_alavancagem_entre_volatilidades() -> None:
    V, L, r, T, s = 100_000.0, 60_000.0, 0.05, 1.0, 0.28
    E = float(preco_call(V, L, r, T, s))
    sigma_E = volatilidade_do_capital(V, E, L, r, T, s)
    # O capital próprio é sempre mais volátil que o ativo em empresa alavancada.
    assert sigma_E > s


def test_pd_cresce_com_alavancagem_e_com_volatilidade() -> None:
    base = float(pd_merton(100.0, 60.0, 0.08, 0.28))
    mais_divida = float(pd_merton(100.0, 80.0, 0.08, 0.28))
    mais_vol = float(pd_merton(100.0, 60.0, 0.08, 0.45))
    assert mais_divida > base
    assert mais_vol > base


def test_pd_neutra_ao_risco_supera_a_fisica() -> None:
    """Trocar mu pela taxa livre de risco aumenta a PD — é o prêmio de risco."""
    fisica = float(pd_merton(100.0, 60.0, P["mu"], 0.28))
    neutra = float(pd_merton(100.0, 60.0, P["r"], 0.28))
    assert neutra > fisica


def test_dd_e_pd_sao_a_mesma_informacao() -> None:
    carteira = gerar_carteira_merton(n_empresas=50)
    dd = distancia_ao_default(
        carteira["V"], carteira["L"], P["mu"], carteira["sigma_V"]
    )
    p = pd_merton(carteira["V"], carteira["L"], P["mu"], carteira["sigma_V"])
    # Ordenação por DD é a ordenação inversa por PD, exatamente.
    assert np.all(np.argsort(dd) == np.argsort(-p))


def test_carteira_tem_dispersao_de_risco() -> None:
    carteira = gerar_carteira_merton()
    p = pd_merton(carteira["V"], carteira["L"], P["mu"], carteira["sigma_V"])
    assert p.min() < 1e-4
    assert p.max() > 0.05
