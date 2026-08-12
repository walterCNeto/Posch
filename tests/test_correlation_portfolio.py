"""Testes dos capítulos 6 (correlação de ativos) e 7 (risco de carteira).

Estes são os dois capítulos em que um erro não se anuncia: uma quadratura mal
condicionada e um estimador de cauda com poucas amostras devolvem números
plausíveis e errados. Por isso a bateria aqui é mais pesada, e todo resultado
que tem forma fechada é conferido contra ela.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from credrisk.correlation.vasicek import (
    _nos_gauss_hermite,
    densidade_vasicek,
    estimar_ml,
    estimar_momentos,
    intervalo_perfil,
    log_verossimilhanca,
    log_verossimilhanca_simples,
    taxa_condicional,
)
from credrisk.data.generators import (
    PARAMS_FATOR,
    gerar_carteira,
    gerar_taxas_vasicek,
)
from credrisk.portfolio.montecarlo import (
    contribuicao_por_posicao,
    erro_padrao_quantil,
    perda_analitica,
    quantil_vasicek,
    simular_com_is,
    simular_perdas,
)

PD_V = PARAMS_FATOR["PD"]
RHO_V = PARAMS_FATOR["RHO"]

# ---------------------------------------------------------------- capítulo 6


def test_taxa_condicional_tem_media_igual_a_pd() -> None:
    """E[p(X)] = PD por construção, para qualquer rho."""
    x, w = _nos_gauss_hermite(80)
    for rho in [0.03, 0.12, 0.35, 0.60]:
        media = float(np.sum(w * taxa_condicional(x, PD_V, rho)))
        assert media == pytest.approx(PD_V, rel=1e-6), f"rho={rho}"


def test_taxa_condicional_e_monotona_no_fator() -> None:
    x = np.linspace(-3, 3, 50)
    p = taxa_condicional(x, PD_V, RHO_V)
    assert np.all(np.diff(p) < 0), "fator alto = bom estado = menos default"


def test_densidade_vasicek_integra_um_e_tem_media_pd() -> None:
    for rho in [0.05, 0.12, 0.30]:
        integral, _ = integrate.quad(
            lambda t, r=rho: densidade_vasicek(t, PD_V, r), 1e-9, 1 - 1e-9, limit=200
        )
        media, _ = integrate.quad(
            lambda t, r=rho: t * densidade_vasicek(t, PD_V, r), 1e-9, 1 - 1e-9, limit=200
        )
        assert integral == pytest.approx(1.0, abs=1e-5), f"rho={rho}"
        assert media == pytest.approx(PD_V, abs=1e-6), f"rho={rho}"


def test_quadratura_adaptativa_bate_com_integracao_direta() -> None:
    """Referência: integração numérica centrada no modo do integrando."""
    d, n = 25.0, 1000.0
    for rho in [0.05, 0.12, 0.30, 0.50]:
        modo = (
            stats.norm.ppf(PD_V) - np.sqrt(1 - rho) * stats.norm.ppf(d / n)
        ) / np.sqrt(rho)
        referencia, _ = integrate.quad(
            lambda x, r=rho: stats.binom.pmf(d, n, taxa_condicional(x, PD_V, r))
            * stats.norm.pdf(x),
            modo - 4,
            modo + 4,
            limit=400,
        )
        nosso = log_verossimilhanca(np.array([d]), np.array([n]), PD_V, rho, n_nos=30)
        assert nosso == pytest.approx(np.log(referencia), abs=1e-6), f"rho={rho}"


def test_quadratura_ingenua_erra_onde_a_adaptativa_acerta() -> None:
    """O resultado que motiva a quadratura adaptativa.

    Com correlação alta o integrando fica estreito demais para nós fixos, e a
    versão ingênua devolve um valor errado sem qualquer sinal de erro.
    """
    d, n, rho = 25.0, 1000.0, 0.50
    adaptativa = log_verossimilhanca(np.array([d]), np.array([n]), PD_V, rho, n_nos=30)
    ingenua = log_verossimilhanca_simples(
        np.array([d]), np.array([n]), PD_V, rho, n_nos=64
    )
    assert abs(ingenua - adaptativa) > 0.1


def test_verossimilhanca_rejeita_parametros_invalidos() -> None:
    d, n = np.array([25.0]), np.array([1000.0])
    assert log_verossimilhanca(d, n, PD_V, 0.0) == -np.inf
    assert log_verossimilhanca(d, n, PD_V, 1.0) == -np.inf
    assert log_verossimilhanca(d, n, 0.0, RHO_V) == -np.inf


def test_estimador_de_momentos_recupera_os_parametros() -> None:
    grande = gerar_taxas_vasicek(n_anos=2000, n_obrigados=20000, semente=3)
    mm = estimar_momentos(grande["taxa_observada"].to_numpy())
    assert mm["PD"] == pytest.approx(PD_V, rel=0.06)
    assert mm["RHO"] == pytest.approx(RHO_V, rel=0.12)


def test_maxima_verossimilhanca_recupera_os_parametros() -> None:
    grande = gerar_taxas_vasicek(n_anos=800, n_obrigados=2000, semente=4)
    ml = estimar_ml(
        grande["Defaults"].to_numpy(), grande["N"].to_numpy()
    )
    assert ml.PD == pytest.approx(PD_V, rel=0.08)
    assert ml.RHO == pytest.approx(RHO_V, rel=0.15)


def test_verossimilhanca_e_maxima_no_estimador() -> None:
    """Checagem de sanidade do otimizador: nada da vizinhança supera o ótimo."""
    dados = gerar_taxas_vasicek(n_anos=40, semente=7)
    d, n = dados["Defaults"].to_numpy(), dados["N"].to_numpy()
    ml = estimar_ml(d, n)
    melhor = log_verossimilhanca(d, n, ml.PD, ml.RHO)
    for dp, dr in [(1.15, 1.0), (0.85, 1.0), (1.0, 1.25), (1.0, 0.8)]:
        vizinho = log_verossimilhanca(d, n, ml.PD * dp, ml.RHO * dr)
        assert vizinho <= melhor + 1e-6


def test_intervalo_perfilado_cobre_a_verdade_e_e_assimetrico() -> None:
    dados = gerar_taxas_vasicek(n_anos=30, semente=11)
    inf, sup, grade, perfil = intervalo_perfil(
        dados["Defaults"].to_numpy(), dados["N"].to_numpy()
    )
    assert inf < RHO_V < sup, f"intervalo [{inf:.4f}, {sup:.4f}] não cobre {RHO_V}"
    assert inf > 0, "correlação não pode ser negativa neste modelo"
    centro = grade[np.argmax(perfil)]
    assert (sup - centro) > (centro - inf), "a verossimilhança em rho é assimétrica"


def test_intervalo_de_rho_e_largo_com_poucos_anos() -> None:
    """A mensagem do capítulo: vinte anos não determinam rho."""
    dados = gerar_taxas_vasicek(n_anos=20, semente=13)
    inf, sup, _, _ = intervalo_perfil(
        dados["Defaults"].to_numpy(), dados["N"].to_numpy()
    )
    assert sup / inf > 2.0, "com 20 anos o intervalo tem de ser largo"


# ---------------------------------------------------------------- capítulo 7


def test_quantil_vasicek_cresce_com_nivel_pd_e_rho() -> None:
    assert quantil_vasicek(PD_V, RHO_V, 0.99) < quantil_vasicek(PD_V, RHO_V, 0.999)
    assert quantil_vasicek(0.01, RHO_V) < quantil_vasicek(0.05, RHO_V)
    assert quantil_vasicek(PD_V, 0.05) < quantil_vasicek(PD_V, 0.30)


def test_quantil_vasicek_no_limite_sem_correlacao() -> None:
    """Sem correlação a taxa é determinística: o quantil colapsa na PD."""
    assert quantil_vasicek(PD_V, 1e-10, 0.999) == pytest.approx(PD_V, abs=1e-4)


def test_taxa_mediana_e_menor_que_a_pd() -> None:
    """A distribuição de Vasicek é assimétrica: a mediana fica abaixo da média.

    No percentil 50% o fator é zero, mas o limiar ainda é dividido por
    sqrt(1-rho), o que empurra a taxa para baixo. Consequência prática: em mais
    da metade dos anos a taxa de default observada fica **abaixo** da PD média,
    e os poucos anos ruins carregam toda a diferença. Quem calibra PD pela
    mediana de uma série curta subestima sistematicamente.
    """
    mediana = quantil_vasicek(PD_V, RHO_V, 0.5)
    assert mediana < PD_V
    esperado = float(
        stats.norm.cdf(stats.norm.ppf(PD_V) / np.sqrt(1 - RHO_V))
    )
    assert mediana == pytest.approx(esperado, rel=1e-12)


def test_simulacao_bate_com_a_formula_analitica() -> None:
    """O fechamento central do capítulo 7.

    Carteira homogênea e granular é o caso em que a fórmula de Vasicek vale, e
    a simulação tem de reproduzi-la.
    """
    carteira = gerar_carteira(n_obrigados=2500, homogenea=True)
    r = simular_perdas(carteira, n_simulacoes=15_000, semente=5)
    escala = carteira["EAD"].sum() * 0.45

    assert r.perda_esperada / escala == pytest.approx(PD_V, rel=0.05)
    for nivel in [0.95, 0.99]:
        simulado = r.var(nivel) / escala
        analitico = quantil_vasicek(PD_V, RHO_V, nivel)
        assert simulado == pytest.approx(analitico, rel=0.06), f"nível {nivel}"


def test_capital_e_o_quantil_menos_a_perda_esperada() -> None:
    carteira = gerar_carteira(n_obrigados=800, homogenea=True)
    r = simular_perdas(carteira, n_simulacoes=8_000, semente=6)
    assert r.capital(0.99) == pytest.approx(r.var(0.99) - r.perda_esperada)
    assert r.expected_shortfall(0.99) >= r.var(0.99)


def test_perda_analitica_e_coerente() -> None:
    r = perda_analitica(PD_V, RHO_V, 0.45, 1_000_000.0, 0.999)
    assert r["perda_estresse"] > r["perda_esperada"] > 0
    assert r["capital"] == pytest.approx(r["perda_estresse"] - r["perda_esperada"])
    assert r["taxa_estresse"] == pytest.approx(quantil_vasicek(PD_V, RHO_V, 0.999))


def test_importance_sampling_nao_enviesa_a_cauda() -> None:
    """Com pesos corretos, IS e MC direto convergem para o mesmo quantil."""
    carteira = gerar_carteira(n_obrigados=1200, homogenea=True)
    referencia = simular_perdas(carteira, 40_000, semente=21).var(0.99)
    com_is = simular_com_is(carteira, 20_000, semente=22).var(0.99)
    assert com_is == pytest.approx(referencia, rel=0.06)


def test_importance_sampling_reduz_a_variancia_do_quantil() -> None:
    """O ganho que justifica a técnica, medido e não suposto."""
    carteira = gerar_carteira(n_obrigados=1000, homogenea=True)
    direto = erro_padrao_quantil(carteira, 5_000, 0.999, n_repeticoes=8, com_is=False)
    com_is = erro_padrao_quantil(carteira, 5_000, 0.999, n_repeticoes=8, com_is=True)
    assert com_is["erro_padrao"] < direto["erro_padrao"]


def test_pesos_do_is_tem_media_um() -> None:
    """A razão de verossimilhanças é não-enviesada por construção."""
    carteira = gerar_carteira(n_obrigados=200, homogenea=True)
    r = simular_com_is(carteira, 20_000, semente=23)
    assert r.pesos is not None
    assert r.pesos.mean() == pytest.approx(1.0, rel=0.05)


def test_concentracao_engorda_a_cauda() -> None:
    """Carteira concentrada tem quantil maior que a granular de mesma PD média."""
    granular = gerar_carteira(n_obrigados=1500, homogenea=True)
    r_gran = simular_perdas(granular, 12_000, semente=31)
    q_gran = r_gran.var(0.999) / r_gran.perda_esperada

    concentrada = gerar_carteira(n_obrigados=150, homogenea=False, semente=32)
    r_conc = simular_perdas(concentrada, 12_000, semente=33)
    q_conc = r_conc.var(0.999) / r_conc.perda_esperada

    assert q_conc > q_gran


def test_contribuicao_por_posicao_soma_a_perda_de_cauda() -> None:
    carteira = gerar_carteira(n_obrigados=300, homogenea=False, semente=41)
    contrib = contribuicao_por_posicao(carteira, n_simulacoes=8_000, nivel=0.99)
    assert len(contrib) == len(carteira)
    assert (contrib["contribuicao_cauda"] >= 0).all()
    # A contribuição de cauda supera a perda esperada — é o que o capital cobre.
    assert contrib["contribuicao_cauda"].sum() > contrib["perda_esperada"].sum()


def test_carteira_heterogenea_e_de_fato_concentrada() -> None:
    carteira = gerar_carteira(n_obrigados=2000, homogenea=False)
    participacao = carteira["EAD"].nlargest(20).sum() / carteira["EAD"].sum()
    assert participacao > 0.05, "1% das posições deveria concentrar parcela relevante"
    assert isinstance(carteira, pd.DataFrame)
