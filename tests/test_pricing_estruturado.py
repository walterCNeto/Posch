"""Testes dos capítulos 10 (CDS) e 11 (CDOs).

O capítulo 10 é rico em identidades fechadas — bootstrap tem de reproduzir os
spreads de entrada, e a aproximação de mercado tem de convergir para a fórmula
exata no limite contínuo. O capítulo 11 tem o fechamento entre a fórmula LHP e
a simulação, além do comportamento qualitativo das tranches em relação à
correlação, que é o conteúdo econômico do produto.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from credrisk.pricing.cds import (
    bootstrap_hazards,
    desconto,
    pd_acumulada,
    pd_fisica_de_neutra,
    perna_premio,
    perna_protecao,
    premio_implicito,
    sobrevivencia,
    spread_aproximado,
    spread_justo,
)
from credrisk.structured.cdo import (
    correlacao_implicita,
    el_tranche_lhp,
    el_tranche_mc,
    estrutura_padrao,
    perda_tranche,
    sensibilidade_a_correlacao,
    simular_perdas_carteira,
    tabela_tranches,
    taxa_condicional,
)

TAXA = 0.04
REC = 0.40

# --------------------------------------------------------------- capítulo 10


def test_desconto_e_monotono_e_vale_um_em_zero() -> None:
    assert float(desconto(0.0, TAXA)) == pytest.approx(1.0)
    t = np.array([0.5, 1.0, 5.0])
    assert np.all(np.diff(desconto(t, TAXA)) < 0)


def test_sobrevivencia_comeca_em_um_e_decresce() -> None:
    h, v = np.array([0.01, 0.02]), np.array([1.0, 5.0])
    assert float(sobrevivencia(0.0, h, v)[0]) == pytest.approx(1.0)
    t = np.array([0.5, 1.0, 3.0, 5.0])
    assert np.all(np.diff(sobrevivencia(t, h, v)) < 0)


def test_sobrevivencia_com_hazard_constante_tem_forma_fechada() -> None:
    h, v = np.array([0.03]), np.array([10.0])
    t = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(sobrevivencia(t, h, v), np.exp(-0.03 * t), rtol=1e-12)


def test_sobrevivencia_por_trechos_acumula_certo() -> None:
    """Dois anos a 1% e depois três anos a 5%."""
    h, v = np.array([0.01, 0.05]), np.array([2.0, 5.0])
    esperado = np.exp(-(0.01 * 2 + 0.05 * 3))
    assert float(sobrevivencia(5.0, h, v)[0]) == pytest.approx(esperado, rel=1e-12)


def test_pd_acumulada_complementa_a_sobrevivencia() -> None:
    h, v = np.array([0.02, 0.03]), np.array([3.0, 7.0])
    t = np.array([1.0, 5.0])
    np.testing.assert_allclose(
        pd_acumulada(t, h, v) + sobrevivencia(t, h, v), 1.0, rtol=1e-12
    )


def test_pernas_sao_positivas_e_crescem_com_o_prazo() -> None:
    h, v = np.array([0.02]), np.array([10.0])
    assert perna_premio(1.0, h, v, TAXA) < perna_premio(5.0, h, v, TAXA)
    assert perna_protecao(1.0, h, v, TAXA, REC) < perna_protecao(5.0, h, v, TAXA, REC)


def test_spread_cresce_com_hazard_e_cai_com_recuperacao() -> None:
    v = np.array([5.0])
    baixo = spread_justo(5.0, np.array([0.01]), v, TAXA, REC)
    alto = spread_justo(5.0, np.array([0.05]), v, TAXA, REC)
    assert alto > baixo
    mais_recuperacao = spread_justo(5.0, np.array([0.05]), v, TAXA, 0.70)
    assert mais_recuperacao < alto


def test_bootstrap_recupera_hazards_plantados() -> None:
    """O fechamento central do capítulo 10."""
    prazos = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
    verdadeiros = np.array([0.010, 0.014, 0.018, 0.020, 0.021])
    spreads = np.array(
        [
            spread_justo(p, verdadeiros[: i + 1], prazos[: i + 1], TAXA, REC)
            for i, p in enumerate(prazos)
        ]
    )
    curva = bootstrap_hazards(prazos, spreads, TAXA, REC)
    np.testing.assert_allclose(curva.hazards, verdadeiros, atol=1e-10)


def test_bootstrap_reprecifica_os_instrumentos_de_entrada() -> None:
    prazos = np.array([1.0, 3.0, 5.0])
    spreads = np.array([0.0060, 0.0090, 0.0120])
    curva = bootstrap_hazards(prazos, spreads, TAXA, REC)
    for prazo, alvo in zip(prazos, spreads, strict=True):
        assert curva.spread(prazo) == pytest.approx(alvo, abs=1e-10)


def test_bootstrap_exige_prazos_ordenados() -> None:
    with pytest.raises(ValueError):
        bootstrap_hazards(np.array([5.0, 1.0]), np.array([0.01, 0.01]), TAXA)
    with pytest.raises(ValueError):
        bootstrap_hazards(np.array([1.0, 5.0]), np.array([0.01]), TAXA)


def test_curva_ascendente_gera_spreads_ascendentes() -> None:
    prazos = np.array([1.0, 5.0, 10.0])
    h = np.array([0.005, 0.015, 0.030])
    spreads = [
        spread_justo(p, h[: i + 1], prazos[: i + 1], TAXA, REC)
        for i, p in enumerate(prazos)
    ]
    assert spreads[0] < spreads[1] < spreads[2]


def test_aproximacao_converge_no_limite_de_pagamento_continuo() -> None:
    """s ≈ h(1-R) é exata em tempo contínuo com hazard constante."""
    h = 0.10
    exato = spread_justo(5.0, np.array([h]), np.array([5.0]), TAXA, REC, por_ano=365)
    assert exato == pytest.approx(spread_aproximado(h, REC), rel=1e-3)


def test_aproximacao_erra_com_curva_inclinada() -> None:
    """O que degrada a aproximação é a inclinação, não a magnitude do spread."""
    prazos = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
    h = np.array([0.004, 0.010, 0.020, 0.030, 0.045])
    exato = spread_justo(10.0, h, prazos, TAXA, REC)
    medio = float(np.average(h, weights=np.diff(np.concatenate([[0.0], prazos]))))
    assert abs(spread_aproximado(medio, REC) / exato - 1) > 0.05


def test_pd_neutra_supera_a_fisica_com_premio_positivo() -> None:
    neutra = 0.05
    fisica = pd_fisica_de_neutra(neutra, premio_de_risco=0.5, prazo=1.0)
    assert fisica < neutra


def test_premio_implicito_inverte_a_conversao() -> None:
    neutra, premio = 0.06, 0.45
    fisica = pd_fisica_de_neutra(neutra, premio, prazo=1.0)
    assert premio_implicito(neutra, fisica, 1.0) == pytest.approx(premio, rel=1e-8)


def test_premio_zero_iguala_as_duas_medidas() -> None:
    assert pd_fisica_de_neutra(0.04, 0.0) == pytest.approx(0.04, rel=1e-10)


# --------------------------------------------------------------- capítulo 11


def test_perda_tranche_e_zero_abaixo_do_attach() -> None:
    assert float(perda_tranche(np.array([0.02]), 0.03, 0.07)[0]) == 0.0


def test_perda_tranche_e_um_acima_do_detach() -> None:
    assert float(perda_tranche(np.array([0.20]), 0.03, 0.07)[0]) == pytest.approx(1.0)


def test_perda_tranche_interpola_linearmente_no_meio() -> None:
    # Perda de 5% numa tranche 3%-7%: consumiu 2 dos 4 pontos, metade.
    assert float(perda_tranche(np.array([0.05]), 0.03, 0.07)[0]) == pytest.approx(0.5)


def test_tranches_somam_a_perda_da_carteira() -> None:
    """Conservação: a estrutura redistribui a perda, não a cria nem destrói."""
    perda = np.array([0.045])
    estrutura = estrutura_padrao()
    total = 0.0
    for a, d in zip(estrutura["attach"], estrutura["detach"], strict=True):
        total += float(perda_tranche(perda, a, d)[0]) * (d - a)
    assert total == pytest.approx(min(float(perda[0]), 0.30), abs=1e-12)


def test_perda_tranche_rejeita_faixa_invalida() -> None:
    with pytest.raises(ValueError):
        perda_tranche(np.array([0.05]), 0.07, 0.03)


def test_lhp_bate_com_monte_carlo_nas_tranches_baixas() -> None:
    """O fechamento do capítulo 11, onde a simulação tem cenários suficientes."""
    pd_inc, rho, lgd = 0.015, 0.12, 0.60
    perdas = simular_perdas_carteira(
        2000, pd_inc, rho, lgd, n_simulacoes=40_000, semente=1
    )
    for a, d in [(0.0, 0.03), (0.03, 0.07)]:
        mc = el_tranche_mc(perdas, a, d)
        lhp = el_tranche_lhp(a, d, pd_inc, rho, lgd)
        assert mc == pytest.approx(lhp, rel=0.06), f"tranche [{a}, {d}]"


def test_el_das_tranches_decresce_na_senioridade() -> None:
    tabela = tabela_tranches(0.015, 0.12, 0.60)
    el = tabela["EL"].to_numpy()
    assert np.all(np.diff(el) < 0)


def test_correlacao_reduz_a_perda_da_equity() -> None:
    """Resultado contraintuitivo e central: o júnior gosta de correlação."""
    s = sensibilidade_a_correlacao(0.0, 0.03, 0.015, 0.60,
                                   rhos=np.array([0.05, 0.20, 0.50]))
    assert np.all(np.diff(s["EL"].to_numpy()) < 0)


def test_correlacao_aumenta_a_perda_da_senior() -> None:
    s = sensibilidade_a_correlacao(0.10, 0.15, 0.015, 0.60,
                                   rhos=np.array([0.05, 0.20, 0.50]))
    assert np.all(np.diff(s["EL"].to_numpy()) > 0)


def test_senior_e_muito_mais_sensivel_que_a_equity() -> None:
    """A mensagem do capítulo, em números.

    A mesma variação de correlação move a equity por dezenas de por cento e a
    sênior por ordens de magnitude.
    """
    baixo, alto = 0.056, 0.177  # o IC estimado no capítulo 6
    eq = [el_tranche_lhp(0.0, 0.03, 0.015, r, 0.60) for r in (baixo, alto)]
    sr = [el_tranche_lhp(0.10, 0.15, 0.015, r, 0.60) for r in (baixo, alto)]
    razao_equity = eq[1] / eq[0]
    razao_senior = sr[1] / max(sr[0], 1e-300)
    assert 0.5 < razao_equity < 2.0
    assert razao_senior > 100


def test_correlacao_implicita_recupera_a_usada() -> None:
    alvo = el_tranche_lhp(0.03, 0.07, 0.015, 0.18, 0.60)
    assert correlacao_implicita(alvo, 0.03, 0.07, 0.015, 0.60) == pytest.approx(
        0.18, abs=1e-4
    )


def test_equity_tem_perda_maxima_inatingivel_por_correlacao_alguma() -> None:
    """Preço que exige EL acima do teto do modelo não tem correlação implícita.

    A perda esperada da equity é decrescente em rho, então existe um máximo em
    rho próximo de zero. Um preço de mercado acima dele rejeita o modelo — é o
    fenômeno do sorriso de correlação.
    """
    teto = el_tranche_lhp(0.0, 0.03, 0.015, 1e-4, 0.60)
    assert np.isnan(correlacao_implicita(teto * 1.5, 0.0, 0.03, 0.015, 0.60))


def test_taxa_condicional_integra_para_a_pd() -> None:
    x = np.linspace(-8, 8, 4001)
    peso = stats.norm.pdf(x)
    media = np.trapezoid(taxa_condicional(x, 0.015, 0.12) * peso, x)
    assert media == pytest.approx(0.015, rel=1e-4)
