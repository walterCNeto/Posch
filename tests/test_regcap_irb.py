"""Testes do capítulo 12 (Basileia e ratings internos).

O teste mais importante do arquivo é o que amarra a fórmula regulatória ao
modelo do capítulo 7: removido o ajuste de maturidade, o requerimento IRB é
exatamente o quantil de Vasicek menos a perda esperada. Se essa identidade
quebrar, ou a implementação da fórmula ou a do modelo de carteira está errada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credrisk.portfolio.montecarlo import quantil_vasicek, simular_perdas
from credrisk.regcap.irb import (
    FATOR_RWA,
    ajuste_maturidade,
    correlacao_prescrita,
    curva_de_capital,
    discretizar_em_grades,
    efeito_da_granularidade,
    perda_esperada_regulatoria,
    requerimento_capital,
    resumo_carteira,
    rwa,
)

LGD_PADRAO = 0.45


def test_correlacao_corporate_decresce_com_pd() -> None:
    p = np.array([0.0003, 0.001, 0.01, 0.05, 0.20])
    R = correlacao_prescrita(p)
    assert np.all(np.diff(R) < 0)


def test_correlacao_corporate_respeita_os_limites() -> None:
    """Interpola entre 12% e 24%, sem sair da faixa."""
    p = np.geomspace(1e-6, 0.99, 200)
    R = correlacao_prescrita(p)
    assert R.max() <= 0.24 + 1e-9
    assert R.min() >= 0.12 - 1e-9


def test_correlacoes_de_varejo() -> None:
    assert float(correlacao_prescrita(0.02, "varejo_hipotecario")) == pytest.approx(0.15)
    assert float(correlacao_prescrita(0.02, "varejo_rotativo")) == pytest.approx(0.04)
    outros = correlacao_prescrita(np.array([0.001, 0.10]), "varejo_outros")
    assert outros[0] > outros[1]
    assert 0.03 - 1e-9 <= outros.min() and outros.max() <= 0.16 + 1e-9


def test_classe_desconhecida_e_rejeitada() -> None:
    with pytest.raises(ValueError):
        correlacao_prescrita(0.01, "classe_inventada")


def test_ajuste_de_porte_reduz_a_correlacao() -> None:
    grande = float(correlacao_prescrita(0.01, "corporate"))
    media = float(correlacao_prescrita(0.01, "corporate", faturamento_milhoes=25.0))
    pequena = float(correlacao_prescrita(0.01, "corporate", faturamento_milhoes=5.0))
    assert pequena < media < grande
    assert grande - pequena == pytest.approx(0.04, abs=1e-9)


def test_ajuste_de_maturidade_vale_um_em_um_ano() -> None:
    """M = 1 é o caso base: o multiplicador colapsa para 1 por construção."""
    for p in [0.001, 0.01, 0.10]:
        assert float(ajuste_maturidade(p, 1.0)) == pytest.approx(1.0, rel=1e-12)


def test_ajuste_de_maturidade_cresce_com_o_prazo() -> None:
    for p in [0.001, 0.01, 0.10]:
        a = [float(ajuste_maturidade(p, m)) for m in (1.0, 2.5, 5.0)]
        assert a[0] < a[1] < a[2]


def test_ajuste_de_maturidade_pesa_mais_em_pd_baixa() -> None:
    """Um crédito bom tem mais espaço para se deteriorar sem quebrar."""
    bom = float(ajuste_maturidade(0.001, 5.0))
    ruim = float(ajuste_maturidade(0.10, 5.0))
    assert bom > ruim


def test_capital_cresce_com_pd_e_com_lgd() -> None:
    assert float(requerimento_capital(0.05, LGD_PADRAO)) > float(
        requerimento_capital(0.01, LGD_PADRAO)
    )
    assert float(requerimento_capital(0.01, 0.90)) > float(
        requerimento_capital(0.01, 0.45)
    )


def test_capital_e_proporcional_a_lgd() -> None:
    a = float(requerimento_capital(0.02, 0.30))
    b = float(requerimento_capital(0.02, 0.60))
    assert b == pytest.approx(2 * a, rel=1e-10)


def test_irb_sem_maturidade_e_o_asrf_do_capitulo_sete() -> None:
    """A identidade que amarra o capítulo 12 ao capítulo 7."""
    for p in [0.0005, 0.001, 0.01, 0.05, 0.20]:
        R = float(correlacao_prescrita(p))
        asrf = (quantil_vasicek(p, R, 0.999) - p) * LGD_PADRAO
        irb = float(requerimento_capital(p, LGD_PADRAO)) / float(
            ajuste_maturidade(p, 2.5)
        )
        assert irb == pytest.approx(asrf, rel=1e-12), f"PD = {p}"


def test_capital_nao_inclui_a_perda_esperada() -> None:
    """Capital cobre perda inesperada; provisão cobre a esperada."""
    p, lgd = 0.03, LGD_PADRAO
    R = float(correlacao_prescrita(p))
    quantil_total = quantil_vasicek(p, R, 0.999) * lgd
    k_sem_maturidade = float(requerimento_capital(p, lgd)) / float(
        ajuste_maturidade(p, 2.5)
    )
    assert k_sem_maturidade == pytest.approx(quantil_total - p * lgd, rel=1e-12)


def test_risk_weight_bate_com_a_tabela_publicada() -> None:
    """PD 1%, LGD 45%, M 2,5 tem ponderação em torno de 90-95%."""
    k = float(requerimento_capital(0.01, 0.45, maturidade=2.5))
    assert 0.85 < k * FATOR_RWA < 1.00


def test_rwa_e_capital_vezes_doze_e_meio() -> None:
    ead = 1_000_000.0
    k = float(requerimento_capital(0.02, LGD_PADRAO))
    assert float(rwa(ead, 0.02, LGD_PADRAO)) == pytest.approx(ead * k * FATOR_RWA)


def test_perda_esperada_regulatoria_e_o_produto_simples() -> None:
    assert float(perda_esperada_regulatoria(1000.0, 0.02, 0.45)) == pytest.approx(9.0)


def test_invariancia_a_carteira() -> None:
    """O requerimento de uma exposição não depende das demais.

    É a propriedade que permite somar posição a posição — e a razão pela qual a
    fórmula ignora concentração.
    """
    a = pd.DataFrame({"EAD": [100.0], "PD": [0.02], "LGD": [0.45]})
    b = pd.DataFrame(
        {"EAD": [100.0, 5000.0], "PD": [0.02, 0.15], "LGD": [0.45, 0.60]}
    )
    k_sozinha = float(
        requerimento_capital(a["PD"].to_numpy(), a["LGD"].to_numpy())[0]
    )
    k_acompanhada = float(
        requerimento_capital(b["PD"].to_numpy(), b["LGD"].to_numpy())[0]
    )
    assert k_sozinha == pytest.approx(k_acompanhada, rel=1e-12)


def test_resumo_de_carteira_agrega_certo() -> None:
    carteira = pd.DataFrame(
        {
            "EAD": [1000.0, 2000.0, 500.0],
            "PD": [0.005, 0.02, 0.10],
            "LGD": [0.45, 0.45, 0.60],
        }
    )
    r = resumo_carteira(carteira)
    assert r["EAD total"] == pytest.approx(3500.0)
    assert r["RWA"] == pytest.approx(r["capital exigido"] * FATOR_RWA)
    assert r["capital / EAD"] == pytest.approx(r["capital exigido"] / 3500.0)


def test_curva_de_capital_cobre_as_classes() -> None:
    curva = curva_de_capital(pds=np.array([0.001, 0.01, 0.10]))
    assert "corporate" in curva.columns
    # Varejo rotativo tem a menor correlação e portanto o menor capital.
    assert (curva["varejo_rotativo"] < curva["corporate"]).all()


def test_discretizar_preserva_o_tamanho_e_reduz_valores_distintos() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.001, 0.20, 500)
    pd_grade, indice = discretizar_em_grades(p, 7)
    assert len(pd_grade) == len(p)
    assert len(np.unique(pd_grade)) <= 7
    assert indice.min() >= 0 and indice.max() <= 6


def test_discretizar_rejeita_zero_grades() -> None:
    with pytest.raises(ValueError):
        discretizar_em_grades(np.array([0.01, 0.02]), 0)


def test_grade_unica_atribui_a_media_a_todos() -> None:
    p = np.array([0.01, 0.05, 0.09])
    pd_grade, _ = discretizar_em_grades(p, 1)
    assert np.allclose(pd_grade, p.mean())


def test_menos_grades_exige_mais_capital() -> None:
    """A fórmula é côncava em PD: agrupar penaliza o banco.

    O sinal importa — é o que dá incentivo a manter um sistema de rating
    granular.
    """
    rng = np.random.default_rng(2)
    n = 800
    carteira = pd.DataFrame(
        {
            "EAD": rng.lognormal(6.0, 1.0, n),
            "PD": np.clip(rng.lognormal(np.log(0.02), 0.9, n), 1e-4, 0.35),
            "LGD": np.full(n, LGD_PADRAO),
        }
    )
    tabela = efeito_da_granularidade(carteira, grades=(1, 5, 25))
    capital = tabela.set_index("grades")["capital"]
    assert capital[1] > capital[5] > capital[25]
    assert capital[1] / capital[25] > 1.02


def test_irb_reproduz_o_capital_economico_de_carteira_granular() -> None:
    """Com a mesma correlação, os dois coincidem quando a carteira é granular."""
    pd_i, lgd = 0.015, LGD_PADRAO
    R = float(correlacao_prescrita(pd_i))
    n = 4000
    carteira = pd.DataFrame(
        {
            "ID": np.arange(n),
            "EAD": np.full(n, 1000.0),
            "PD": np.full(n, pd_i),
            "LGD": np.full(n, lgd),
            "RHO": np.full(n, R),
        }
    )
    economico = simular_perdas(carteira, 25_000, semente=3).capital(0.999)
    regulatorio = float(
        (carteira["EAD"].to_numpy() * requerimento_capital(
            carteira["PD"].to_numpy(), carteira["LGD"].to_numpy(), maturidade=1.0
        )).sum()
    )
    assert regulatorio == pytest.approx(economico, rel=0.12)


def test_irb_subestima_carteira_concentrada() -> None:
    """A limitação estrutural: invariância à carteira ignora concentração."""
    pd_i, lgd = 0.015, LGD_PADRAO
    R = float(correlacao_prescrita(pd_i))
    rng = np.random.default_rng(5)
    n = 400

    def montar(ead: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ID": np.arange(n),
                "EAD": ead,
                "PD": np.full(n, pd_i),
                "LGD": np.full(n, lgd),
                "RHO": np.full(n, R),
            }
        )

    concentrada = rng.lognormal(0, 1.6, n)
    concentrada = concentrada / concentrada.sum() * (n * 1000.0)

    economico = simular_perdas(montar(concentrada), 25_000, semente=7).capital(0.999)
    regulatorio = float(
        (concentrada * requerimento_capital(
            np.full(n, pd_i), np.full(n, lgd), maturidade=1.0
        )).sum()
    )
    assert regulatorio < economico
