"""Testes do capítulo 1.

Onde existe resposta fechada, testamos contra ela: o Firth tem de bater com a
MLE em amostra grande, o AUC tem de bater com a fórmula de Mann-Whitney e a AR
tem de satisfazer ``AR = 2*AUC - 1`` por construção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from credrisk.data.generators import (
    COEFS_VERDADEIROS,
    PREDITORES_CAP01,
    gerar_painel_scoring,
)
from credrisk.scoring.logit import (
    ajustar,
    ajustar_firth,
    auc,
    comparar_com_verdadeiro,
    matriz_de_desenho,
    prever_pd,
    razao_de_acuracia,
)


@pytest.fixture(scope="module")
def painel() -> pd.DataFrame:
    return gerar_painel_scoring()


def test_matriz_de_desenho_tem_constante_nomeada(painel: pd.DataFrame) -> None:
    X = matriz_de_desenho(painel, PREDITORES_CAP01)
    assert list(X.columns) == ["CONST", *PREDITORES_CAP01]
    assert (X["CONST"] == 1.0).all()


def test_ajuste_bate_com_statsmodels_direto(painel: pd.DataFrame) -> None:
    nosso = ajustar(painel, PREDITORES_CAP01)
    X = sm.add_constant(painel[PREDITORES_CAP01])
    deles = sm.Logit(painel["Default"], X).fit(disp=0)
    np.testing.assert_allclose(nosso.params.to_numpy(), deles.params.to_numpy(), rtol=1e-8)


def test_cluster_nao_muda_coeficientes_mas_muda_erros(painel: pd.DataFrame) -> None:
    ingenuo = ajustar(painel, PREDITORES_CAP01)
    agrupado = ajustar(painel, PREDITORES_CAP01, cluster="ID")
    np.testing.assert_allclose(
        ingenuo.params.to_numpy(), agrupado.params.to_numpy(), rtol=1e-6
    )
    assert not np.allclose(ingenuo.bse.to_numpy(), agrupado.bse.to_numpy())


def test_firth_converge_e_tem_a_forma_certa(painel: pd.DataFrame) -> None:
    r = ajustar_firth(painel, PREDITORES_CAP01)
    assert r.convergiu, f"Firth não convergiu em {r.iteracoes} iterações"
    assert list(r.params.index) == ["CONST", *PREDITORES_CAP01]
    assert (r.bse > 0).all()
    assert r.n_eventos == int(painel["Default"].sum())


def test_firth_encolhe_para_a_mle_em_amostra_grande() -> None:
    """A penalização é O(1) e a verossimilhança é O(n): em amostra grande, somem."""
    grande = gerar_painel_scoring(n_empresas=6000, semente=11)
    mle = ajustar(grande, PREDITORES_CAP01)
    firth = ajustar_firth(grande, PREDITORES_CAP01)
    np.testing.assert_allclose(
        firth.params.to_numpy(), mle.params.to_numpy(), rtol=0.06, atol=0.06
    )


def test_firth_resolve_separacao_completa() -> None:
    """Sob separação perfeita a MLE diverge; o Firth continua finito."""
    dados = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "Default": [0, 0, 0, 1, 1, 1]}
    )
    firth = ajustar_firth(dados, ["x"])
    assert firth.convergiu
    assert np.isfinite(firth.params).all()
    assert firth.params["x"] > 0


def test_prever_pd_esta_no_intervalo_unitario(painel: pd.DataFrame) -> None:
    r = ajustar(painel, PREDITORES_CAP01)
    pd_prevista = prever_pd(r, painel, PREDITORES_CAP01)
    assert ((pd_prevista > 0) & (pd_prevista < 1)).all()
    # No logit com constante, a PD média ajustada iguala a taxa observada.
    assert pd_prevista.mean() == pytest.approx(painel["Default"].mean(), abs=1e-6)


def test_auc_contra_calculo_direto() -> None:
    escore = np.array([0.1, 0.4, 0.35, 0.8])
    alvo = np.array([0, 0, 1, 1])
    # Pares (mau, bom): (0.35,0.1)=1, (0.35,0.4)=0, (0.8,0.1)=1, (0.8,0.4)=1 -> 3/4
    assert auc(escore, alvo) == pytest.approx(0.75)


def test_auc_de_escore_aleatorio_fica_perto_de_meio() -> None:
    rng = np.random.default_rng(3)
    escore = rng.random(20000)
    alvo = rng.integers(0, 2, 20000)
    assert abs(auc(escore, alvo) - 0.5) < 0.02


def test_identidade_ar_auc(painel: pd.DataFrame) -> None:
    r = ajustar(painel, PREDITORES_CAP01)
    p = prever_pd(r, painel, PREDITORES_CAP01)
    alvo = painel["Default"].to_numpy()
    assert razao_de_acuracia(p, alvo) == pytest.approx(2 * auc(p, alvo) - 1)


def test_auc_exige_as_duas_classes() -> None:
    with pytest.raises(ValueError):
        auc(np.array([0.1, 0.2]), np.array([0, 0]))


def test_comparacao_com_verdadeiro(painel: pd.DataFrame) -> None:
    r = ajustar(painel, PREDITORES_CAP01)
    tabela = comparar_com_verdadeiro(r, COEFS_VERDADEIROS)
    assert list(tabela.index) == ["CONST", *PREDITORES_CAP01]
    assert tabela["ic95_cobre"].dtype == bool
    # Reprodução manual do desvio padronizado de uma linha.
    linha = tabela.loc["ME/TL"]
    esperado = (linha["estimado"] - linha["verdadeiro"]) / linha["ep"]
    assert linha["desvio_em_ep"] == pytest.approx(esperado)
