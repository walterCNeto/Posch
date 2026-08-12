"""Testes dos capítulos 4 (previsão de taxas) e 5 (LGD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credrisk.data.generators import (
    COEFS_LGD,
    COEFS_TAXA,
    PREDITORES_CAP04,
    SENIORIDADES,
    gerar_lgd,
    gerar_series_macro,
)
from credrisk.forecast.taxa_default import (
    ajustar_taxa,
    backtest_expandindo,
    logito,
    logito_inverso,
    prever_taxa,
    ruido_binomial_esperado,
)
from credrisk.lgd.fracionaria import (
    ajustar_fracionaria,
    ajustar_ols,
    fracao_fora_do_intervalo,
    lgd_downturn,
    lgd_media_por_ano,
    matriz_lgd,
    prever_lgd,
)

# ---------------------------------------------------------------- capítulo 4


@pytest.fixture(scope="module")
def macro() -> pd.DataFrame:
    return gerar_series_macro()


def test_logito_e_sua_inversa_se_cancelam() -> None:
    p = np.array([0.001, 0.02, 0.5, 0.9])
    np.testing.assert_allclose(logito_inverso(logito(p)), p, rtol=1e-8)


def test_logito_protege_contra_extremos() -> None:
    assert np.isfinite(logito(0.0))
    assert np.isfinite(logito(1.0))


def test_serie_macro_tem_forma_e_nivel_plausiveis(macro: pd.DataFrame) -> None:
    assert list(macro.columns) == [
        "Ano", "SPR", "PRF", "PIB", "N_emissores", "Defaults", "IDR", "taxa_latente",
    ]
    assert (macro["IDR"] > 0).all()
    assert 0.01 < macro["IDR"].mean() < 0.08
    assert macro["IDR"].max() / macro["IDR"].min() > 3, "sem ciclo não há o que prever"


def test_taxa_observada_orbita_a_latente(macro: pd.DataFrame) -> None:
    """A IDR é frequência binomial em torno da taxa verdadeira, não a taxa."""
    assert not np.allclose(macro["IDR"], macro["taxa_latente"])
    assert np.corrcoef(macro["IDR"], macro["taxa_latente"])[0, 1] > 0.9


def test_dgp_da_taxa_e_consistente() -> None:
    """Em muitas réplicas, o OLS no logit recupera os coeficientes plantados."""
    estimativas = []
    for semente in range(300, 360):
        d = gerar_series_macro(semente=semente)
        estimativas.append(ajustar_taxa(d, PREDITORES_CAP04).params)
    E = pd.DataFrame(estimativas)
    E.columns = ["CONST", *PREDITORES_CAP04]
    for nome in E.columns:
        erro = E[nome].mean() - COEFS_TAXA[nome]
        assert abs(erro) < 0.5 * E[nome].std(), f"{nome} enviesado: {erro:+.4f}"


def test_previsao_transformada_fica_no_intervalo_unitario(macro: pd.DataFrame) -> None:
    ajuste = ajustar_taxa(macro, PREDITORES_CAP04, transformar=True)
    previsto = prever_taxa(ajuste, macro, PREDITORES_CAP04, transformar=True)
    assert ((previsto > 0) & (previsto < 1)).all()


def test_newey_west_muda_apenas_os_erros_padrao(macro: pd.DataFrame) -> None:
    simples = ajustar_taxa(macro, PREDITORES_CAP04)
    hac = ajustar_taxa(macro, PREDITORES_CAP04, hac_lags=2)
    np.testing.assert_allclose(simples.params, hac.params, rtol=1e-10)
    assert not np.allclose(simples.bse, hac.bse)


def test_backtest_nao_usa_o_futuro(macro: pd.DataFrame) -> None:
    """A média histórica de cada linha só pode usar anos anteriores."""
    bt = backtest_expandindo(macro, PREDITORES_CAP04, minimo_treino=20)
    ordenado = macro.sort_values("Ano").reset_index(drop=True)
    for _, linha in bt.previsoes.iterrows():
        k = int(ordenado.index[ordenado["Ano"] == linha["Ano"]][0])
        assert linha["media_historica"] == pytest.approx(ordenado["IDR"].iloc[:k].mean())
        assert linha["passeio_aleatorio"] == pytest.approx(ordenado["IDR"].iloc[k - 1])


def test_modelo_bate_os_comparativos_ingenuos(macro: pd.DataFrame) -> None:
    bt = backtest_expandindo(macro, PREDITORES_CAP04)
    assert bt.rmse_modelo < bt.rmse_media
    assert bt.rmse_modelo < bt.rmse_ingenuo
    assert bt.ganho_sobre_media > 0


def test_erro_do_modelo_nao_desce_abaixo_do_ruido_binomial(macro: pd.DataFrame) -> None:
    """Existe um piso irredutível: nem a taxa verdadeira prevê a frequência."""
    bt = backtest_expandindo(macro, PREDITORES_CAP04)
    piso = ruido_binomial_esperado(macro["taxa_latente"], macro["N_emissores"])
    assert bt.rmse_modelo > piso


# ---------------------------------------------------------------- capítulo 5


@pytest.fixture(scope="module")
def lgd() -> pd.DataFrame:
    return gerar_lgd()


def test_lgd_esta_no_intervalo_e_e_bimodal(lgd: pd.DataFrame) -> None:
    assert lgd["LGD"].between(0, 1).all()
    perto_de_zero = (lgd["LGD"] < 0.05).mean()
    perto_de_um = (lgd["LGD"] > 0.95).mean()
    assert perto_de_zero > 0.05, "sem massa perto de zero não há bimodalidade"
    assert perto_de_um > 0.02


def test_senioridade_ordena_a_perda(lgd: pd.DataFrame) -> None:
    medias = lgd.groupby("Senioridade")["LGD"].mean()
    assert medias["Sr. Sec."] < medias["Sr. Unsec."] < medias["Sub."]


def test_matriz_lgd_omite_a_referencia(lgd: pd.DataFrame) -> None:
    X = matriz_lgd(lgd)
    assert "Sr. Sec." not in X.columns
    for nivel in SENIORIDADES:
        if nivel != "Sr. Sec.":
            assert nivel in X.columns
    assert (X["CONST"] == 1.0).all()


def test_ols_preve_fora_do_intervalo_admissivel(lgd: pd.DataFrame) -> None:
    """O contraexemplo que motiva o capítulo."""
    previsto = prever_lgd(ajustar_ols(lgd), lgd)
    assert fracao_fora_do_intervalo(previsto) > 0.0
    assert previsto.min() < 0.0


def test_fracionaria_nunca_sai_do_intervalo(lgd: pd.DataFrame) -> None:
    previsto = prever_lgd(ajustar_fracionaria(lgd), lgd)
    assert fracao_fora_do_intervalo(previsto) == 0.0
    assert (previsto > 0).all() and (previsto < 1).all()


def test_fracionaria_recupera_os_coeficientes_verdadeiros(lgd: pd.DataFrame) -> None:
    """Com o modelo completo — incluindo o fator de ciclo — o estimador acerta."""
    ajuste = ajustar_fracionaria(lgd, numericas=["LEV", "COB", "CICLO"])
    for nome in ["Sr. Unsec.", "Sub.", "COB", "CICLO"]:
        desvio = abs(ajuste.params[nome] - COEFS_LGD[nome]) / ajuste.bse[nome]
        assert desvio < 3.0, f"{nome} desviou {desvio:.1f} erros-padrão"


def test_omitir_o_ciclo_atenua_os_demais_coeficientes(lgd: pd.DataFrame) -> None:
    """Variável omitida em modelo não-linear encolhe os coeficientes que ficam.

    Não é viés de variável omitida no sentido usual — o ciclo é ortogonal às
    covariáveis do devedor. É a diferença entre média condicional e média
    marginal, que só existe porque o link é não-linear.
    """
    sem_ciclo = ajustar_fracionaria(lgd, numericas=["LEV", "COB"])
    com_ciclo = ajustar_fracionaria(lgd, numericas=["LEV", "COB", "CICLO"])
    assert abs(sem_ciclo.params["COB"]) < abs(com_ciclo.params["COB"])
    verdadeiro = COEFS_LGD["COB"]
    assert abs(sem_ciclo.params["COB"] - verdadeiro) > abs(
        com_ciclo.params["COB"] - verdadeiro
    )


def test_garantia_reduz_a_perda(lgd: pd.DataFrame) -> None:
    ajuste = ajustar_fracionaria(lgd)
    assert ajuste.params["COB"] < 0
    assert ajuste.pvalues["COB"] < 0.01


def test_lgd_media_por_ano_agrega_certo(lgd: pd.DataFrame) -> None:
    por_ano = lgd_media_por_ano(lgd)
    assert len(por_ano) == lgd["Ano"].nunique()
    assert por_ano["n"].sum() == len(lgd)


def test_existe_correlacao_entre_severidade_e_frequencia(lgd: pd.DataFrame) -> None:
    """A premissa que obriga o LGD de downturn."""
    por_ano = lgd_media_por_ano(lgd)
    rho = np.corrcoef(por_ano["LGD_media"], por_ano["taxa_default"])[0, 1]
    assert rho > 0.4, f"correlação fraca demais: {rho:.3f}"


def test_lgd_downturn_supera_a_media(lgd: pd.DataFrame) -> None:
    r = lgd_downturn(lgd)
    assert r["media_downturn"] > r["media_geral"]
    assert r["razao"] > 1.0
    assert r["anos_downturn"] >= 1
