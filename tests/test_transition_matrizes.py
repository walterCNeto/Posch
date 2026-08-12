"""Testes do capítulo 3.

A cadeia de Markov dá muitas identidades fechadas para testar: linhas de
probabilidade somam 1, linhas da geradora somam 0, ``expm(Q·0) = I``, e a
propriedade de semigrupo ``P(s)P(t) = P(s+t)``. Além delas, o teste que importa
para o curso: o estimador de duração recupera a geradora verdadeira.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from credrisk.data.generators import (
    RATINGS,
    gerador_verdadeiro,
    gerar_historico_ratings,
)
from credrisk.transition.matrizes import (
    bootstrap_coorte,
    exposicao_e_transicoes,
    gerador_duracao,
    matriz_coorte,
    matriz_do_gerador,
    pd_por_horizonte,
    rating_na_data,
)


@pytest.fixture(scope="module")
def historico():
    return gerar_historico_ratings()


def test_geradora_verdadeira_e_valida() -> None:
    Q = gerador_verdadeiro()
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)
    fora = ~np.eye(len(RATINGS), dtype=bool)
    assert (Q[fora] >= 0).all(), "intensidades fora da diagonal não podem ser negativas"
    assert (np.diag(Q) <= 0).all()
    assert np.allclose(Q[-1], 0.0), "D tem de ser absorvente"


def test_historico_tem_a_forma_esperada(historico) -> None:
    assert list(historico.columns) == ["ID", "Data", "Rating", "Estado"]
    assert historico["Estado"].between(0, len(RATINGS) - 1).all()
    assert not historico.isna().any().any()
    # Uma vez em D, nada acontece depois.
    for _, g in historico.groupby("ID"):
        estados = g["Estado"].to_numpy()
        if (estados == len(RATINGS) - 1).any():
            assert estados[-1] == len(RATINGS) - 1
            assert (estados == len(RATINGS) - 1).sum() == 1


def test_rating_na_data_respeita_a_entrada(historico) -> None:
    datas = np.array([0.0, 5.0, 10.0])
    painel = rating_na_data(historico, datas)
    assert painel.shape[0] == historico["ID"].nunique()
    # Em t=0 muita empresa ainda não entrou; em t=10 quase todas já entraram.
    assert painel.iloc[:, 0].isna().sum() > painel.iloc[:, -1].isna().sum()


def test_matriz_coorte_e_estocastica(historico) -> None:
    P = matriz_coorte(historico)
    np.testing.assert_allclose(P.to_numpy().sum(axis=1), 1.0, atol=1e-12)
    assert (P.to_numpy() >= 0).all()
    assert P.loc["D", "D"] == pytest.approx(1.0)


def test_geradora_estimada_e_valida(historico) -> None:
    Q = gerador_duracao(historico).to_numpy()
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)
    fora = ~np.eye(len(RATINGS), dtype=bool)
    assert (Q[fora] >= 0).all()


def test_exponencial_de_matriz_em_t_zero_e_identidade() -> None:
    P = matriz_do_gerador(gerador_verdadeiro(), 0.0).to_numpy()
    np.testing.assert_allclose(P, np.eye(len(RATINGS)), atol=1e-12)


def test_propriedade_de_semigrupo() -> None:
    """P(2) tem de ser P(1) @ P(1) — a cadeia é consistente entre horizontes."""
    Q = gerador_verdadeiro()
    P1 = matriz_do_gerador(Q, 1.0).to_numpy()
    P2 = matriz_do_gerador(Q, 2.0).to_numpy()
    np.testing.assert_allclose(P1 @ P1, P2, atol=1e-10)


def test_matriz_do_gerador_bate_com_expm_direto() -> None:
    Q = gerador_verdadeiro()
    np.testing.assert_allclose(
        matriz_do_gerador(Q, 3.0).to_numpy(), linalg.expm(Q * 3.0), atol=1e-10
    )


def test_duracao_recupera_a_geradora_verdadeira() -> None:
    """O teste central: com amostra grande, o estimador acha as intensidades."""
    grande = gerar_historico_ratings(n_empresas=20000, anos=25.0, semente=5)
    Q_est = gerador_duracao(grande, anos=25.0).to_numpy()
    Q_verd = gerador_verdadeiro()

    # Compara apenas as intensidades com massa suficiente para serem estimáveis.
    mascara = Q_verd > 1e-3
    erro_rel = np.abs(Q_est[mascara] - Q_verd[mascara]) / Q_verd[mascara]
    assert erro_rel.max() < 0.25, f"pior erro relativo: {erro_rel.max():.3f}"


def test_coorte_zera_transicoes_raras_e_duracao_nao(historico) -> None:
    """O resultado que motiva o capítulo inteiro."""
    P_coorte = matriz_coorte(historico)
    P_duracao = matriz_do_gerador(gerador_duracao(historico), 1.0)

    assert P_coorte.loc["AAA", "D"] == 0.0
    assert P_duracao.loc["AAA", "D"] > 0.0
    # A geradora verdadeira também é positiva ali.
    assert matriz_do_gerador(gerador_verdadeiro(), 1.0).loc["AAA", "D"] > 0.0


def test_duracao_reproduz_a_esparsidade_verdadeira(historico) -> None:
    P_coorte = matriz_coorte(historico).to_numpy()
    P_duracao = matriz_do_gerador(gerador_duracao(historico), 1.0).to_numpy()
    P_verd = matriz_do_gerador(gerador_verdadeiro(), 1.0).to_numpy()
    assert (P_duracao == 0).sum() == (P_verd == 0).sum()
    assert (P_coorte == 0).sum() > (P_verd == 0).sum()


def test_exposicao_nao_conta_tempo_apos_default(historico) -> None:
    tempo, N = exposicao_e_transicoes(historico)
    assert tempo[-1] == 0.0, "default é absorvente e não acumula exposição"
    assert N.sum() > 0


def test_pd_cumulativa_cresce_com_horizonte_e_com_risco() -> None:
    tabela = pd_por_horizonte(gerador_verdadeiro(), [1.0, 3.0, 5.0])
    for rating in ["AA", "BBB", "B"]:
        linha = tabela.loc[rating].to_numpy()
        assert np.all(np.diff(linha) > 0), f"{rating} não é monótona no horizonte"
    assert tabela.loc["CCC", "1a"] > tabela.loc["BBB", "1a"] > tabela.loc["AA", "1a"]


def test_bootstrap_produz_dispersao(historico) -> None:
    amostras = bootstrap_coorte(historico, n_reamostras=25, semente=1)
    assert amostras.shape == (25, len(RATINGS), len(RATINGS))
    np.testing.assert_allclose(amostras.sum(axis=2), 1.0, atol=1e-10)
    # A célula BBB->D tem de variar entre reamostras.
    i, j = RATINGS.index("BBB"), RATINGS.index("D")
    assert amostras[:, i, j].std() > 0
