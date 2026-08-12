"""Testes do processo gerador de dados do capítulo 1.

A base é sintética, então podemos testar mais do que forma e ausência de nulo:
testamos se o DGP tem as propriedades que o capítulo vai explorar — taxa de
default plausível, painel desbalanceado por saída em default, correlação serial
dentro da empresa e variação da taxa de default entre anos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credrisk.data import registry
from credrisk.data.generators import (
    COEFS_VERDADEIROS,
    PREDITORES_CAP01,
    gerar_painel_scoring,
)


@pytest.fixture(scope="module")
def painel() -> pd.DataFrame:
    return gerar_painel_scoring()


def test_colunas_e_tipos(painel: pd.DataFrame) -> None:
    esperadas = ["ID", "Ano", "Default", *PREDITORES_CAP01]
    assert list(painel.columns) == esperadas
    assert painel["Default"].isin([0, 1]).all()
    assert not painel.isna().any().any()


def test_tamanho_do_painel(painel: pd.DataFrame) -> None:
    # 250 empresas x 20 anos, menos as saídas por default.
    assert 3500 <= len(painel) <= 5000
    assert painel["ID"].nunique() == 250
    assert painel["Ano"].nunique() == 20


def test_taxa_de_default_plausivel(painel: pd.DataFrame) -> None:
    taxa = painel["Default"].mean()
    assert 0.015 <= taxa <= 0.030, f"taxa fora da faixa alvo: {taxa:.4f}"


def test_empresa_sai_do_painel_apos_default(painel: pd.DataFrame) -> None:
    """Nenhuma observação depois do ano de default da empresa."""
    em_default = painel[painel["Default"] == 1]
    ultimo_ano = painel.groupby("ID")["Ano"].max()
    for id_, ano in zip(em_default["ID"], em_default["Ano"], strict=True):
        assert ano == ultimo_ano[id_]
    # E cada empresa entra em default no máximo uma vez.
    assert em_default.groupby("ID").size().max() == 1


def test_taxa_de_default_varia_entre_anos(painel: pd.DataFrame) -> None:
    """O fator sistêmico anual precisa produzir ciclo — base dos caps. 6 e 7."""
    por_ano = painel.groupby("Ano")["Default"].mean()
    assert por_ano.std() > 0.004
    assert por_ano.max() > 2 * por_ano.min()


def test_correlacao_serial_dentro_da_empresa(painel: pd.DataFrame) -> None:
    """Razões financeiras são persistentes; é isso que invalida o erro-padrão ingênuo."""
    base = painel.sort_values(["ID", "Ano"])
    for coluna in PREDITORES_CAP01:
        defasada = base.groupby("ID")[coluna].shift(1)
        valido = defasada.notna()
        rho = np.corrcoef(base.loc[valido, coluna], defasada[valido])[0, 1]
        assert rho > 0.5, f"{coluna} pouco persistente: rho={rho:.3f}"


def test_sinal_dos_coeficientes_verdadeiros() -> None:
    """Toda razão saudável reduz a probabilidade de default."""
    for nome in PREDITORES_CAP01:
        assert COEFS_VERDADEIROS[nome] < 0


def test_reprodutibilidade() -> None:
    pd.testing.assert_frame_equal(gerar_painel_scoring(), gerar_painel_scoring())


def test_semente_diferente_muda_a_base() -> None:
    a = gerar_painel_scoring(semente=1)
    b = gerar_painel_scoring(semente=2)
    assert not a.equals(b)


@pytest.mark.slow
def test_dgp_e_consistente_em_amostra_grande() -> None:
    """O teste que garante que o gerador e os coeficientes verdadeiros batem.

    Em amostra grande, o logit tem de recuperar cada coeficiente plantado dentro
    de três erros-padrão. Se este teste falhar, o problema é o DGP — não o
    estimador do capítulo 1.
    """
    import statsmodels.api as sm

    grande = gerar_painel_scoring(n_empresas=8000, semente=7)
    X = sm.add_constant(grande[PREDITORES_CAP01])
    ajuste = sm.Logit(grande["Default"], X).fit(disp=0)

    nomes = ["CONST", *PREDITORES_CAP01]
    for nome, estimado, erro in zip(nomes, ajuste.params, ajuste.bse, strict=True):
        verdadeiro = COEFS_VERDADEIROS[nome]
        assert abs(estimado - verdadeiro) < 3 * erro, (
            f"{nome}: verdadeiro={verdadeiro:.2f}, estimado={estimado:.2f}, ep={erro:.2f}"
        )


def test_registry_carrega_e_cacheia(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CREDRISK_DATA", str(tmp_path))
    primeira = registry.carregar("cap01_scoring")
    assert (tmp_path / "cap01_scoring.parquet").exists()
    segunda = registry.carregar("cap01_scoring")
    pd.testing.assert_frame_equal(primeira, segunda)


def test_registry_rejeita_nome_desconhecido() -> None:
    with pytest.raises(KeyError, match="cap01_scoring"):
        registry.carregar("base_inexistente")
