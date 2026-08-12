"""Testes dos capítulos 8 (validação de rating) e 9 (validação de carteira)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from credrisk.correlation.vasicek import taxa_condicional
from credrisk.data.generators import (
    GRADES,
    PD_POR_GRADE,
    gerar_carteira_rating,
    gerar_painel_validacao,
)
from credrisk.validation.carteira import (
    poder_do_teste,
    teste_berkowitz,
    teste_excedencias,
    teste_uniformidade,
    transformada_pit,
)
from credrisk.validation.rating import (
    auc,
    brier,
    curva_cap,
    curva_roc,
    erro_padrao_auc,
    hosmer_lemeshow,
    razao_de_acuracia,
    tabela_por_grade,
    teste_binomial,
    teste_binomial_com_correlacao,
    teste_binomial_por_grade,
)

# ---------------------------------------------------------------- capítulo 8


@pytest.fixture(scope="module")
def carteira() -> pd.DataFrame:
    return gerar_carteira_rating(rho=0.0, n_obrigados=8000, semente=1)


def test_grades_estao_ordenadas_por_risco() -> None:
    pds = [PD_POR_GRADE[g] for g in GRADES]
    assert all(a < b for a, b in zip(pds, pds[1:], strict=False))


def test_auc_bate_com_calculo_manual() -> None:
    escore = np.array([0.1, 0.4, 0.35, 0.8])
    alvo = np.array([0, 0, 1, 1])
    assert auc(escore, alvo) == pytest.approx(0.75)


def test_identidade_ar_igual_dois_auc_menos_um(carteira: pd.DataFrame) -> None:
    a = auc(carteira["escore"], carteira["Default"])
    assert razao_de_acuracia(carteira["escore"], carteira["Default"]) == pytest.approx(
        2 * a - 1
    )


def test_escore_aleatorio_tem_auc_meio() -> None:
    rng = np.random.default_rng(3)
    assert abs(auc(rng.random(20_000), rng.integers(0, 2, 20_000)) - 0.5) < 0.02


def test_sistema_ruidoso_discrimina_pior() -> None:
    """Erro de classificação degrada discriminação sem mexer na calibração."""
    bom = gerar_carteira_rating(rho=0.0, ruido_rating=0.0, n_obrigados=8000, semente=2)
    ruim = gerar_carteira_rating(rho=0.0, ruido_rating=1.5, n_obrigados=8000, semente=2)
    assert auc(bom["escore"], bom["Default"]) > auc(ruim["escore"], ruim["Default"])


def test_curvas_comecam_e_terminam_nos_cantos(carteira: pd.DataFrame) -> None:
    cap = curva_cap(carteira["escore"], carteira["Default"])
    roc = curva_roc(carteira["escore"], carteira["Default"])
    assert cap["fracao_defaults"].iloc[-1] == pytest.approx(1.0)
    assert cap["fracao_carteira"].iloc[-1] == pytest.approx(1.0)
    assert roc["verdadeiros_positivos"].iloc[-1] == pytest.approx(1.0)
    assert (np.diff(cap["fracao_defaults"]) >= -1e-12).all(), "CAP é monótona"


def test_erro_padrao_do_auc_e_positivo_e_pequeno(carteira: pd.DataFrame) -> None:
    ep = erro_padrao_auc(carteira["escore"], carteira["Default"])
    assert 0 < ep < 0.1


def test_brier_e_menor_para_o_modelo_melhor() -> None:
    """Prever a PD correta bate prever a média da carteira."""
    d = gerar_carteira_rating(rho=0.0, n_obrigados=8000, semente=4)
    do_modelo = brier(d["PD_atribuida"], d["Default"])
    da_media = brier(np.full(len(d), d["Default"].mean()), d["Default"])
    assert do_modelo < da_media


def test_tabela_por_grade_cobre_toda_a_carteira(carteira: pd.DataFrame) -> None:
    tabela = tabela_por_grade(carteira)
    assert tabela["n"].sum() == len(carteira)
    assert tabela["defaults"].sum() == carteira["Default"].sum()


def test_teste_binomial_rejeita_excesso_evidente() -> None:
    r = teste_binomial(n=1000, defaults=80, pd_prevista=0.01)
    assert r["rejeita"]
    assert r["p_valor"] < 1e-6


def test_teste_binomial_nao_rejeita_o_esperado() -> None:
    r = teste_binomial(n=1000, defaults=10, pd_prevista=0.01)
    assert not r["rejeita"]


def test_teste_binomial_e_unicaudal() -> None:
    """Default abaixo do previsto não é rejeitado: o risco prudencial é o excesso."""
    r = teste_binomial(n=1000, defaults=0, pd_prevista=0.01)
    assert not r["rejeita"]


def test_correlacao_alarga_muito_o_limite_critico() -> None:
    """O resultado central do capítulo 8."""
    sem = teste_binomial(n=848, defaults=30, pd_prevista=0.032)
    com = teste_binomial_com_correlacao(n=848, defaults=30, pd_prevista=0.032, rho=0.12)
    assert com["limite_critico"] > 1.5 * sem["limite_critico"]


def test_teste_com_rho_zero_reduz_ao_binomial() -> None:
    a = teste_binomial(n=500, defaults=12, pd_prevista=0.02)
    b = teste_binomial_com_correlacao(n=500, defaults=12, pd_prevista=0.02, rho=0.0)
    assert a["limite_critico"] == b["limite_critico"]


def test_teste_binomial_por_grade_devolve_uma_linha_por_grade(
    carteira: pd.DataFrame,
) -> None:
    t = teste_binomial_por_grade(carteira)
    assert len(t) == carteira["Grade"].nunique()
    assert set(t.columns) >= {"n", "defaults", "p_valor", "rejeita"}


def test_erro_tipo_um_do_binomial_explode_com_correlacao() -> None:
    """Sem correlação o teste respeita o nível nominal; com correlação, não.

    Em ambos os casos o modelo está CORRETO — toda rejeição é falso positivo.
    """
    def taxa(rho: float) -> float:
        rejeicoes = 0
        for s in range(60):
            d = gerar_carteira_rating(rho=rho, n_obrigados=4000, semente=5000 + s)
            t = teste_binomial_por_grade(d)
            rejeicoes += int(t.loc[t["Grade"] == "BB", "rejeita"].iloc[0])
        return rejeicoes / 60

    sem_correlacao = taxa(0.0)
    com_correlacao = taxa(0.12)
    assert sem_correlacao < 0.15, "sem correlação, perto dos 5% nominais"
    assert com_correlacao > 2 * max(sem_correlacao, 0.05)


def test_hosmer_lemeshow_detecta_descalibragem_grosseira() -> None:
    d = gerar_carteira_rating(rho=0.0, vies_pd=0.25, n_obrigados=8000, semente=6)
    assert hosmer_lemeshow(d["PD_atribuida"], d["Default"]).rejeita


def test_vies_de_pd_nao_altera_a_discriminacao() -> None:
    """Multiplicar todas as PDs por uma constante não muda a ordenação."""
    a = gerar_carteira_rating(rho=0.0, vies_pd=1.0, n_obrigados=6000, semente=7)
    b = gerar_carteira_rating(rho=0.0, vies_pd=0.25, n_obrigados=6000, semente=7)
    assert auc(a["escore"], a["Default"]) == pytest.approx(
        auc(b["escore"], b["Default"]), abs=1e-9
    )


def test_painel_de_validacao_tem_variacao_entre_anos() -> None:
    painel = gerar_painel_validacao(n_anos=12, n_obrigados=2000, rho=0.12)
    por_ano = painel.groupby("Ano")["Default"].mean()
    assert len(por_ano) == 12
    assert por_ano.std() > 0.005, "com correlação, a taxa tem de variar entre anos"


# ---------------------------------------------------------------- capítulo 9


def _u_simulado(n_anos: int, rho_real: float, rho_modelo: float, semente: int,
                n_cenarios: int = 3000) -> np.ndarray:
    """Gera a transformada PIT de perdas reais contra um modelo possivelmente errado."""
    rng = np.random.default_rng(semente)
    pd_inc = 0.015
    perdas = taxa_condicional(rng.normal(0, 1, n_anos), pd_inc, rho_real)
    cenarios = [
        taxa_condicional(rng.normal(0, 1, n_cenarios), pd_inc, rho_modelo)
        for _ in range(n_anos)
    ]
    return transformada_pit(perdas, cenarios)


def test_pit_fica_no_intervalo_unitario() -> None:
    u = _u_simulado(20, 0.12, 0.12, semente=1)
    assert ((u > 0) & (u < 1)).all()
    assert len(u) == 20


def test_pit_de_modelo_correto_e_aproximadamente_uniforme() -> None:
    u = _u_simulado(500, 0.12, 0.12, semente=2)
    assert not teste_uniformidade(u)["rejeita"]
    assert abs(u.mean() - 0.5) < 0.06


def test_pit_de_modelo_que_subestima_a_cauda_se_acumula_a_direita() -> None:
    """Modelo com rho baixo demais: perdas reais caem alto na distribuição prevista."""
    u = _u_simulado(400, 0.24, 0.06, semente=3)
    assert teste_uniformidade(u)["rejeita"]


def test_pit_exige_um_conjunto_de_cenarios_por_ano() -> None:
    with pytest.raises(ValueError):
        transformada_pit(np.array([0.01, 0.02]), [np.random.random(100)])


def test_berkowitz_nao_rejeita_modelo_correto() -> None:
    u = _u_simulado(200, 0.12, 0.12, semente=4)
    assert not teste_berkowitz(u).rejeita


def test_berkowitz_rejeita_modelo_errado() -> None:
    u = _u_simulado(200, 0.12, 0.03, semente=5)
    r = teste_berkowitz(u)
    assert r.rejeita
    assert r.desvio > 1.0, "cauda subestimada infla a dispersão dos z"


def test_berkowitz_exige_amostra_minima() -> None:
    with pytest.raises(ValueError):
        teste_berkowitz(np.array([0.2, 0.5, 0.8]))


def test_berkowitz_respeita_o_nivel_nominal() -> None:
    taxa = poder_do_teste(
        lambda k: _u_simulado(20, 0.12, 0.12, semente=100 + k), n_repeticoes=60
    )
    assert taxa < 0.20, f"erro tipo I alto demais: {taxa:.2f}"


def test_poder_cresce_com_o_numero_de_anos() -> None:
    """A mensagem do capítulo 9: poder é escasso porque anos são escassos."""
    curto = poder_do_teste(
        lambda k: _u_simulado(10, 0.12, 0.05, semente=200 + k), n_repeticoes=60
    )
    longo = poder_do_teste(
        lambda k: _u_simulado(60, 0.12, 0.05, semente=300 + k), n_repeticoes=60
    )
    assert longo > curto
    assert curto < 0.75, "com 10 anos, o teste não deveria ser confiável"


def test_berkowitz_tem_mais_poder_que_contar_violacoes() -> None:
    def gerar(k: int) -> np.ndarray:
        return _u_simulado(20, 0.12, 0.05, semente=400 + k)

    poder_b = poder_do_teste(gerar, n_repeticoes=60, teste="berkowitz")
    poder_e = poder_do_teste(gerar, n_repeticoes=60, teste="excedencias")
    assert poder_b > poder_e


def test_excedencias_conta_certo() -> None:
    u = np.array([0.1, 0.5, 0.995, 0.999, 0.7])
    r = teste_excedencias(u, nivel=0.99)
    assert r["violacoes"] == 2
    assert r["esperadas"] == pytest.approx(0.05)


def test_z_de_berkowitz_e_normal_sob_modelo_correto() -> None:
    u = _u_simulado(1000, 0.12, 0.12, semente=8)
    z = stats.norm.ppf(u)
    assert abs(z.mean()) < 0.15
    assert abs(z.std(ddof=1) - 1.0) < 0.15
