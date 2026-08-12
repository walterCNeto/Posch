"""Validação de sistemas de rating — capítulo 8.

Validar um sistema de rating é responder a duas perguntas distintas, que a
prática frequentemente confunde:

**Discriminação** — o sistema ordena corretamente? Os piores devedores recebem
as piores notas? Medida por CAP/AR e ROC/AUC.

**Calibração** — os níveis estão certos? A grade que promete 1,1% de default
entrega 1,1%? Medida por testes de aderência entre PD prevista e frequência
observada.

As duas são independentes. Um sistema pode ordenar perfeitamente e errar todos
os níveis por um fator de três; outro pode acertar a média da carteira e não
distinguir bom de ruim. O capítulo 2 já mostrou o primeiro caso; aqui ele é
medido.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Poder discriminante
# --------------------------------------------------------------------------

def auc(escore: np.ndarray, alvo: np.ndarray) -> float:
    """Área sob a curva ROC, pela identidade com a estatística de Mann-Whitney.

    Escore alto deve indicar maior risco. Interpretação direta: é a
    probabilidade de que um devedor sorteado entre os que quebraram tenha
    escore pior que um sorteado entre os que não quebraram.
    """
    escore = np.asarray(escore, dtype=float)
    alvo = np.asarray(alvo)
    maus, bons = escore[alvo == 1], escore[alvo == 0]
    if len(maus) == 0 or len(bons) == 0:
        raise ValueError("É preciso ter ao menos um default e um não-default.")
    postos = stats.rankdata(np.concatenate([maus, bons]))
    soma = postos[: len(maus)].sum()
    return float((soma - len(maus) * (len(maus) + 1) / 2) / (len(maus) * len(bons)))


def razao_de_acuracia(escore: np.ndarray, alvo: np.ndarray) -> float:
    """Razão de acurácia (AR, ou Gini): ``2 × AUC - 1``."""
    return 2.0 * auc(escore, alvo) - 1.0


def curva_cap(escore: np.ndarray, alvo: np.ndarray) -> pd.DataFrame:
    """Curva de perfil de acurácia (CAP), do pior escore para o melhor.

    O eixo horizontal é a fração da carteira examinada em ordem decrescente de
    risco; o vertical, a fração dos defaults já capturada. Um sistema perfeito
    sobe na diagonal máxima; um sistema aleatório segue a diagonal de 45 graus.
    """
    escore = np.asarray(escore, dtype=float)
    alvo = np.asarray(alvo, dtype=float)
    ordem = np.argsort(-escore)
    capturados = np.cumsum(alvo[ordem])
    return pd.DataFrame(
        {
            "fracao_carteira": np.arange(1, len(alvo) + 1) / len(alvo),
            "fracao_defaults": capturados / capturados[-1],
        }
    )


def curva_roc(escore: np.ndarray, alvo: np.ndarray) -> pd.DataFrame:
    """Curva ROC: taxa de verdadeiros positivos contra falsos positivos."""
    escore = np.asarray(escore, dtype=float)
    alvo = np.asarray(alvo, dtype=float)
    ordem = np.argsort(-escore)
    y = alvo[ordem]
    tvp = np.cumsum(y) / y.sum()
    tfp = np.cumsum(1 - y) / (len(y) - y.sum())
    return pd.DataFrame({"falsos_positivos": tfp, "verdadeiros_positivos": tvp})


def erro_padrao_auc(escore: np.ndarray, alvo: np.ndarray) -> float:
    """Erro-padrão do AUC pela aproximação de Hanley-McNeil.

    Supõe observações independentes — hipótese que a correlação de defaults
    viola, do mesmo modo que viola os testes de calibração adiante.
    """
    a = auc(escore, alvo)
    alvo = np.asarray(alvo)
    n1, n0 = int((alvo == 1).sum()), int((alvo == 0).sum())
    q1 = a / (2 - a)
    q2 = 2 * a**2 / (1 + a)
    var = (
        a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)
    ) / (n1 * n0)
    return float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------

def brier(pd_prevista: np.ndarray, alvo: np.ndarray) -> float:
    """Escore de Brier: erro quadrático médio da probabilidade prevista."""
    return float(np.mean((np.asarray(pd_prevista, float) - np.asarray(alvo, float)) ** 2))


def tabela_por_grade(dados: pd.DataFrame, coluna_grade: str = "Grade") -> pd.DataFrame:
    """Agrega defaults observados e PD prometida por grade de rating."""
    tabela = (
        dados.groupby(coluna_grade)
        .agg(
            n=("Default", "size"),
            defaults=("Default", "sum"),
            pd_prevista=("PD_atribuida", "mean"),
        )
        .reset_index()
    )
    tabela["taxa_observada"] = tabela["defaults"] / tabela["n"]
    return tabela


def teste_binomial(
    n: int, defaults: int, pd_prevista: float, nivel: float = 0.95
) -> dict[str, float]:
    """Teste binomial unicaudal de calibração de uma grade.

    Sob a hipótese de que a PD prometida está correta **e** de que os defaults
    são independentes, o número de defaults segue uma binomial. O teste rejeita
    quando o observado ultrapassa o quantil superior dessa binomial.

    A segunda hipótese é a frágil, e o capítulo mede o estrago que ela causa: o
    capítulo 6 mostrou que os defaults são correlacionados, e a binomial não
    tem como saber disso.
    """
    limite = int(stats.binom.ppf(nivel, n, pd_prevista))
    p_valor = float(stats.binom.sf(defaults - 1, n, pd_prevista))
    return {
        "n": int(n),
        "defaults": int(defaults),
        "esperados": float(n * pd_prevista),
        "limite_critico": limite,
        "p_valor": p_valor,
        "rejeita": bool(defaults > limite),
    }


def teste_binomial_por_grade(
    dados: pd.DataFrame, nivel: float = 0.95, coluna_grade: str = "Grade"
) -> pd.DataFrame:
    """Aplica o teste binomial a cada grade da carteira."""
    tabela = tabela_por_grade(dados, coluna_grade)
    saida = []
    for _, linha in tabela.iterrows():
        r = teste_binomial(int(linha["n"]), int(linha["defaults"]),
                           float(linha["pd_prevista"]), nivel)
        r[coluna_grade] = linha[coluna_grade]
        r["taxa_observada"] = linha["taxa_observada"]
        r["pd_prevista"] = linha["pd_prevista"]
        saida.append(r)
    colunas = [coluna_grade, "n", "defaults", "esperados", "pd_prevista",
               "taxa_observada", "limite_critico", "p_valor", "rejeita"]
    return pd.DataFrame(saida)[colunas]


@dataclass
class ResultadoHL:
    """Saída do teste de Hosmer-Lemeshow."""

    estatistica: float
    graus_liberdade: int
    p_valor: float

    @property
    def rejeita(self) -> bool:
        return self.p_valor < 0.05


def hosmer_lemeshow(
    pd_prevista: np.ndarray, alvo: np.ndarray, n_grupos: int = 10
) -> ResultadoHL:
    """Teste de aderência de Hosmer-Lemeshow.

    Agrupa as observações por decil de PD prevista e compara defaults previstos
    com observados em cada grupo. É um teste conjunto: responde se o sistema
    está calibrado **como um todo**, não grade a grade.
    """
    p = np.asarray(pd_prevista, float)
    y = np.asarray(alvo, float)
    cortes = np.quantile(p, np.linspace(0, 1, n_grupos + 1))
    cortes[0], cortes[-1] = -np.inf, np.inf
    grupo = np.digitize(p, cortes[1:-1])

    estat = 0.0
    usados = 0
    for g in range(n_grupos):
        m = grupo == g
        if m.sum() == 0:
            continue
        obs = y[m].sum()
        esp = p[m].sum()
        n_g = m.sum()
        if esp <= 0 or esp >= n_g:
            continue
        estat += (obs - esp) ** 2 / (esp * (1 - esp / n_g))
        usados += 1

    gl = max(usados - 2, 1)
    return ResultadoHL(
        estatistica=float(estat),
        graus_liberdade=gl,
        p_valor=float(stats.chi2.sf(estat, gl)),
    )


def teste_binomial_com_correlacao(
    n: int, defaults: int, pd_prevista: float, rho: float, nivel: float = 0.95,
    n_nos: int = 200,
) -> dict[str, float]:
    """Versão do teste binomial que reconhece a correlação de ativos.

    Em vez de supor defaults independentes, integra a binomial sobre o fator
    sistêmico — exatamente a verossimilhança do capítulo 6. O limite crítico
    resultante é bem mais alto, porque a distribuição do número de defaults tem
    cauda muito mais gorda que a binomial.

    Este é o teste que deveria ser usado; o binomial simples é o que costuma
    aparecer nos relatórios.
    """
    if rho <= 0:
        return teste_binomial(n, defaults, pd_prevista, nivel)

    t, w = np.polynomial.hermite.hermgauss(n_nos)
    x = np.sqrt(2.0) * t
    peso = w / np.sqrt(np.pi)
    limiar = stats.norm.ppf(pd_prevista)
    p_cond = stats.norm.cdf((limiar - np.sqrt(rho) * x) / np.sqrt(1 - rho))

    k = np.arange(0, n + 1)
    massa = (stats.binom.pmf(k[:, None], n, p_cond[None, :]) * peso[None, :]).sum(axis=1)
    massa = massa / massa.sum()
    acumulada = np.cumsum(massa)

    limite = int(np.searchsorted(acumulada, nivel))
    p_valor = float(massa[defaults:].sum())
    return {
        "n": int(n),
        "defaults": int(defaults),
        "esperados": float(n * pd_prevista),
        "limite_critico": limite,
        "p_valor": p_valor,
        "rejeita": bool(defaults > limite),
    }
