"""Capital regulatório pela abordagem de ratings internos — capítulo 12.

A fórmula de requerimento de capital para risco de crédito não é um número
arbitrário negociado em comitê: é exatamente o modelo do capítulo 7, com duas
diferenças deliberadas.

A primeira é que a correlação **não é estimada** pelo banco — é prescrita por
fórmula, decrescente na probabilidade de default. O capítulo 6 explica por quê:
o parâmetro não é estimável com a precisão que o capital exigiria, e deixá-lo
livre criaria dispersão enorme entre bancos com carteiras parecidas.

A segunda é o ajuste de maturidade, que cobre o risco de migração de rating ao
longo do prazo — algo que o modelo de default puro do capítulo 7 ignora.

Todas as constantes aqui vêm do arcabouço de Basileia e são reproduzidas como
publicadas. Elas são convenção regulatória, não resultado de estimação: o objeto
de estudo deste módulo é o que a fórmula faz, não de onde vieram seus números.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Percentil de confiança do requerimento de capital.
NIVEL_CONFIANCA = 0.999

#: Fator de conversão de requerimento de capital em ativo ponderado pelo risco,
#: igual ao inverso do requerimento mínimo de 8%.
FATOR_RWA = 12.5


def correlacao_prescrita(
    pd_i: np.ndarray | float,
    classe: str = "corporate",
    faturamento_milhoes: float | None = None,
) -> np.ndarray:
    """Correlação de ativos prescrita pelo arcabouço, por classe de exposição.

    Para exposições corporativas a correlação interpola entre 24% e 12%,
    decrescendo com a PD. A lógica declarada é que devedores de pior qualidade
    quebram mais por razões próprias e menos por razões sistêmicas — o risco
    idiossincrático domina na ponta ruim da escala.

    Parameters
    ----------
    classe
        ``corporate``, ``varejo_hipotecario``, ``varejo_rotativo`` ou
        ``varejo_outros``.
    faturamento_milhoes
        Para ``corporate``, aplica o ajuste de porte a empresas com faturamento
        entre 5 e 50 milhões, reduzindo a correlação em até 4 pontos
        percentuais. Empresas menores são consideradas menos expostas ao fator
        comum.
    """
    p = np.clip(np.asarray(pd_i, dtype=float), 1e-8, 1.0)

    if classe == "varejo_hipotecario":
        return np.full_like(p, 0.15)
    if classe == "varejo_rotativo":
        return np.full_like(p, 0.04)
    if classe == "varejo_outros":
        peso = (1 - np.exp(-35 * p)) / (1 - np.exp(-35.0))
        return 0.03 * peso + 0.16 * (1 - peso)
    if classe != "corporate":
        raise ValueError(f"classe desconhecida: {classe}")

    peso = (1 - np.exp(-50 * p)) / (1 - np.exp(-50.0))
    R = 0.12 * peso + 0.24 * (1 - peso)

    if faturamento_milhoes is not None:
        S = float(np.clip(faturamento_milhoes, 5.0, 50.0))
        R = R - 0.04 * (1 - (S - 5.0) / 45.0)
    return R


def ajuste_maturidade(pd_i: np.ndarray | float, maturidade: float = 2.5) -> np.ndarray:
    """Multiplicador de maturidade do requerimento corporativo.

    .. math:: b(PD) = \\big(0{,}11852 - 0{,}05478 \\ln PD\\big)^2

    e o multiplicador é :math:`\\frac{1 + (M - 2{,}5)\\,b}{1 - 1{,}5\\,b}`.

    Cobre o risco de **migração**: um crédito de prazo longo pode se deteriorar
    sem entrar em default, e essa deterioração custa. O efeito é maior para PDs
    baixas, porque um AAA tem muito mais espaço para piorar que um CCC.
    """
    p = np.clip(np.asarray(pd_i, dtype=float), 1e-8, 1.0)
    b = (0.11852 - 0.05478 * np.log(p)) ** 2
    return (1 + (maturidade - 2.5) * b) / (1 - 1.5 * b)


def requerimento_capital(
    pd_i: np.ndarray | float,
    lgd: np.ndarray | float,
    classe: str = "corporate",
    maturidade: float = 2.5,
    faturamento_milhoes: float | None = None,
) -> np.ndarray:
    """Requerimento de capital ``K`` como fração da exposição.

    .. math::
       K = \\left[LGD \\cdot \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) + \\sqrt{R}\\,\\Phi^{-1}(0{,}999)}{\\sqrt{1-R}}\\right) - PD \\cdot LGD\\right] \\times \\text{ajuste de maturidade}

    O primeiro termo é a perda no percentil 99,9% — exatamente o quantil de
    Vasicek do capítulo 7. O segundo subtrai a perda esperada, que é coberta por
    provisão e não por capital. Somar os dois seria contar a mesma perda duas
    vezes.
    """
    p = np.clip(np.asarray(pd_i, dtype=float), 1e-8, 1 - 1e-8)
    lgd = np.asarray(lgd, dtype=float)
    R = correlacao_prescrita(p, classe, faturamento_milhoes)

    quantil = stats.norm.cdf(
        (stats.norm.ppf(p) + np.sqrt(R) * stats.norm.ppf(NIVEL_CONFIANCA))
        / np.sqrt(1 - R)
    )
    k = lgd * quantil - p * lgd

    if classe == "corporate":
        k = k * ajuste_maturidade(p, maturidade)
    return np.maximum(k, 0.0)


def rwa(
    ead: np.ndarray | float,
    pd_i: np.ndarray | float,
    lgd: np.ndarray | float,
    classe: str = "corporate",
    maturidade: float = 2.5,
) -> np.ndarray:
    """Ativo ponderado pelo risco: ``K × 12,5 × EAD``."""
    k = requerimento_capital(pd_i, lgd, classe, maturidade)
    return np.asarray(ead, dtype=float) * k * FATOR_RWA


def perda_esperada_regulatoria(
    ead: np.ndarray | float, pd_i: np.ndarray | float, lgd: np.ndarray | float
) -> np.ndarray:
    """Perda esperada regulatória: ``PD × LGD × EAD``, coberta por provisão."""
    return (
        np.asarray(ead, dtype=float)
        * np.asarray(pd_i, dtype=float)
        * np.asarray(lgd, dtype=float)
    )


def resumo_carteira(
    carteira: pd.DataFrame,
    classe: str = "corporate",
    maturidade: float = 2.5,
) -> pd.Series:
    """Agrega capital, RWA e perda esperada de uma carteira posição a posição.

    Note que a agregação é uma **soma simples**: o requerimento de cada exposição
    não depende das demais. Essa propriedade — invariância à carteira — é o que
    torna a fórmula aplicável operacionalmente, e é também sua principal
    limitação. Ver :func:`comparar_com_economico`.
    """
    ead = carteira["EAD"].to_numpy(float)
    p = carteira["PD"].to_numpy(float)
    lgd = carteira["LGD"].to_numpy(float)

    k = requerimento_capital(p, lgd, classe, maturidade)
    ativos = ead * k * FATOR_RWA
    pe = perda_esperada_regulatoria(ead, p, lgd)

    total = ead.sum()
    return pd.Series(
        {
            "EAD total": total,
            "PD média ponderada": float(np.average(p, weights=ead)),
            "perda esperada": float(pe.sum()),
            "capital exigido": float((ead * k).sum()),
            "RWA": float(ativos.sum()),
            "capital / EAD": float((ead * k).sum() / total),
            "RWA / EAD": float(ativos.sum() / total),
        }
    )


def curva_de_capital(
    pds: np.ndarray | None = None,
    lgd: float = 0.45,
    classes: tuple[str, ...] = ("corporate", "varejo_outros", "varejo_hipotecario",
                                "varejo_rotativo"),
    maturidade: float = 2.5,
) -> pd.DataFrame:
    """Requerimento de capital em função da PD, por classe de exposição."""
    pds = np.geomspace(0.0003, 0.30, 120) if pds is None else np.asarray(pds)
    saida = {"PD": pds}
    for classe in classes:
        saida[classe] = requerimento_capital(pds, lgd, classe, maturidade)
    return pd.DataFrame(saida)


def discretizar_em_grades(
    pd_continua: np.ndarray, n_grades: int, geometrica: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa PDs contínuas em grades e devolve a PD média de cada uma.

    Sistemas de rating internos não atribuem PD contínua: classificam em um
    número finito de grades, e todos os devedores de uma grade recebem a mesma
    PD — a média da grade.

    Fronteiras geométricas (igualmente espaçadas em log) são a prática usual,
    porque a PD varia por ordens de grandeza ao longo da escala.

    Returns
    -------
    pd_da_grade : PD atribuída a cada devedor após a discretização.
    indice : índice da grade de cada devedor.
    """
    p = np.asarray(pd_continua, dtype=float)
    if n_grades < 1:
        raise ValueError("é preciso ao menos uma grade")

    if geometrica:
        bordas = np.geomspace(max(p.min(), 1e-6), p.max() * 1.0001, n_grades + 1)
    else:
        bordas = np.linspace(p.min(), p.max() * 1.0001, n_grades + 1)

    indice = np.clip(np.digitize(p, bordas[1:-1]), 0, n_grades - 1)
    pd_grade = np.empty_like(p)
    for g in range(n_grades):
        m = indice == g
        if m.any():
            pd_grade[m] = p[m].mean()
    return pd_grade, indice


def efeito_da_granularidade(
    carteira: pd.DataFrame,
    grades: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 15, 25),
    classe: str = "corporate",
) -> pd.DataFrame:
    """Capital exigido em função do número de grades do sistema de rating.

    Discretizar a PD é perda de informação, e a fórmula de capital é
    **côncava** em PD na faixa relevante — o requerimento cresce rápido no início
    da escala e achata na ponta ruim. Por Jensen, substituir PDs distintas pela
    média da grade **aumenta** o capital calculado, sem que o risco tenha mudado.

    O sinal importa: agrupar demais penaliza o banco, o que alinha o incentivo
    na direção certa. Um sistema de rating grosseiro custa capital, e o custo é
    mensurável — que é o que esta função mede.
    """
    p = carteira["PD"].to_numpy(float)
    ead = carteira["EAD"].to_numpy(float)
    lgd = carteira["LGD"].to_numpy(float)

    referencia = float((ead * requerimento_capital(p, lgd, classe)).sum())

    linhas = []
    for n in grades:
        pd_grade, _ = discretizar_em_grades(p, n)
        capital = float((ead * requerimento_capital(pd_grade, lgd, classe)).sum())
        linhas.append(
            {
                "grades": n,
                "capital": capital,
                "vs PD contínua": capital / referencia - 1,
            }
        )
    linhas.append({"grades": "contínua", "capital": referencia, "vs PD contínua": 0.0})
    return pd.DataFrame(linhas)
