"""Geradores de bases sintéticas do curso.

Cada capítulo tem um processo gerador de dados (DGP) próprio, com semente fixa e
parâmetros verdadeiros expostos como constante do módulo. Isso permite que os
notebooks e os testes comparem o que o estimador *recuperou* com o que foi
*plantado* — algo que nenhuma base real permite fazer.

Convenção: toda função `gerar_*` recebe `semente` e devolve um ``DataFrame``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Capítulo 1 — escore de crédito com logit
# --------------------------------------------------------------------------

#: Coeficientes verdadeiros do modelo logit latente que gera o default.
#: As magnitudes seguem a ordem de grandeza dos índices de Altman: razões
#: financeiras saudáveis puxam a probabilidade de default para baixo.
COEFS_VERDADEIROS: dict[str, float] = {
    "CONST": -1.30,
    "WC/TA": -0.40,
    "RE/TA": -1.45,
    "EBIT/TA": -8.00,
    "ME/TL": -1.60,
    "S/TA": -0.50,
}

PREDITORES_CAP01: list[str] = ["WC/TA", "RE/TA", "EBIT/TA", "ME/TL", "S/TA"]

_N_EMPRESAS_PADRAO = 250
_ANO_INICIAL_PADRAO = 1985
_N_ANOS_PADRAO = 20


def gerar_painel_scoring(
    n_empresas: int = _N_EMPRESAS_PADRAO,
    n_anos: int = _N_ANOS_PADRAO,
    ano_inicial: int = _ANO_INICIAL_PADRAO,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera o painel empresa-ano de razões financeiras e default do capítulo 1.

    Estrutura do DGP, em três camadas:

    1. **Efeito de empresa** — cada empresa tem um nível próprio e persistente de
       cada razão financeira. É o que torna as observações da mesma empresa
       correlacionadas e, portanto, o que quebra a hipótese de independência
       usada nos erros-padrão ingênuos do logit.
    2. **Fator sistêmico anual** — um choque macro comum a todas as empresas no
       mesmo ano, que desloca simultaneamente as razões e o índice latente. É a
       semente da correlação de defaults explorada nos capítulos 6 e 7.
    3. **Ruído idiossincrático AR(1)** — variação própria da empresa ao longo do
       tempo, com memória.

    O default é sorteado de um logit sobre o índice latente. Uma vez em default,
    a empresa sai do painel (não há observações pós-falência), o que produz o
    painel desbalanceado típico dos dados de bancarrota.

    Parameters
    ----------
    n_empresas
        Número de empresas simuladas.
    n_anos
        Horizonte em anos.
    ano_inicial
        Primeiro ano do painel.
    semente
        Semente do gerador aleatório. Fixa por padrão para reprodutibilidade.

    Returns
    -------
    pandas.DataFrame
        Colunas ``ID``, ``Ano``, ``Default`` e as cinco razões de
        :data:`PREDITORES_CAP01`.
    """
    rng = np.random.default_rng(semente)

    # Camada 1: nível próprio de cada empresa (média e dispersão por razão).
    niveis = {
        "WC/TA": (0.15, 0.16),
        "RE/TA": (0.20, 0.28),
        "EBIT/TA": (0.07, 0.06),
        "ME/TL": (1.10, 0.55),
        "S/TA": (1.00, 0.45),
    }
    efeito_empresa = {
        nome: rng.normal(media, desvio, size=n_empresas)
        for nome, (media, desvio) in niveis.items()
    }

    # Camada 2: fator sistêmico anual (choque macro comum).
    fator_ano = rng.normal(0.0, 1.0, size=n_anos)

    # Camada 3: ruído AR(1) por empresa e por razão.
    rho = 0.55
    sensibilidade_macro = {
        "WC/TA": 0.030,
        "RE/TA": 0.055,
        "EBIT/TA": 0.020,
        "ME/TL": 0.140,
        "S/TA": 0.040,
    }
    escala_ruido = {
        "WC/TA": 0.060,
        "RE/TA": 0.090,
        "EBIT/TA": 0.030,
        "ME/TL": 0.180,
        "S/TA": 0.090,
    }

    ruido = {
        nome: rng.normal(0.0, escala_ruido[nome], size=n_empresas)
        for nome in PREDITORES_CAP01
    }

    ativa = np.ones(n_empresas, dtype=bool)
    registros: list[dict[str, float]] = []

    for t in range(n_anos):
        ano = ano_inicial + t

        valores: dict[str, np.ndarray] = {}
        for nome in PREDITORES_CAP01:
            choque = rng.normal(0.0, escala_ruido[nome] * np.sqrt(1 - rho**2), n_empresas)
            ruido[nome] = rho * ruido[nome] + choque
            valores[nome] = (
                efeito_empresa[nome]
                + sensibilidade_macro[nome] * fator_ano[t]
                + ruido[nome]
            )

        # ME/TL e S/TA são razões estritamente positivas por construção econômica.
        valores["ME/TL"] = np.clip(valores["ME/TL"], 0.02, None)
        valores["S/TA"] = np.clip(valores["S/TA"], 0.02, None)

        indice = np.full(n_empresas, COEFS_VERDADEIROS["CONST"])
        for nome in PREDITORES_CAP01:
            indice += COEFS_VERDADEIROS[nome] * valores[nome]

        prob = 1.0 / (1.0 + np.exp(-indice))
        default = rng.random(n_empresas) < prob

        for i in np.flatnonzero(ativa):
            registro = {"ID": i + 1, "Ano": ano, "Default": int(default[i])}
            registro.update({nome: float(valores[nome][i]) for nome in PREDITORES_CAP01})
            registros.append(registro)

        # Empresa em default sai do painel a partir do ano seguinte.
        ativa &= ~default

    painel = pd.DataFrame.from_records(registros)
    return painel.sort_values(["ID", "Ano"], ignore_index=True)


# --------------------------------------------------------------------------
# Capítulo 2 — abordagem estrutural (Merton)
# --------------------------------------------------------------------------

#: Parâmetros verdadeiros do processo de valor do ativo do capítulo 2.
PARAMS_MERTON: dict[str, float] = {
    "V0": 100_000.0,      # valor do ativo no primeiro dia
    "mu": 0.08,           # deriva anual do ativo (medida física)
    "sigma_V": 0.28,      # volatilidade anual do ativo
    "L": 60_000.0,        # dívida (valor de face, vencimento em 1 ano)
    "r": 0.05,            # taxa livre de risco (log)
    "T": 1.0,             # horizonte, em anos
}

DIAS_UTEIS_ANO = 260


def gerar_serie_merton(
    n_dias: int = DIAS_UTEIS_ANO,
    semente: int = 42,
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Gera a série diária de valor de mercado do capital próprio de uma empresa.

    O valor do ativo segue um movimento browniano geométrico com deriva e
    volatilidade conhecidas. O capital próprio é então calculado como uma opção
    de compra sobre o ativo, com preço de exercício igual à dívida — que é
    exatamente o que o modelo de Merton postula.

    O ponto do capítulo é que, na prática, **só se observa o capital próprio**.
    O valor e a volatilidade do ativo têm de ser inferidos. Como aqui eles são
    conhecidos, dá para medir o erro dessa inferência.

    Returns
    -------
    pandas.DataFrame
        Colunas ``Dia``, ``E`` (valor de mercado do capital próprio), ``L``
        (dívida), ``r`` (taxa livre de risco) e ``V_verdadeiro`` — esta última
        é o gabarito, e o notebook a esconde até a hora da conferência.
    """
    from credrisk.structural.merton import preco_call

    p = dict(PARAMS_MERTON if params is None else params)
    rng = np.random.default_rng(semente)

    dt = 1.0 / DIAS_UTEIS_ANO
    choques = rng.normal(0.0, 1.0, size=n_dias - 1)
    deriva = (p["mu"] - 0.5 * p["sigma_V"] ** 2) * dt
    log_retornos = deriva + p["sigma_V"] * np.sqrt(dt) * choques
    V = p["V0"] * np.exp(np.concatenate([[0.0], np.cumsum(log_retornos)]))

    # A dívida vence sempre daqui a T anos (horizonte rolante), como na prática
    # de mercado: a cada dia recalcula-se a opção com o mesmo prazo residual.
    E = preco_call(V, p["L"], p["r"], p["T"], p["sigma_V"])

    return pd.DataFrame(
        {
            "Dia": np.arange(1, n_dias + 1),
            "E": E,
            "L": p["L"],
            "r": p["r"],
            "V_verdadeiro": V,
        }
    )


def gerar_carteira_merton(
    n_empresas: int = 200, semente: int = 7
) -> pd.DataFrame:
    """Gera uma seção transversal de empresas com alavancagem e volatilidade variadas.

    Usada para mostrar como a distância ao default ordena o risco e como a
    tradução de distância em probabilidade depende da distribuição assumida.
    """
    rng = np.random.default_rng(semente)
    p = PARAMS_MERTON

    alavancagem = rng.uniform(0.20, 0.85, n_empresas)     # L / V
    sigma_V = rng.uniform(0.15, 0.50, n_empresas)
    V = rng.uniform(50_000, 500_000, n_empresas)
    L = alavancagem * V

    return pd.DataFrame(
        {
            "ID": np.arange(1, n_empresas + 1),
            "V": V,
            "L": L,
            "sigma_V": sigma_V,
            "mu": p["mu"],
            "r": p["r"],
        }
    )


# --------------------------------------------------------------------------
# Capítulo 3 — matrizes de transição
# --------------------------------------------------------------------------

#: Escala de rating usada no curso, do melhor ao pior. ``D`` é absorvente.
RATINGS: list[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


def gerador_verdadeiro() -> np.ndarray:
    """Matriz geradora (intensidades de transição) usada para simular o cap. 3.

    Entrada :math:`q_{ij}` é a taxa instantânea de migração de ``i`` para ``j``,
    por ano. As linhas somam zero; ``D`` é absorvente (linha inteira nula).

    Note que **AAA tem intensidade estritamente positiva para D**: é possível,
    embora raríssimo, migrar direto. Nenhuma amostra finita observará isso, e é
    exatamente aí que os dois estimadores do capítulo divergem.
    """
    Q = np.zeros((8, 8))
    # Vizinhos imediatos concentram a massa; saltos longos são raros mas existem.
    intensidades = {
        (0, 1): 0.0800, (0, 2): 0.0080, (0, 3): 0.0010, (0, 4): 0.0002,
        (0, 5): 0.00005, (0, 6): 0.00002, (0, 7): 0.00001,
        (1, 0): 0.0060, (1, 2): 0.0900, (1, 3): 0.0090, (1, 4): 0.0012,
        (1, 5): 0.0003, (1, 6): 0.00008, (1, 7): 0.00003,
        (2, 0): 0.0006, (2, 1): 0.0230, (2, 3): 0.0570, (2, 4): 0.0060,
        (2, 5): 0.0012, (2, 6): 0.0003, (2, 7): 0.0002,
        (3, 0): 0.0002, (3, 1): 0.0025, (3, 2): 0.0460, (3, 4): 0.0490,
        (3, 5): 0.0110, (3, 6): 0.0016, (3, 7): 0.0022,
        (4, 0): 0.0001, (4, 1): 0.0008, (4, 2): 0.0035, (4, 3): 0.0620,
        (4, 5): 0.0800, (4, 6): 0.0075, (4, 7): 0.0090,
        (5, 1): 0.0002, (5, 2): 0.0010, (5, 3): 0.0045, (5, 4): 0.0620,
        (5, 6): 0.0480, (5, 7): 0.0410,
        (6, 2): 0.0004, (6, 3): 0.0015, (6, 4): 0.0110, (6, 5): 0.1400,
        (6, 7): 0.2300,
    }
    for (i, j), taxa in intensidades.items():
        Q[i, j] = taxa
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def gerar_historico_ratings(
    n_empresas: int = 1200,
    anos: float = 15.0,
    semente: int = 42,
    Q: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simula históricos de rating por cadeia de Markov em tempo contínuo.

    Cada empresa entra com um rating inicial e migra em tempos exponenciais com
    intensidade dada pela geradora. Quem chega a ``D`` para ali; quem não migra
    até o fim da janela é censurado à direita — as duas situações que qualquer
    base de rating real apresenta.

    Returns
    -------
    pandas.DataFrame
        Colunas ``ID``, ``Data`` (em anos desde o início), ``Rating`` (rótulo) e
        ``Estado`` (índice em :data:`RATINGS`). Uma linha por observação de
        rating, incluindo a inicial.
    """
    Q = gerador_verdadeiro() if Q is None else Q
    rng = np.random.default_rng(semente)
    n_estados = len(RATINGS)
    absorvente = n_estados - 1

    # Distribuição inicial concentrada no miolo da escala, como uma carteira real.
    pesos = np.array([0.04, 0.10, 0.22, 0.26, 0.20, 0.13, 0.05])
    pesos = pesos / pesos.sum()

    taxa_saida = -np.diag(Q)
    registros: list[dict] = []

    for empresa in range(1, n_empresas + 1):
        estado = int(rng.choice(n_estados - 1, p=pesos))
        # Entrada escalonada: nem toda empresa está na base desde o dia zero.
        t = float(rng.uniform(0.0, anos * 0.35))
        registros.append({"ID": empresa, "Data": t, "Estado": estado})

        while True:
            if estado == absorvente or taxa_saida[estado] <= 0:
                break
            t = t + float(rng.exponential(1.0 / taxa_saida[estado]))
            if t > anos:
                break
            probs = Q[estado].copy()
            probs[estado] = 0.0
            probs = probs / probs.sum()
            estado = int(rng.choice(n_estados, p=probs))
            registros.append({"ID": empresa, "Data": t, "Estado": estado})

    historico = pd.DataFrame.from_records(registros)
    historico["Rating"] = [RATINGS[e] for e in historico["Estado"]]
    return historico.sort_values(["ID", "Data"], ignore_index=True)[
        ["ID", "Data", "Rating", "Estado"]
    ]


# --------------------------------------------------------------------------
# Capítulo 4 — previsão de taxas de default
# --------------------------------------------------------------------------

#: Coeficientes verdadeiros do modelo de taxa de default agregada, na escala
#: logit. A taxa observada é uma frequência binomial em torno dessa taxa.
COEFS_TAXA: dict[str, float] = {
    "CONST": -6.00,
    "SPR": 0.42,      # spread de crédito, em pontos percentuais
    "PRF": 0.030,     # proporção de emissores de baixa qualidade, em %
    "PIB": -0.140,    # crescimento do PIB, em %
}

PREDITORES_CAP04: list[str] = ["SPR", "PRF", "PIB"]


def gerar_series_macro(
    n_anos: int = 45, ano_inicial: int = 1981, semente: int = 42
) -> pd.DataFrame:
    """Gera a série anual de taxa de default agregada e de fatores explicativos.

    A taxa latente segue um logit nos fatores, com um choque persistente que
    representa tudo o que move o ciclo de crédito e não está no modelo. A taxa
    **observada** é a frequência de default de um universo finito de emissores —
    ou seja, carrega ruído binomial além do ruído do próprio ciclo.

    Essa distinção é o que separa o capítulo 4 de uma regressão qualquer: parte
    do que se tenta explicar é ruído amostral, e nenhum modelo explica ruído.

    Returns
    -------
    pandas.DataFrame
        Colunas ``Ano``, ``SPR`` (spread de crédito, p.p.), ``PRF`` (% de
        emissores de baixa qualidade), ``PIB`` (crescimento real, %),
        ``N_emissores``, ``Defaults`` e ``IDR`` (taxa observada, em fração).
    """
    rng = np.random.default_rng(semente)

    def ar1(rho: float, media: float, desvio: float) -> np.ndarray:
        x = np.empty(n_anos)
        x[0] = media + rng.normal(0.0, desvio)
        for t in range(1, n_anos):
            x[t] = media + rho * (x[t - 1] - media) + rng.normal(
                0.0, desvio * np.sqrt(1 - rho**2)
            )
        return x

    SPR = ar1(0.62, 3.4, 1.15)
    PRF = ar1(0.80, 42.0, 7.0)
    PIB = ar1(0.30, 2.6, 2.3)

    # Choque persistente: o ciclo que o modelo não observa.
    choque = ar1(0.45, 0.0, 0.34)

    indice = (
        COEFS_TAXA["CONST"]
        + COEFS_TAXA["SPR"] * SPR
        + COEFS_TAXA["PRF"] * PRF
        + COEFS_TAXA["PIB"] * PIB
        + choque
    )
    taxa_latente = 1.0 / (1.0 + np.exp(-indice))

    N = rng.integers(700, 1400, size=n_anos)
    defaults = rng.binomial(N, taxa_latente)

    return pd.DataFrame(
        {
            "Ano": np.arange(ano_inicial, ano_inicial + n_anos),
            "SPR": SPR,
            "PRF": PRF,
            "PIB": PIB,
            "N_emissores": N,
            "Defaults": defaults,
            "IDR": defaults / N,
            "taxa_latente": taxa_latente,
        }
    )


# --------------------------------------------------------------------------
# Capítulo 5 — perda dada o default (LGD)
# --------------------------------------------------------------------------

SENIORIDADES: list[str] = ["Sr. Sec.", "Sr. Unsec.", "Sub."]

#: Coeficientes verdadeiros da média condicional da LGD, na escala logit.
COEFS_LGD: dict[str, float] = {
    "CONST": -0.55,
    "Sr. Unsec.": 0.80,   # efeito frente à referência (Sr. Sec.)
    "Sub.": 1.55,
    "LEV": 1.20,          # alavancagem do emissor
    "COB": -2.60,         # cobertura por garantia (valor da garantia / exposição)
    "CICLO": 0.85,        # sensibilidade ao fator sistêmico anual
}

#: Precisão da beta: quanto menor, mais massa nos extremos 0 e 1.
PRECISAO_LGD: float = 4.5


def gerar_lgd(
    n_anos: int = 18, n_por_ano: int = 90, ano_inicial: int = 2008, semente: int = 42
) -> pd.DataFrame:
    """Gera observações de LGD com senioridade, alavancagem e fator de ciclo.

    A LGD é sorteada de uma distribuição beta cuja média depende das
    covariáveis. Com precisão baixa, a beta acumula massa perto de 0 e de 1 —
    o formato bimodal que qualquer base de recuperação real apresenta e que
    nenhuma regressão linear reproduz.

    O fator sistêmico anual é o mesmo que desloca a frequência de default: anos
    ruins têm mais defaults **e** recuperações piores. Essa correlação é a razão
    de existir a exigência de LGD de *downturn*.

    Returns
    -------
    pandas.DataFrame
        Colunas ``Ano``, ``ID``, ``Senioridade``, ``LEV`` (alavancagem), ``COB``
        (cobertura por garantia), ``CICLO``, ``LGD`` e ``taxa_default_ano``.
    """
    rng = np.random.default_rng(semente)

    # Fator sistêmico: valores altos = ano ruim.
    ciclo = np.empty(n_anos)
    ciclo[0] = rng.normal(0.0, 1.0)
    for t in range(1, n_anos):
        ciclo[t] = 0.5 * ciclo[t - 1] + rng.normal(0.0, np.sqrt(1 - 0.25))

    registros = []
    for t in range(n_anos):
        ano = ano_inicial + t
        senioridade = rng.choice(SENIORIDADES, size=n_por_ano, p=[0.30, 0.45, 0.25])
        LEV = np.clip(rng.normal(0.42, 0.16, n_por_ano), 0.03, 0.95)
        # Cobertura por garantia: boa parte das operações não tem nenhuma, e as
        # que têm variam de parcial a sobregarantida (alienação fiduciária).
        COB = np.where(
            rng.random(n_por_ano) < 0.45,
            0.0,
            np.clip(rng.gamma(2.2, 0.34, n_por_ano), 0.0, 1.6),
        )

        indice = np.full(n_por_ano, COEFS_LGD["CONST"])
        indice += COEFS_LGD["LEV"] * LEV
        indice += COEFS_LGD["COB"] * COB
        indice += COEFS_LGD["CICLO"] * ciclo[t]
        for nome in ["Sr. Unsec.", "Sub."]:
            indice += COEFS_LGD[nome] * (senioridade == nome)

        media = 1.0 / (1.0 + np.exp(-indice))
        a = media * PRECISAO_LGD
        b = (1.0 - media) * PRECISAO_LGD
        lgd = rng.beta(a, b)

        # Taxa de default do ano, movida pelo mesmo fator sistêmico.
        taxa = 1.0 / (1.0 + np.exp(-(-4.0 + 0.75 * ciclo[t])))

        for i in range(n_por_ano):
            registros.append(
                {
                    "Ano": ano,
                    "ID": t * n_por_ano + i + 1,
                    "Senioridade": senioridade[i],
                    "LEV": float(LEV[i]),
                    "COB": float(COB[i]),
                    "CICLO": float(ciclo[t]),
                    "LGD": float(lgd[i]),
                    "taxa_default_ano": float(taxa),
                }
            )
    return pd.DataFrame.from_records(registros)


# --------------------------------------------------------------------------
# Capítulos 6 e 7 — correlação de ativos e risco de carteira
# --------------------------------------------------------------------------

#: Parâmetros verdadeiros do modelo de fator único usado nos capítulos 6 e 7.
PARAMS_FATOR: dict[str, float] = {
    "PD": 0.0150,   # probabilidade de default incondicional
    "RHO": 0.1200,  # correlação de ativos
}


def gerar_taxas_vasicek(
    n_anos: int = 25,
    n_obrigados: int = 1000,
    pd_incondicional: float | None = None,
    rho: float | None = None,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera taxas de default anuais de um modelo de fator único (Vasicek).

    O valor do ativo padronizado do devedor ``i`` no ano ``t`` é

    .. math:: Z_{it} = \\sqrt{\\rho}\\,X_t + \\sqrt{1-\\rho}\\,\\varepsilon_{it},

    com :math:`X_t` o fator sistêmico e :math:`\\varepsilon_{it}` o risco
    idiossincrático, ambos normais padrão e independentes. Há default quando
    :math:`Z_{it}` cai abaixo do limiar :math:`\\Phi^{-1}(PD)`.

    A correlação :math:`\\rho` é a única fonte de dependência entre devedores —
    e é ela que impede que a perda de uma carteira grande seja determinística.

    Returns
    -------
    pandas.DataFrame
        Colunas ``Ano``, ``X`` (fator realizado, não observável na prática),
        ``taxa_condicional``, ``N``, ``Defaults`` e ``taxa_observada``.
    """
    from scipy import stats as _st

    p = PARAMS_FATOR["PD"] if pd_incondicional is None else pd_incondicional
    r = PARAMS_FATOR["RHO"] if rho is None else rho
    rng = np.random.default_rng(semente)

    limiar = _st.norm.ppf(p)
    X = rng.normal(0.0, 1.0, size=n_anos)
    taxa_condicional = _st.norm.cdf((limiar - np.sqrt(r) * X) / np.sqrt(1 - r))
    defaults = rng.binomial(n_obrigados, taxa_condicional)

    return pd.DataFrame(
        {
            "Ano": np.arange(1, n_anos + 1),
            "X": X,
            "taxa_condicional": taxa_condicional,
            "N": n_obrigados,
            "Defaults": defaults,
            "taxa_observada": defaults / n_obrigados,
        }
    )


def gerar_carteira(
    n_obrigados: int = 2000,
    homogenea: bool = False,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera uma carteira de crédito com exposição, PD, LGD e correlação.

    Com ``homogenea=True``, todas as posições têm a mesma exposição e a mesma
    PD — o caso em que a fórmula analítica de Vasicek vale exatamente no limite,
    e portanto o caso em que a simulação pode ser conferida contra resposta
    fechada.

    Com ``homogenea=False``, a exposição segue uma distribuição bastante
    assimétrica: poucas posições grandes respondem por parcela desproporcional
    do total. É a concentração que existe em qualquer carteira corporativa real
    e que a fórmula analítica ignora.
    """
    rng = np.random.default_rng(semente)
    p = PARAMS_FATOR

    if homogenea:
        ead = np.full(n_obrigados, 1_000.0)
        pd_i = np.full(n_obrigados, p["PD"])
        lgd = np.full(n_obrigados, 0.45)
    else:
        ead = rng.lognormal(mean=6.4, sigma=1.15, size=n_obrigados)
        pd_i = np.clip(rng.lognormal(mean=np.log(p["PD"]), sigma=0.85, size=n_obrigados),
                       1e-4, 0.35)
        lgd = np.clip(rng.beta(4.0, 5.0, size=n_obrigados), 0.05, 0.95)

    return pd.DataFrame(
        {
            "ID": np.arange(1, n_obrigados + 1),
            "EAD": ead,
            "PD": pd_i,
            "LGD": lgd,
            "RHO": p["RHO"],
        }
    )
