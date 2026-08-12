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
