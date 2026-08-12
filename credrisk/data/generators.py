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
