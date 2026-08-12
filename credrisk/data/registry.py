"""Registro central das bases do curso.

Os notebooks nunca chamam um gerador diretamente: pedem a base pelo nome a
:func:`carregar`, que gera na primeira vez e reaproveita o arquivo em disco nas
seguintes. Assim o notebook fica curto e a base é idêntica entre execuções.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from credrisk.data import generators

#: Nome da base -> função geradora sem argumentos obrigatórios.
GERADORES: dict[str, Callable[[], pd.DataFrame]] = {
    "cap01_scoring": generators.gerar_painel_scoring,
}


def diretorio_cache() -> Path:
    """Diretório onde as bases geradas são guardadas.

    Pode ser redirecionado pela variável de ambiente ``CREDRISK_DATA``, útil em
    CI e em ambientes onde a raiz do projeto é somente leitura.
    """
    bruto = os.environ.get("CREDRISK_DATA")
    if bruto:
        destino = Path(bruto)
    else:
        destino = Path(__file__).resolve().parents[2] / "data" / "gerado"
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def carregar(nome: str, forcar: bool = False) -> pd.DataFrame:
    """Devolve a base ``nome``, gerando-a se ainda não estiver em cache.

    Parameters
    ----------
    nome
        Chave registrada em :data:`GERADORES`, por exemplo ``"cap01_scoring"``.
    forcar
        Se ``True``, ignora o cache e regenera a base.

    Raises
    ------
    KeyError
        Se o nome não estiver registrado. A mensagem lista as opções válidas.
    """
    if nome not in GERADORES:
        disponiveis = ", ".join(sorted(GERADORES))
        raise KeyError(f"Base desconhecida: {nome!r}. Disponíveis: {disponiveis}")

    caminho = diretorio_cache() / f"{nome}.parquet"
    if caminho.exists() and not forcar:
        return pd.read_parquet(caminho)

    base = GERADORES[nome]()
    base.to_parquet(caminho, index=False)
    return base


def listar() -> list[str]:
    """Lista os nomes de base disponíveis."""
    return sorted(GERADORES)
