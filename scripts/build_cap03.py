"""Monta o notebook do capítulo 3 (matrizes de transição de rating)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 3 — Matrizes de transição

## O problema

Os dois primeiros capítulos estimaram uma probabilidade de default em um
horizonte fixo, tipicamente um ano. Mas crédito não vence em um ano. Uma
debênture de sete anos, um financiamento de projeto de doze, uma carteira
consignada de quarenta e oito meses — todos exigem saber o que acontece **ao
longo do tempo**, e não só no primeiro ano.

Além disso, default não é o único evento que importa. Um título que migra de
BBB para B não deu calote, mas perdeu valor de mercado, consome mais capital e
pode disparar covenant. Para provisão sob perda esperada ao longo da vida, para
precificação de crédito estruturado e para capital econômico, é preciso modelar
**migração**, não só default.

A matriz de transição responde a isso: a probabilidade de sair de cada rating e
chegar a cada outro dentro de um horizonte. Este capítulo mostra duas maneiras
de estimá-la — e por que a mais simples produz um número que é indefensável."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""
    ),
    md(
        """## Os dados

O histórico sintético segue uma cadeia de Markov em tempo contínuo com matriz
geradora conhecida. Cada empresa entra na base em um momento próprio, migra em
tempos aleatórios, e sai por default ou por censura no fim da janela.

Isso reproduz o formato de qualquer base de rating real: uma linha por evento de
mudança, não um painel balanceado."""
    ),
    code(
        """historico = gerar_historico_ratings()

print(f"{len(historico):,} observações de rating · "
      f"{historico['ID'].nunique():,} empresas · janela de 15 anos")
print(f"defaults observados: {(historico['Rating'] == 'D').sum()}")
historico.head(8)"""
    ),
    md(
        """## A formulação em tempo contínuo

A cadeia é descrita por uma **matriz geradora** $Q$, cujo elemento fora da
diagonal $q_{ij}$ é a intensidade instantânea de migrar de $i$ para $j$, medida
em migrações por ano. As linhas somam zero, e o estado de default é absorvente
— dele não se sai.

A matriz de transição para um horizonte $t$ qualquer é a exponencial de matriz

$$
P(t) = \\exp(Qt) = \\sum_{k=0}^{\\infty} \\frac{(Qt)^k}{k!}.
$$

Essa formulação tem uma propriedade que a versão discreta não tem de graça: a
consistência entre horizontes. Vale $P(s)P(t) = P(s+t)$ para quaisquer $s$ e $t$,
inclusive fracionários. Estimada a geradora, a matriz de 6 meses, de 3 anos ou
de 7,5 anos sai sem reestimar nada.

## Os dois estimadores

**Coorte.** Olhe onde cada empresa estava em 1º de janeiro e onde estava em 31 de
dezembro. Conte. Divida. É o que as agências publicam e o que quase toda área de
risco calcula.

**Duração.** Some o tempo total que a carteira passou em cada rating e conte
todas as migrações observadas, com data. A intensidade estimada é

$$
\\hat q_{ij} = \\frac{N_{ij}}{T_i},
$$

o número de migrações de $i$ para $j$ dividido pelos anos-empresa de exposição
em $i$. É o estimador de máxima verossimilhança da cadeia em tempo contínuo.

A diferença parece técnica. Não é."""
    ),
    code(
        """P_coorte = matriz_coorte(historico)
Q_duracao = gerador_duracao(historico)
P_duracao = matriz_do_gerador(Q_duracao, 1.0)

print("Matriz de coorte — probabilidades de transição em 1 ano (%)")
(P_coorte * 100).round(3)"""
    ),
    code(
        """print("Matriz por duração — probabilidades de transição em 1 ano (%)")
(P_duracao * 100).round(3)"""
    ),
    md(
        """## O zero que não deveria estar lá

Olhe a coluna `D` — probabilidade de default em um ano — nas duas matrizes, ao
lado da verdade que o gerador conhece."""
    ),
    code(
        """P_verdadeira = matriz_do_gerador(gerador_verdadeiro(), 1.0)

colunaD = pd.DataFrame({
    "verdadeira": P_verdadeira["D"] * 100,
    "coorte": P_coorte["D"] * 100,
    "duração": P_duracao["D"] * 100,
})
colunaD.round(4)"""
    ),
    md(
        """A matriz de coorte afirma que a probabilidade de uma empresa AAA entrar
em default em um ano é **exatamente zero**. E o mesmo para AA.

Isso não é uma estimativa pequena. É uma afirmação de impossibilidade. E ela não
vem dos dados — vem do estimador: nenhuma empresa AAA quebrou nesta janela, e o
estimador de coorte não tem outro jeito de dizer "raro" além de dizer "nunca".

O estimador de duração, olhando os mesmos dados, devolve um número pequeno e
positivo. Ele consegue porque não precisa observar AAA → D diretamente: observa
AAA → AA, AA → A, e assim por diante, e a exponencial de matriz compõe os
caminhos. **A probabilidade de default de um AAA é inferida pelo caminho, não
pelo evento.**

Se você já viu uma planilha de risco com zeros na parte de cima da coluna de
default, viu esse problema. Ele importa porque:

- provisão de perda esperada com PD zero é provisão zero, para sempre;
- capital regulatório tem piso de PD justamente por causa disso;
- precificação de tranche sênior de CDO com PD zero no ativo subjacente é como o
  mercado precificou boa parte do que quebrou em 2008 — assunto do capítulo 11.

Vale conferir se a vantagem se sustenta ou se é só essa célula:"""
    ),
    code(
        """zeros = pd.Series({
    "verdadeira": int((P_verdadeira.to_numpy() == 0).sum()),
    "coorte": int((P_coorte.to_numpy() == 0).sum()),
    "duração": int((P_duracao.to_numpy() == 0).sum()),
}, name="células iguais a zero")

erro = pd.Series({
    "coorte": np.abs(P_coorte.to_numpy() - P_verdadeira.to_numpy()).mean(),
    "duração": np.abs(P_duracao.to_numpy() - P_verdadeira.to_numpy()).mean(),
}, name="erro absoluto médio")

print(zeros.to_string())
print()
print(erro.round(6).to_string())"""
    ),
    md(
        """Duas leituras, e a segunda é a que costuma ser omitida.

A duração reproduz **exatamente** a estrutura de esparsidade verdadeira — mesmo
número de zeros, e nos mesmos lugares. A coorte inventa zeros que não existem.

Mas o erro absoluto médio dos dois estimadores é praticamente **igual**. A
vantagem da duração não está em ser mais precisa no geral: está em ser
utilizável na cauda. No miolo da matriz, onde há dados de sobra, os dois
concordam.

Isso é uma lição recorrente em risco de crédito: o ganho de um método melhor
raramente aparece na métrica agregada. Se você escolher estimador por erro
médio, vai escolher errado.

## Um horizonte qualquer, de graça

Com a geradora estimada, qualquer horizonte sai por exponencial de matriz."""
    ),
    code(
        """horizontes = [0.5, 1, 2, 3, 5, 7, 10]
curva = pd_por_horizonte(Q_duracao, horizontes) * 100

fig, ax = plt.subplots()
for rating in ["AA", "BBB", "BB", "B", "CCC"]:
    ax.plot(horizontes, curva.loc[rating], marker="o", ms=4, label=rating)
ax.set_title("Probabilidade cumulativa de default por rating")
ax.set_xlabel("horizonte (anos)")
ax.set_ylabel("PD acumulada (%)")
ax.legend(title="rating inicial")
plt.show()

curva.round(3)"""
    ),
    md(
        """Repare no formato das curvas, que é onde mora a economia do problema.

A curva de CCC é **côncava**: sobe rápido e desacelera. Quem já está mal ou
quebra logo ou se recupera; o risco está concentrado nos primeiros anos. A curva
de AA é **convexa**: quase plana no começo e acelerando. Um AA não quebra no ano
que vem, mas pode ir degradando ao longo de uma década.

Essa diferença de formato tem consequência direta em provisão sob perda esperada
ao longo da vida. Para um ativo de alta qualidade e prazo longo, a perda
esperada é dominada pelo risco de migração acumulada, não pela PD de 12 meses. É
por isso que a PD *lifetime* de um ativo bom não é a PD de um ano multiplicada
pelo prazo — a multiplicação erra a direção nos dois extremos da escala.

## E quão confiável é isso?

Toda matriz até aqui é uma estimativa pontual. A dispersão dela vem por
bootstrap, e a unidade de reamostragem é a **empresa**, não a observação — pela
razão discutida no capítulo 1."""
    ),
    code(
        """amostras = bootstrap_coorte(historico, n_reamostras=300, semente=11)

i_bbb, i_d = RATINGS.index("BBB"), RATINGS.index("D")
celula = amostras[:, i_bbb, i_d] * 100

fig, ax = plt.subplots()
ax.hist(celula, bins=25, alpha=0.85)
ax.axvline(P_verdadeira.loc["BBB", "D"] * 100, color="#c1553b", lw=2,
           label="verdadeira")
ax.axvline(P_coorte.loc["BBB", "D"] * 100, color="#1f4e5f", lw=2, ls="--",
           label="nossa amostra")
ax.set_title("Incerteza da célula BBB → D (300 reamostras de empresas)")
ax.set_xlabel("PD de 1 ano (%)")
ax.set_ylabel("reamostras")
ax.legend()
plt.show()

print(f"IC de 95%: [{np.percentile(celula, 2.5):.3f}%, "
      f"{np.percentile(celula, 97.5):.3f}%]")
print(f"verdadeira: {P_verdadeira.loc['BBB', 'D'] * 100:.3f}%")"""
    ),
    md(
        """O intervalo é largo — a razão entre o extremo superior e o inferior é
grande para uma quantidade que entra em provisão como se fosse um número exato.

E BBB é uma célula relativamente bem povoada. Nas linhas de cima da escala, o
intervalo é tão largo que a estimativa pontual carrega pouca informação.

## De onde vem a informação

Vale ver onde o dado realmente está, porque isso explica todo o resto."""
    ),
    code(
        """tempo, N = exposicao_e_transicoes(historico)

exposicao = pd.DataFrame({
    "anos-empresa": tempo,
    "migrações observadas": N.sum(axis=1),
}, index=RATINGS)
exposicao["migrações por ano-empresa"] = (
    exposicao["migrações observadas"] / exposicao["anos-empresa"].replace(0, np.nan)
)
exposicao.round(2)"""
    ),
    md(
        """A exposição se concentra no miolo da escala. As pontas — AAA e CCC —
têm pouquíssimo tempo acumulado, e é por isso que suas linhas são as mais mal
estimadas nas duas matrizes.

O estimador de duração aproveita melhor o pouco que existe, porque conta cada
migração com sua data e usa toda a exposição parcial. Uma empresa que ficou
quatro meses em CCC antes de quebrar contribui com quatro meses de exposição —
enquanto para o estimador de coorte ela pode não existir, se entrou e saiu entre
duas datas de corte.

## O que quebra fora do laboratório

**Markov é uma hipótese, não um fato.** A cadeia supõe que a probabilidade de
migrar depende só do rating atual. Dados reais mostram *momentum*: quem foi
rebaixado recentemente tem mais chance de ser rebaixado de novo. Também há
dependência do tempo de permanência no rating. Testar a hipótese de Markov é
parte de validar o modelo, não uma sofisticação opcional.

**A matriz não é estável no tempo.** Migrações são fortemente cíclicas — em
recessão, toda a matriz se desloca para baixo. Uma matriz estimada em janela
longa é uma média de regimes que talvez nunca ocorra. Condicionar a matriz ao
ciclo é o assunto do capítulo 4.

**Nem toda geradora tem raiz.** Dada uma matriz de coorte anual, nem sempre
existe uma geradora $Q$ tal que $\\exp(Q) = P$ — é o problema da incorporabilidade
(*embeddability*). Quando não existe, gerar a matriz de 6 meses tirando "raiz
quadrada" de $P$ pode produzir probabilidades negativas. Estimar $Q$ diretamente,
como fizemos, contorna o problema pela origem.

**Rating de agência não é rating interno.** A escala interna de um banco muda de
definição, de dono e de calibragem ao longo do tempo. Uma migração pode refletir
mudança de metodologia, não de risco — e nenhum estimador distingue as duas.

## Ponte regulatória

**Piso de PD.** O arcabouço de capital impõe piso à PD justamente porque
estimadores ingênuos produzem zero em graus superiores. O piso é um remendo
prudencial para uma deficiência de estimação; usar o estimador de duração ataca
a causa, mas não dispensa o piso.

**Perda esperada ao longo da vida.** A curva de PD por horizonte deste capítulo
é o insumo direto do cálculo de perda esperada *lifetime* para ativos em estágio
2. A escolha entre coorte e duração muda o número provisionado, e essa escolha
precisa estar documentada e justificada — não herdada de uma planilha.

**Migração significativa de risco.** O critério de transferência entre estágios
se apoia em comparação de risco de default entre o reconhecimento inicial e a
data-base. Isso é, literalmente, uma leitura de matriz de transição. Qual matriz
— e estimada como — deixa de ser detalhe técnico e vira decisão contábil.

**Validação.** Um relatório que apresenta a matriz sem intervalo de confiança
está apresentando meia informação. O bootstrap acima custa segundos.

## Exercícios

1. Estime a matriz de coorte com passo trimestral em vez de anual e componha
   quatro trimestres. O resultado bate com a matriz anual? Se não, o que a
   diferença revela sobre migrações que se revertem dentro do ano?

2. Divida a janela em duas metades e estime a geradora em cada uma. Quanto muda
   a coluna de default? Isso é ciclo, ruído ou as duas coisas — e como você
   distinguiria?

3. Simule uma base com apenas 300 empresas e refaça a comparação entre coorte e
   duração. A vantagem da duração cresce ou diminui quando o dado escasseia?

4. Tome a matriz de coorte anual e tente extrair sua geradora por logaritmo de
   matriz (`scipy.linalg.logm`). Apareceram elementos negativos fora da
   diagonal? O que isso diz sobre a incorporabilidade dessa matriz?"""
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook(cells=CELULAS)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap03_transicao.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
