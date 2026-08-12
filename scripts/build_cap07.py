"""Monta o notebook do capítulo 7 (risco de carteira de crédito)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 7 — Risco de carteira

## O problema

Os capítulos anteriores produziram parâmetros por operação: probabilidade de
default, perda dada o default, exposição, e a correlação que liga os devedores
uns aos outros. Este capítulo os agrega.

A pergunta que o banco precisa responder não é qual a perda média — essa é
provisão, e sai da multiplicação dos três primeiros. É **quão ruim pode ficar**.
Capital existe para absorver a diferença entre o ano médio e o ano péssimo, e
para dimensioná-lo é preciso a distribuição inteira de perda, não seu centro.

Três instrumentos, em ordem crescente de sofisticação e decrescente de
generalidade:

1. **Monte Carlo direto** — funciona para qualquer carteira, e é lento onde
   importa;
2. **fórmula fechada de Vasicek** — instantânea, exata apenas sob hipóteses que
   nenhuma carteira real satisfaz, e é a base do capital regulatório;
3. **Monte Carlo com amostragem por importância** — a resposta do primeiro pelo
   custo do segundo, quase.

O capítulo mede os três um contra o outro."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.data.generators import PARAMS_FATOR, gerar_carteira
from credrisk.portfolio.montecarlo import (
    contribuicao_por_posicao,
    erro_padrao_quantil,
    numero_efetivo_posicoes,
    quantil_vasicek,
    simular_perdas,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

PD_V, RHO_V = PARAMS_FATOR["PD"], PARAMS_FATOR["RHO"]
LGD_FIXA = 0.45"""
    ),
    md(
        """## A distribuição de perda

A simulação é direta: sorteia-se o fator sistêmico, depois o ruído próprio de
cada devedor; quebra quem tem valor de ativo abaixo do limiar; soma-se
`EAD × LGD` dos que quebraram. Repete-se muitas vezes.

Começamos com a carteira homogênea — todas as posições iguais — porque é o caso
em que existe resposta fechada para conferir."""
    ),
    code(
        """homogenea = gerar_carteira(n_obrigados=2000, homogenea=True)
escala = homogenea["EAD"].sum() * LGD_FIXA

mc = simular_perdas(homogenea, n_simulacoes=30_000, semente=1)

fig, ax = plt.subplots()
ax.hist(mc.perdas / escala * 100, bins=80, alpha=0.85)
ax.axvline(mc.perda_esperada / escala * 100, color="#c1553b", lw=2,
           label="perda esperada")
ax.axvline(mc.var(0.999) / escala * 100, color="#d9a441", lw=2,
           label="percentil 99,9%")
ax.set_title("Distribuição de perda da carteira")
ax.set_xlabel("perda (% da perda máxima possível)")
ax.set_ylabel("cenários")
ax.set_xlim(0, 20)
ax.legend()
plt.show()"""
    ),
    md(
        """A forma da distribuição é o argumento de todo o capítulo: fortemente
assimétrica, com a maior parte da massa abaixo da média e uma cauda direita
longa. Em mais da metade dos cenários a perda fica **abaixo** da perda esperada.

Isso significa que a experiência típica de um banco de crédito é ser
agradavelmente surpreendido — vários anos seguidos melhores que o orçado — até o
ano em que não é. Sistemas de remuneração que premiam resultado anual sem
ajustar por risco estão, na prática, premiando quem vende a cauda.

## O fechamento contra a fórmula

Sob duas hipóteses — posições idênticas e infinitas em número — a distribuição
de perda tem forma fechada e o percentil é

$$
q(\\alpha) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) + \\sqrt{\\rho}\\,\\Phi^{-1}(\\alpha)}{\\sqrt{1-\\rho}}\\right).
$$

É a fórmula por trás do requerimento de capital do IRB. Com infinitos devedores
idênticos, o risco próprio de cada um desaparece por diversificação e sobra só o
fator sistêmico — por isso basta inverter a taxa condicional no percentil
desejado.

A simulação tem de reproduzi-la. Se não reproduzir, uma das duas está errada."""
    ),
    code(
        """comparacao = []
for nivel in [0.90, 0.99, 0.999]:
    comparacao.append({
        "nível": nivel,
        "simulado": mc.var(nivel) / escala,
        "analítico": quantil_vasicek(PD_V, RHO_V, nivel),
    })
comparacao = pd.DataFrame(comparacao).set_index("nível")
comparacao["diferença"] = comparacao["simulado"] / comparacao["analítico"] - 1
comparacao.round(5)"""
    ),
    md(
        """As duas concordam dentro de poucos por cento — e o resíduo tem
explicação: a carteira simulada tem dois mil devedores, não infinitos, então
sobra um pouco de risco idiossincrático que a fórmula ignora.

Este é o tipo de conferência que separa um modelo de carteira confiável de um
que produz números. Toda implementação de Monte Carlo de crédito deveria ter
este teste rodando, porque erro de sinal, de escala ou de índice num simulador
produz distribuições que parecem perfeitamente razoáveis.

## O que a fórmula ignora: granularidade

A hipótese de infinitos devedores não é inócua. Vale medir o custo dela."""
    ),
    code(
        """def carteira_controlada(n: int, ead: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"ID": np.arange(n), "EAD": ead, "PD": PD_V, "LGD": LGD_FIXA, "RHO": RHO_V}
    )


linhas = []
for n in [25, 100, 500, 2000, 8000]:
    c = carteira_controlada(n, np.full(n, 1000.0))
    r = simular_perdas(c, 20_000, semente=3)
    linhas.append({"posições": n, "q99,9 / perda esperada": r.var(0.999) / r.perda_esperada})

granularidade = pd.DataFrame(linhas).set_index("posições")
granularidade.loc["∞ (fórmula)"] = quantil_vasicek(PD_V, RHO_V, 0.999) / PD_V
granularidade.round(2)"""
    ),
    md(
        """Com vinte e cinco posições, a perda de cauda é bem maior que a fórmula
prevê. Com oito mil, praticamente igual. A convergência é rápida no começo e
depois estagna: o ganho de diversificação entre duas mil e oito mil posições é
pequeno, porque o que sobra é risco sistêmico, e esse não diversifica.

**Nenhuma carteira diversifica o fator comum.** É por isso que existe capital.

## O que a fórmula ignora ainda mais: concentração

Contar posições é enganoso quando elas têm tamanhos muito diferentes. A medida
correta é o inverso do índice de Herfindahl, que responde: a quantas posições
*iguais* esta carteira equivale?"""
    ),
    code(
        """rng = np.random.default_rng(1)
n = 500
linhas = []
for sigma in [0.0, 0.8, 1.5, 2.2]:
    ead = np.full(n, 1000.0) if sigma == 0 else rng.lognormal(0, sigma, n)
    ead = ead / ead.sum() * (n * 1000.0)
    c = carteira_controlada(n, ead)
    r = simular_perdas(c, 20_000, semente=3)
    m = numero_efetivo_posicoes(c)
    linhas.append({
        "dispersão da exposição (σ)": sigma,
        "posições": n,
        "número efetivo": m["numero_efetivo"],
        "q99,9 / PE": r.var(0.999) / r.perda_esperada,
    })
pd.DataFrame(linhas).set_index("dispersão da exposição (σ)").round(2)"""
    ),
    md(
        """Todas as quatro carteiras têm quinhentas posições, a mesma PD, a mesma
LGD e a mesma exposição total. A única diferença é como a exposição se distribui.

A última se comporta como se tivesse **algumas dezenas** de posições, e sua
perda de cauda é mais que o dobro da carteira uniforme. Quinhentas operações no
sistema, algumas dezenas de operações em risco.

Este é o argumento técnico por trás dos limites de concentração e do ajuste de
granularidade. A fórmula do IRB, aplicada a uma carteira dessas, subestima o
capital — e subestima de forma invisível, porque a fórmula não recebe nenhuma
informação sobre distribuição de exposição. Ela recebe PD, LGD, EAD e prazo, e
nenhum desses insumos diz que a carteira está concentrada.

## Amostragem por importância

O Monte Carlo direto tem um problema aritmético na cauda: para estimar o
percentil 99,9%, apenas um em mil cenários é informativo. Com trinta mil
cenários, trinta sustentam a estimativa.

A saída é sortear de propósito mais cenários ruins e corrigir o viés com pesos.
Como quase toda perda extrema vem de realizações ruins do fator sistêmico,
desloca-se a média de $X$ para o território negativo e pondera-se cada cenário
pela razão de verossimilhanças

$$
w = \\frac{\\phi(x)}{\\phi_\\mu(x)} = \\exp\\!\\left(-\\mu x + \\tfrac{\\mu^2}{2}\\right).
$$

O estimador continua não-enviesado; muda apenas onde o esforço é gasto."""
    ),
    code(
        """pequena = gerar_carteira(n_obrigados=1000, homogenea=True)

# Referência cara, para termos contra o que comparar.
referencia = simular_perdas(pequena, 120_000, semente=99).var(0.999)

direto = erro_padrao_quantil(pequena, 5_000, 0.999, n_repeticoes=10, com_is=False)
com_is = erro_padrao_quantil(pequena, 5_000, 0.999, n_repeticoes=10, com_is=True)

resultado = pd.DataFrame({
    "Monte Carlo direto": direto,
    "com amostragem por importância": com_is,
}).T
resultado["viés vs referência"] = resultado["media"] / referencia - 1
resultado.round(4)"""
    ),
    md(
        """Duas leituras, e a segunda é a mais importante.

O erro-padrão cai por um fator relevante — cada execução do estimador com
amostragem por importância é bem mais estável. Como a variância cai com o
quadrado, o ganho equivale a rodar várias vezes mais cenários no método direto.

Mas repare também no **viés**: o Monte Carlo direto com poucos cenários não é
apenas impreciso, ele erra sistematicamente **para baixo**. Estimar um percentil
extremo com poucas observações na cauda tende a subestimá-lo, porque os cenários
mais extremos simplesmente não foram sorteados.

Isso é pior que imprecisão. Um estimador impreciso erra para os dois lados e o
comitê percebe a instabilidade; um estimador enviesado para baixo produz capital
insuficiente de forma consistente e parece estável.

## O deslocamento importa, e não é o que a intuição diz

Vale ver como o ganho depende da escolha do deslocamento."""
    ),
    code(
        """linhas = []
for mu in [-0.5, -1.0, -1.5, -2.0, -3.0]:
    r = erro_padrao_quantil(pequena, 5_000, 0.999, n_repeticoes=8,
                            com_is=True, deslocamento=mu)
    linhas.append({
        "deslocamento μ": mu,
        "média": r["media"],
        "erro-padrão": r["erro_padrao"],
        "viés vs referência": r["media"] / referencia - 1,
    })
pd.DataFrame(linhas).set_index("deslocamento μ").round(4)"""
    ),
    md(
        """O melhor deslocamento é bem menor em magnitude do que a intuição
sugere. O percentil 99,9% corresponde a um fator em torno de $-3{,}1$, então
parece natural deslocar para lá. Mas isso é o ótimo para estimar a
**probabilidade** de ultrapassar um limite, não para estimar o **quantil**.

Para o quantil é preciso massa **em torno** do ponto de interesse, não além
dele. Deslocar demais joga quase tudo para a cauda extrema, esvazia a região
onde o percentil está e a variância volta a subir.

Este foi um erro meu na primeira versão: escolhi $\\mu = -2{,}5$ por raciocínio
plausível e obtive redução de variância de apenas 1,4×. Medindo, o ótimo em
torno de $-1{,}5$ entrega quase dez vezes mais. **Amostragem por importância mal
calibrada pode ser pior que não usar** — e, como sempre neste par de capítulos,
nada sinaliza.

## De onde vem o capital

A perda esperada por posição é fácil de decompor. A perda de cauda, não — porque
quem contribui para o cenário ruim não é quem tem PD alta, é quem tem exposição
grande **e** quebra junto com os outros."""
    ),
    code(
        """heterogenea = gerar_carteira(n_obrigados=800, homogenea=False, semente=8)
contrib = contribuicao_por_posicao(heterogenea, n_simulacoes=25_000, nivel=0.99)

contrib = contrib.sort_values("contribuicao_cauda", ascending=False)
contrib["part. na perda esperada"] = (
    contrib["perda_esperada"] / contrib["perda_esperada"].sum()
)
contrib["part. na perda de cauda"] = (
    contrib["contribuicao_cauda"] / contrib["contribuicao_cauda"].sum()
)

topo = contrib.head(20)
print(f"as 20 maiores contribuintes respondem por "
      f"{topo['part. na perda de cauda'].sum():.1%} da perda de cauda "
      f"e {topo['part. na perda esperada'].sum():.1%} da perda esperada")

medidas = numero_efetivo_posicoes(heterogenea)
print(f"\\nposições: {medidas['posicoes']} · "
      f"número efetivo: {medidas['numero_efetivo']:.0f}")
topo[["EAD", "PD", "part. na perda esperada", "part. na perda de cauda"]].head(8).round(4)"""
    ),
    md(
        """A decomposição de cauda é o instrumento que responde "de onde vem o
capital", e é ela — não a perda esperada — que deveria orientar limite por
contraparte e precificação ajustada a risco.

Uma posição pode ter contribuição pequena para a perda esperada e grande para a
perda de cauda. Precificar pelo custo de provisão, como se faz frequentemente,
cobra dessa posição bem menos do que o capital que ela consome.

## O que quebra fora do laboratório

**LGD é tratada como constante.** O capítulo 5 mostrou que ela sobe justamente
nos anos ruins. Um modelo de carteira com LGD fixa subestima a cauda — e a
correção exige tratar LGD como aleatória e correlacionada com o fator.

**A exposição também é aleatória.** Em linhas de crédito rotativas, o devedor
saca mais quando está em dificuldade. Exposição no momento do default é maior
que o saldo médio, e correlacionada com o default.

**Um fator só.** Vale aqui a mesma ressalva do capítulo 6, com consequência
maior: numa carteira com concentração setorial, o modelo de fator único mistura
diversificação que não existe.

**Cauda gaussiana.** A cópula implícita não tem dependência de cauda: eventos
extremos conjuntos são mais raros no modelo do que na realidade. O capítulo 11
mostra o preço disso em crédito estruturado.

## Ponte regulatória

**De onde vem a fórmula do IRB.** O requerimento de capital de risco de crédito
é, essencialmente, a fórmula deste capítulo aplicada a cada exposição
individualmente e somada. Isso só é legítimo porque, sob as hipóteses de
carteira infinitamente granular e fator único, a contribuição de cada exposição
ao capital não depende do resto da carteira. É essa propriedade — invariância à
carteira — que permite um requerimento aditivo por operação.

**O que a fórmula não cobre.** Concentração de nome e concentração setorial estão
fora, por construção. Por isso o arcabouço as trata em pilar separado, com
avaliação própria do banco e do supervisor. A tabela de número efetivo de
posições deste capítulo é a evidência quantitativa que essa avaliação demanda.

**Capital econômico versus regulatório.** O modelo interno de capital econômico
usa a mesma máquina sem as hipóteses simplificadoras. A diferença entre os dois
números é informativa: se o econômico é sistematicamente maior, a carteira tem
concentração que o regulatório não captura.

**Validação de simulador.** Todo modelo de carteira interno deveria ser validado
contra a resposta fechada no caso em que ela existe. É a única checagem
verdadeiramente independente disponível, e custa segundos.

## Exercícios

1. Torne a LGD aleatória e correlacionada com o fator sistêmico, usando o
   mecanismo do capítulo 5. Quanto aumenta o percentil 99,9%? Compare com o
   aumento que viria de elevar a LGD média para o valor de downturn.

2. Construa duas carteiras com a mesma perda esperada: uma com muitas operações
   de PD baixa e outra com poucas de PD alta. Qual consome mais capital? O
   resultado muda se você medir por *expected shortfall* em vez de percentil?

3. Reproduza a tabela de granularidade usando amostragem por importância e veja
   quantas simulações são necessárias para a mesma precisão do Monte Carlo
   direto. Quanto tempo isso economizaria numa carteira de cem mil operações?

4. Divida a carteira em dois setores com fatores distintos, correlacionados
   entre si a 0,4. Compare o capital com o do modelo de fator único calibrado
   para a mesma correlação média. O fator único superestima ou subestima a
   diversificação?"""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap07_carteira.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
