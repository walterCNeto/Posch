"""Monta o notebook do capítulo 11 (risco de crédito estruturado — CDOs)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 11 — Crédito estruturado

## O problema

Uma carteira de crédito corporativo de qualidade média tem perda esperada de
alguns por cento. Ninguém a compraria como se fosse um título AAA.

A securitização propõe uma alquimia: fatiar essa carteira em tranches com ordem
de prioridade. Quem compra a tranche mais júnior absorve os primeiros prejuízos;
quem compra a sênior só perde depois que todos os subordinados foram consumidos.
Com subordinação suficiente, a tranche sênior fica de fato muito segura — perda
esperada de frações de ponto-base, compatível com classificação AAA.

**A alquimia funciona.** A matemática está correta. O capítulo vai reproduzi-la.

O problema é outro, e é o assunto real deste capítulo: a segurança da tranche
sênior depende inteiramente de um parâmetro que o capítulo 6 mostrou ser o pior
estimado de todo o curso. E a dependência não é linear — é catastroficamente
convexa.

Este capítulo fecha o arco que começou no capítulo 6."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.structured.cdo import (
    correlacao_implicita,
    el_tranche_lhp,
    el_tranche_mc,
    estrutura_padrao,
    perda_tranche,
    sensibilidade_a_correlacao,
    simular_perdas_carteira,
    tabela_tranches,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.6f}")

PD = 0.015
RHO = 0.12
LGD = 0.60"""
    ),
    md(
        """## A mecânica da tranche

A tranche entre os pontos de *attachment* $A$ e *detachment* $D$ absorve a
parcela da perda da carteira que cai naquela faixa:

$$
L_{\\text{tranche}} = \\frac{\\min(L, D) - \\min(L, A)}{D - A}.
$$

Ela é zero enquanto a perda não atinge $A$, cresce linearmente entre $A$ e $D$, e
satura em 1 depois de $D$. Repare no formato: é o payoff de uma **trava de
opções** sobre a perda da carteira. Isso não é analogia — é a razão de tudo que
vem a seguir. Opções são não-lineares, e não-linearidade transforma incerteza de
parâmetro em incerteza desproporcional de preço."""
    ),
    code(
        """perdas = np.linspace(0, 0.30, 400)
estrutura = estrutura_padrao()

fig, ax = plt.subplots()
for _, linha in estrutura.iterrows():
    ax.plot(perdas * 100,
            perda_tranche(perdas, linha["attach"], linha["detach"]) * 100,
            lw=2, label=f"{linha['tranche']} ({linha['attach']:.0%}–{linha['detach']:.0%})")
ax.set_title("Perda da tranche em função da perda da carteira")
ax.set_xlabel("perda da carteira (%)")
ax.set_ylabel("perda da tranche (% do seu tamanho)")
ax.legend(fontsize=8)
plt.show()

estrutura"""
    ),
    md(
        """## A alquimia, medida

Com a carteira do modelo de fator único — PD de 1,5%, correlação de 0,12, LGD de
60% — a perda esperada de cada tranche sai da integral sobre o fator sistêmico."""
    ),
    code(
        """tabela = tabela_tranches(PD, RHO, LGD)
tabela["EL (bps)"] = tabela["EL"] * 1e4
tabela[["tranche", "attach", "detach", "EL", "EL (bps)", "spread (bps)"]].round(4)"""
    ),
    md(
        """A promessa se cumpre. A carteira tem perda esperada de 0,9% — nada que
mereça grau de investimento alto. A tranche sênior tem perda esperada de menos de
um ponto-base, e a supersênior é indistinguível de zero.

A partir de uma carteira medíocre, o modelo produz títulos aparentemente
excelentes. Nenhum truque contábil: é subordinação genuína.

Antes de seguir, vale confirmar que a fórmula está certa, comparando com
simulação direta."""
    ),
    code(
        """perdas_sim = simular_perdas_carteira(
    n_obrigados=3000, pd_inc=PD, rho=RHO, lgd=LGD,
    n_simulacoes=120_000, semente=1,
)

comparacao = []
for _, linha in estrutura.head(3).iterrows():
    a, d = linha["attach"], linha["detach"]
    comparacao.append({
        "tranche": linha["tranche"],
        "Monte Carlo": el_tranche_mc(perdas_sim, a, d),
        "fórmula LHP": el_tranche_lhp(a, d, PD, RHO, LGD),
    })
comparacao = pd.DataFrame(comparacao).set_index("tranche")
comparacao["diferença"] = comparacao["Monte Carlo"] / comparacao["fórmula LHP"] - 1
comparacao.round(5)"""
    ),
    md(
        """As três primeiras tranches concordam dentro de poucos por cento — a fórmula
está certa e a simulação também.

A tabela para nas três primeiras de propósito. Nas tranches sênior, com cento e
vinte mil cenários, pouquíssimos chegam a atingir a faixa, e a estimativa de
Monte Carlo passa a ter erro relativo grande demais para servir de comparação. A
fórmula LHP não sofre disso, porque integra analiticamente sobre o fator em vez
de esperar o sorteio produzir a catástrofe.

Guarde a observação: **a tranche sênior é justamente a que a simulação estima
pior**, e é a que carrega o maior nocional numa estrutura real.

## O que a correlação faz

Agora o parâmetro. Vamos variar $\\rho$ mantendo tudo o mais constante — mesma
carteira, mesma PD, mesma LGD."""
    ),
    code(
        """rhos = np.linspace(0.01, 0.60, 45)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

for a, d, nome in [(0.00, 0.03, "equity"), (0.03, 0.07, "júnior mezanino")]:
    s = sensibilidade_a_correlacao(a, d, PD, LGD, rhos=rhos)
    ax1.plot(s["rho"], s["EL"] * 100, lw=2, label=nome)
ax1.set_title("Tranches júnior")
ax1.set_xlabel("correlação ρ")
ax1.set_ylabel("perda esperada (%)")
ax1.legend()

for a, d, nome in [(0.10, 0.15, "sênior"), (0.15, 0.30, "supersênior")]:
    s = sensibilidade_a_correlacao(a, d, PD, LGD, rhos=rhos)
    ax2.plot(s["rho"], s["EL"] * 1e4, lw=2, label=nome)
ax2.set_title("Tranches sênior")
ax2.set_xlabel("correlação ρ")
ax2.set_ylabel("perda esperada (bps)")
ax2.legend()
plt.show()"""
    ),
    md(
        """Os dois painéis vão em direções **opostas**, e isso é a característica
econômica central do produto.

A tranche equity **melhora** quando a correlação sobe. A sênior **piora**.

A intuição: correlação alta torna os extremos mais prováveis. Ou quase todo mundo
quebra junto, ou quase ninguém quebra. O detentor da equity, que perde tudo em
qualquer cenário mediano, se beneficia de haver chance real de nada acontecer. O
detentor da sênior, que só é atingido em catástrofe, sofre com o aumento da
probabilidade de catástrofe.

Correlação zero, por outro lado, é o paraíso da tranche sênior: com defaults
independentes e uma carteira grande, a perda é praticamente determinística em
0,9% e **nunca** alcança 10%. A sênior seria literalmente livre de risco.

## Onde a coisa quebra

O capítulo 6 estimou $\\rho$ com vinte e cinco anos de dados e chegou a um
intervalo de confiança de 95% entre 0,056 e 0,177. Vamos aplicar exatamente esse
intervalo a esta estrutura."""
    ),
    code(
        """IC_INFERIOR, PONTUAL, IC_SUPERIOR = 0.056, 0.098, 0.177

linhas = []
for _, linha in estrutura.iterrows():
    a, d = linha["attach"], linha["detach"]
    els = [el_tranche_lhp(a, d, PD, r, LGD)
           for r in (IC_INFERIOR, PONTUAL, IC_SUPERIOR)]
    linhas.append({
        "tranche": linha["tranche"],
        "ρ = 0,056": els[0],
        "ρ = 0,098": els[1],
        "ρ = 0,177": els[2],
        "razão sup/inf": els[2] / els[0] if els[0] > 1e-14 else np.inf,
    })
pd.DataFrame(linhas).set_index("tranche").round(6)"""
    ),
    md(
        """Esta tabela é o capítulo inteiro.

**Dentro do mesmo intervalo de confiança**, estimado com vinte e cinco anos de
dados no capítulo 6:

- a equity praticamente não se move — a razão entre os extremos é próxima de 1;
- o júnior mezanino varia por uma ordem de grandeza;
- a sênior varia por **quatro ou cinco ordens de grandeza**.

A mesma incerteza de parâmetro que produzia um fator de 2,8× no capital de
carteira no capítulo 7 produz um fator de dezenas de milhares na perda esperada
da tranche sênior.

E o ponto que importa: **não é possível estimar $\\rho$ melhor.** O intervalo do
capítulo 6 não vem de preguiça metodológica — vem de haver vinte e cinco
observações do fator sistêmico. A tranche sênior exige precisão em um parâmetro
que o dado disponível não determina.

É por isso que classificar tranches sênior como AAA foi um erro estrutural, e não
um erro de calibragem que uma agência mais cuidadosa teria evitado. O rating de
uma tranche sênior é uma afirmação sobre a terceira casa decimal de um parâmetro
conhecido na primeira.

## O modelo rejeitado pelos próprios preços

Existe um teste ainda mais direto, e ele estava disponível antes de 2008.

Se o modelo de fator único fosse adequado, a mesma correlação deveria explicar o
preço de **todas** as tranches da mesma carteira — os devedores são os mesmos, a
correlação entre eles é a mesma. Invertendo cada preço observado para achar o
$\\rho$ que o reproduz — a *correlação implícita*, análoga à volatilidade
implícita —, os valores deveriam coincidir."""
    ),
    code(
        """# Preços de mercado hipotéticos, no padrão observado antes da crise:
# tranches das pontas negociadas acima do que o modelo prevê.
precos = []
for _, linha in estrutura.head(4).iterrows():
    a, d = linha["attach"], linha["detach"]
    modelo = el_tranche_lhp(a, d, PD, RHO, LGD)
    fator = 1.5 if (a == 0.0 or a >= 0.10) else 1.0
    precos.append({
        "tranche": linha["tranche"],
        "attach": a,
        "detach": d,
        "EL do modelo": modelo,
        "EL de mercado": modelo * fator,
    })
precos = pd.DataFrame(precos)
precos["ρ implícito"] = [
    correlacao_implicita(linha["EL de mercado"], linha["attach"],
                         linha["detach"], PD, LGD)
    for _, linha in precos.iterrows()
]
precos.set_index("tranche").round(6)"""
    ),
    md(
        """Duas coisas aparecem, e a segunda é fatal.

As correlações implícitas **não coincidem** entre tranches: as fatias
intermediárias, precificadas pelo modelo, devolvem exatamente o $\\rho$ de
partida, enquanto a sênior exige um valor mais alto para justificar seu preço. O
mesmo conjunto de devedores exigiria correlações diferentes conforme a fatia que
se olha. É o fenômeno que o mercado batizou de sorriso de correlação, por
analogia com o sorriso de volatilidade em opções. Aqui ele é modesto porque os
preços do exemplo são modestos; nos dados de mercado antes da crise, a diferença
entre as pontas era muito maior.

O segundo achado não admite ajuste: a equity precificada acima do modelo devolve
`NaN`. **Nenhuma correlação reproduz aquele preço.** A perda esperada da equity é
decrescente em $\\rho$, com máximo em $\\rho$ próximo de zero; um preço acima desse
teto está fora do alcance do modelo, qualquer que seja o parâmetro escolhido.

Isso é uma rejeição limpa. Não é que o parâmetro esteja mal calibrado — é que o
modelo não consegue gerar os preços que o mercado pratica. E era observável a
qualquer momento, sem esperar nenhuma crise, apenas confrontando o modelo com a
grade de preços que ele mesmo deveria explicar.

A resposta do mercado à época foi tratar a correlação implícita como um objeto a
ser cotado — uma superfície, com correlação base por tranche — em vez de tratar a
discordância como evidência contra o modelo. Precificar consistentemente é
diferente de precificar corretamente: uma superfície interpolada garante que
posições fechem entre si, e não diz nada sobre o risco de cauda.

## O que quebra fora do laboratório

**Dependência de cauda nula.** A cópula gaussiana implícita no modelo tem uma
propriedade específica: eventos extremos conjuntos são assintoticamente
independentes. Na prática, créditos ruins quebram juntos com mais frequência do
que qualquer $\\rho$ gaussiano permite. Cópulas com dependência de cauda — $t$ de
Student, por exemplo — engordam a perda da sênior substancialmente com os mesmos
parâmetros de primeira ordem.

**Carteira homogênea.** A aproximação LHP supõe devedores idênticos e infinitos.
O capítulo 7 mostrou o efeito da concentração no quantil; em tranche sênior, o
efeito é maior, porque a cauda é tudo que importa.

**Correlação não é constante.** Há evidência de que a dependência aumenta em
crises — exatamente quando a tranche sênior é testada.

**Ressecuritização.** Tranches de CDO empacotadas em novos CDOs multiplicam a
sensibilidade à correlação, porque compõem duas camadas de não-linearidade. Foi
o instrumento que mais destruiu valor por unidade de nocional em 2008.

**Risco de modelo é o risco dominante.** Numa tranche sênior, o risco de o modelo
estar errado é maior que o risco de crédito que ele mede. Isso inverte a
hierarquia usual de preocupações.

## Ponte regulatória

**Tratamento de securitização é deliberadamente conservador.** As abordagens de
capital para exposições securitizadas impõem pisos e penalidades que não se
justificam pela perda esperada modelada. Este capítulo explica a razão: a perda
esperada modelada não é confiável na faixa sênior, e o piso protege contra o
risco de modelo, não contra o risco de crédito.

**Ressecuritização recebe tratamento mais duro**, pelo mesmo motivo, elevado ao
quadrado.

**Retenção de risco.** Exigir que o originador retenha parte da estrutura ataca o
problema de incentivo — quem estrutura conhece a carteira melhor que quem compra
a tranche sênior.

**Validação de modelo de tranche.** Um relatório que reporta a perda esperada da
tranche sênior sem a tabela de sensibilidade a $\\rho$ deste capítulo está
omitindo a única informação que importa. A recomendação prática é simples:
reporte sempre a faixa, nunca o ponto.

**A lição transferível.** Sempre que um produto tem payoff convexo em um
parâmetro mal estimado, a incerteza do parâmetro domina o resultado. Vale para
tranches, para garantias, para qualquer estrutura com gatilho. A pergunta de
validação não é "qual o valor do parâmetro?", é "quanto o resultado muda dentro
do intervalo em que o parâmetro pode estar?".

## Exercícios

1. Refaça a tabela de sensibilidade com PD de 3% em vez de 1,5%. A tranche sênior
   fica mais ou menos sensível à correlação em termos relativos?

2. Qual subordinação seria necessária para que a perda esperada da tranche sênior
   ficasse abaixo de 1 ponto-base **para todo** $\\rho$ dentro do intervalo de
   confiança do capítulo 6? Compare com a subordinação de 10% da estrutura padrão.

3. Substitua a normal por uma $t$ de Student com 5 graus de liberdade no fator
   sistêmico e recalcule a tabela de tranches. Quanto muda a sênior? E a equity?

4. Monte um CDO de tranches mezanino de outros CDOs (ressecuritização) e calcule
   a sensibilidade a $\\rho$ da tranche sênior resultante. Compare com a
   sensibilidade da sênior de um CDO simples."""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap11_cdos.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
