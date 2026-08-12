"""Monta o notebook do capítulo 4 (previsão de taxas de default)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 4 — Previsão de taxas de default

## O problema

Os três primeiros capítulos estimaram risco **relativo**: quais devedores são
piores que quais. O capítulo 3 já deixou o incômodo à vista — a matriz de
transição se desloca conforme o ciclo, e uma matriz estimada em janela longa é
a média de regimes que talvez nunca ocorra.

Este capítulo encara o nível. A taxa de default do sistema varia por um fator de
dez entre o melhor e o pior ano de um ciclo. Um modelo que produz a mesma PD em
2007 e em 2009 não está errado por pouco.

E a exigência não é acadêmica. Provisão sob perda esperada requer incorporar
informação prospectiva — projeção de condições futuras, não média histórica. Um
modelo que só sabe a média dos últimos vinte anos não atende, por construção.

A pergunta operacional é direta: **dá para prever a taxa de default do ano que
vem?** E, se dá, prever melhor do que quê?"""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson

from credrisk.data.generators import COEFS_TAXA, PREDITORES_CAP04, gerar_series_macro
from credrisk.forecast.taxa_default import (
    ajustar_taxa,
    backtest_expandindo,
    ruido_binomial_esperado,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""
    ),
    md(
        """## Os dados

Quarenta e cinco anos de taxa de default agregada e três fatores: o spread de
crédito, a proporção de emissores de baixa qualidade no universo e o crescimento
do PIB.

Uma característica do gerador merece destaque agora, porque muda a leitura de
tudo o que vem depois: a taxa **observada** não é a taxa verdadeira. É a
frequência de default de um universo finito de emissores — e portanto carrega
ruído binomial além do ruído do ciclo. Parte do que tentaremos explicar é ruído
amostral, e nenhum modelo explica ruído."""
    ),
    code(
        """macro = gerar_series_macro()
observado = macro.drop(columns=["taxa_latente"])

print(f"{len(macro)} anos · taxa média {macro['IDR'].mean():.2%} · "
      f"mínima {macro['IDR'].min():.2%} · máxima {macro['IDR'].max():.2%}")
observado.head()"""
    ),
    code(
        """fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

ax1.plot(macro["Ano"], macro["IDR"] * 100, marker="o", ms=3.5, label="observada")
ax1.axhline(macro["IDR"].mean() * 100, ls="--", color="#7a8b8b",
            label="média do período")
ax1.set_title("Taxa de default agregada")
ax1.set_ylabel("% ao ano")
ax1.legend()

ax2.plot(macro["Ano"], macro["SPR"], label="spread de crédito (p.p.)")
ax2.plot(macro["Ano"], macro["PIB"], label="crescimento do PIB (%)")
ax2.set_title("Fatores")
ax2.set_xlabel("")
ax2.legend()
plt.show()"""
    ),
    md(
        """A taxa varia de menos de 1% a quase 10%. Prever com a média histórica
significaria errar por um fator de três nos dois extremos — e é exatamente nos
extremos que a provisão importa.

## A formulação

A taxa é uma fração em $(0,1)$, então regredir seu nível é convidar previsão
negativa. Modelamos a transformada logit:

$$
\\ln\\!\\left(\\frac{p_t}{1-p_t}\\right)
= \\beta_0 + \\beta_1 \\text{SPR}_t + \\beta_2 \\text{PRF}_t + \\beta_3 \\text{PIB}_t + \\varepsilon_t.
$$

A transformação faz duas coisas de uma vez: garante previsão em $(0,1)$ e torna
o efeito dos fatores multiplicativo na razão de chances — o que é mais plausível
economicamente. Um agravamento do spread que dobra a taxa de default quando ela
está em 1% dificilmente a dobra quando ela está em 8%."""
    ),
    code(
        """ajuste = ajustar_taxa(macro, PREDITORES_CAP04)
print(ajuste.summary2().tables[1].round(4).to_string())
print(f"\\nR² dentro da amostra: {ajuste.rsquared:.3f}")"""
    ),
    md(
        """Todos os fatores são significativos, com os sinais que a economia
prevê: spread alto e proporção de emissores fracos aumentam a taxa, crescimento
do PIB reduz. O $R^2$ é respeitável para uma série anual.

E, como sempre neste curso, dá para abrir o gabarito."""
    ),
    code(
        """nomes = ["CONST", *PREDITORES_CAP04]
tabela = pd.DataFrame({
    "verdadeiro": pd.Series(COEFS_TAXA)[nomes],
    "estimado": pd.Series(ajuste.params.to_numpy(), index=nomes),
    "ep": pd.Series(ajuste.bse.to_numpy(), index=nomes),
})
tabela["desvio_em_ep"] = (tabela["estimado"] - tabela["verdadeiro"]) / tabela["ep"]
tabela.round(4)"""
    ),
    md(
        """`SPR` e `PIB` são recuperados quase exatamente. `PRF` e a constante
desviam mais — `PRF` é a série mais persistente das três, e séries muito
autocorrelacionadas carregam pouca informação independente em quarenta e cinco
anos. Com uma observação por ano, o tamanho efetivo da amostra é menor que o
número de linhas.

Mas o ponto do capítulo não é o coeficiente. É a previsão. E aqui entra a
distinção que separa modelo útil de modelo bonito.

## Dentro da amostra é fácil

Um $R^2$ de 0,56 diz o quanto o modelo explica **os anos que ele usou para se
estimar**. Isso não é previsão: é ajuste. O modelo viu 2008 quando estimou os
coeficientes com que "prevê" 2008.

A avaliação honesta é fora da amostra, com janela expansível: em cada ano, o
modelo é reestimado usando só o que existiria naquele momento, e prevê o ano
seguinte. E precisa ser comparado com o que uma área de risco faria sem modelo
nenhum."""
    ),
    code(
        """bt = backtest_expandindo(macro, PREDITORES_CAP04, minimo_treino=20)
bt.resumo().round(5)"""
    ),
    code(
        """p = bt.previsoes

fig, ax = plt.subplots()
ax.plot(p["Ano"], p["observado"] * 100, marker="o", ms=4, lw=2.2, label="observada")
ax.plot(p["Ano"], p["modelo"] * 100, marker="s", ms=4, ls="--", label="modelo")
ax.plot(p["Ano"], p["media_historica"] * 100, ls=":", lw=1.8,
        label="média histórica")
ax.set_title("Previsão um passo à frente, fora da amostra")
ax.set_ylabel("taxa de default (%)")
ax.legend()
plt.show()"""
    ),
    md(
        """O modelo bate os dois comparativos: erra menos que a média histórica e
bem menos que o passeio aleatório. O ganho não é marginal.

Vale confirmar que não é sorte desta série — a mesma pergunta que o capítulo 1
ensinou a fazer."""
    ),
    code(
        """ganhos = []
for semente in range(300, 340):
    d = gerar_series_macro(semente=semente)
    ganhos.append(backtest_expandindo(d, PREDITORES_CAP04).ganho_sobre_media)
ganhos = np.array(ganhos)

print(f"ganho mediano sobre a média histórica: {np.median(ganhos):.1%}")
print(f"réplicas em que o modelo ganha:        {(ganhos > 0).mean():.0%}")"""
    ),
    md(
        """O resultado é robusto: o modelo ganha em todas as réplicas. Este é um
capítulo com desfecho positivo — nem sempre a conclusão é que o método falha.

## Quanto ainda dá para melhorar?

A pergunta que o $R^2$ nunca responde: quanto do erro que sobra é ruído
irredutível?

A taxa observada é uma frequência binomial. Mesmo conhecendo a taxa verdadeira
com precisão infinita, a frequência realizada de um universo de mil emissores
flutua em torno dela. Esse desvio é um piso que nenhum modelo atravessa."""
    ),
    code(
        """piso = ruido_binomial_esperado(macro["taxa_latente"], macro["N_emissores"])

print(f"RMSE do modelo:                  {bt.rmse_modelo:.5f}")
print(f"piso de ruído binomial:          {piso:.5f}")
print(f"o modelo está a {bt.rmse_modelo / piso:.1f}× do piso irredutível")"""
    ),
    md(
        """O modelo está a poucas vezes do piso. Isso reenquadra a conversa: a
margem de melhora existe, mas é finita e menor do que o $R^2$ de 0,56 sugere.

Essa medida raramente aparece em relatório de validação, e deveria. Sem ela,
"o modelo explica 56% da variância" não diz se o que falta é modelo ruim ou
mundo aleatório. E as duas conclusões levam a decisões opostas: uma manda
investir em modelagem, a outra manda parar de gastar com isso.

## Autocorrelação: a hipótese que costuma falhar

Séries de taxa de default têm resíduo persistente — um ano ruim tende a ser
seguido de outro. Se isso ocorre, os erros-padrão do OLS estão errados. Vale
testar em vez de supor."""
    ),
    code(
        """dw = durbin_watson(ajuste.resid)
hac = ajustar_taxa(macro, PREDITORES_CAP04, hac_lags=2)

print(f"Durbin-Watson: {dw:.3f}   (2,0 = sem autocorrelação)\\n")
pd.DataFrame({
    "coef": ajuste.params,
    "ep OLS": ajuste.bse,
    "ep Newey-West": hac.bse,
    "razão": hac.bse / ajuste.bse,
}).round(4)"""
    ),
    md(
        """O Durbin-Watson fica abaixo de 2, indicando alguma persistência, mas os
erros-padrão de Newey-West mudam pouco — para cima em alguns coeficientes, para
baixo em outros.

A razão é que boa parte da persistência da taxa já está **explicada** pelos
fatores: o spread e o PIB são eles próprios séries persistentes, e ao incluí-los
o resíduo sobra bem mais comportado que a série original. Corrigir autocorrelação
que os regressores já absorveram não muda nada.

O procedimento correto continua sendo testar. Se o resíduo fosse fortemente
autocorrelacionado e você reportasse o erro-padrão do OLS, estaria declarando
significância que não existe.

## O que quebra fora do laboratório

**Os fatores também precisam ser previstos.** Para prever a taxa de 2027 é
preciso saber o PIB de 2027. Na prática, usa-se projeção — que tem erro próprio,
e esse erro não aparece em nenhum intervalo de confiança do modelo. A incerteza
real de uma previsão condicional é maior que a reportada, às vezes muito maior.

**Quarenta e cinco anos são três ou quatro ciclos.** O tamanho efetivo da amostra
não é 45; é o número de recessões observadas. Estimar comportamento de cauda com
quatro observações de cauda é o problema central, e ele não tem solução técnica.

**Relação estável é hipótese.** A sensibilidade da taxa de default ao spread
mudou com a estrutura do mercado de crédito, com o papel dos bancos centrais e
com a composição do universo de emissores. Modelo estimado em quarenta anos supõe
que o de hoje é o mesmo de 1985.

**Agregado não é carteira.** A taxa do sistema não é a da sua carteira. A ponte
entre as duas — o modelo de fator — é o assunto dos capítulos 6 e 7.

## Ponte regulatória

**Informação prospectiva.** A perda esperada contábil exige condicionar a
estimativa a expectativas sobre condições futuras, e normalmente a mais de um
cenário com pesos. Este capítulo é a máquina que faz isso: com o modelo
estimado, basta alimentar cenários de fatores e ler as taxas correspondentes.

**Cenários e pesos são premissa, não resultado.** A escolha dos cenários e a
ponderação entre eles não sai de nenhuma regressão — é julgamento, e é onde a
governança tem de morder. Um modelo tecnicamente impecável alimentado por
cenários otimistas produz provisão insuficiente, e a validação técnica do modelo
não detecta isso.

**Backtest de previsão, não de ajuste.** Um relatório que apresenta $R^2$ dentro
da amostra como evidência de capacidade preditiva está apresentando a métrica
errada. O backtest com janela expansível deste capítulo é a evidência
apropriada, e comparar com a média histórica é o mínimo — se o modelo não bate a
média, ele não deveria estar em produção.

## Exercícios

1. Refaça o backtest usando a taxa em nível, sem transformação logit. As
   previsões saem do intervalo admissível em algum ano? O RMSE piora?

2. Estime o modelo só com dados até 2005 e projete os vinte anos seguintes sem
   reestimar. Compare com o backtest expansível. Quanto da performance vem de
   reestimar todo ano?

3. Introduza uma quebra estrutural: multiplique por dois o coeficiente de `SPR`
   na segunda metade da amostra e refaça a análise. O modelo detecta? Que teste
   você usaria para detectá-la em dado real?

4. Construa três cenários de fatores — base, adverso, severamente adverso — e
   produza a taxa de default de cada um com intervalo de previsão. Esse é,
   literalmente, o insumo de um cálculo de perda esperada prospectiva."""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap04_previsao.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
