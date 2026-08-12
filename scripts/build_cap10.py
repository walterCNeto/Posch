"""Monta o notebook do capítulo 10 (CDS e PDs neutras ao risco)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 10 — CDS e probabilidades neutras ao risco

## O problema

Todos os capítulos anteriores estimaram probabilidades a partir de **frequências
observadas**: quantos devedores quebraram, em quantos anos, sob quais condições.
O gargalo sempre foi o mesmo — poucos eventos, poucos anos.

Existe outra fonte de informação, e ela é atualizada a cada segundo: **preços**.
O mercado de derivativos de crédito cota, continuamente, o custo de se proteger
contra o default de um emissor. Esse preço contém uma probabilidade implícita.

A tentação óbvia é usá-la. Se o CDS de cinco anos de uma empresa está em 120
pontos-base, por que estimar PD com vinte anos de dados contábeis em vez de ler
o número do mercado?

A resposta é que as duas probabilidades **não são a mesma coisa**, e a diferença
entre elas não é ruído: é prêmio de risco. Este capítulo extrai a probabilidade
implícita corretamente e depois mede a distância entre ela e a probabilidade
física dos capítulos anteriores."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.pricing.cds import (
    bootstrap_hazards,
    premio_implicito,
    spread_aproximado,
    spread_justo,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.6f}")

TAXA = 0.04
RECUPERACAO = 0.40"""
    ),
    md(
        """## A mecânica do contrato

Um CDS é um seguro. O comprador de proteção paga um prêmio periódico — o
*spread* — enquanto o emissor de referência sobrevive. Se ocorre o evento de
crédito, o vendedor paga a perda, convencionalmente $1 - R$, onde $R$ é a
recuperação.

O contrato vale zero na origem, o que dá a condição de precificação: o valor
presente do que se paga tem de igualar o valor presente do que se recebe.

$$
\\underbrace{s \\sum_i \\Delta_i\\, S(t_i)\\, D(t_i)}_{\\text{perna de prêmio}}
\\;=\\;
\\underbrace{(1-R)\\int_0^T D(u)\\,\\big(-dS(u)\\big)}_{\\text{perna de proteção}}
$$

Aqui $D$ é o fator de desconto e $S$ a probabilidade de sobrevivência. A
incógnita está dentro de $S$: a **intensidade de default** $\\lambda$, ou taxa de
hazard, que satisfaz $S(t) = \\exp\\!\\big(-\\int_0^t \\lambda(u)du\\big)$.

Note que a probabilidade extraída é neutra ao risco por construção: tudo foi
descontado à taxa livre de risco, então qualquer compensação por risco que o
mercado exija acaba absorvida por $\\lambda$."""
    ),
    md(
        """## Bootstrap da curva de hazard

Emissores líquidos têm CDS cotados em vários prazos. Cada um traz uma equação, e
resolve-se em cascata: o contrato de um ano determina a intensidade do primeiro
trecho; com ela fixa, o de três anos determina o segundo; e assim por diante.

É o mesmo raciocínio do bootstrap de curva de juros, com a intensidade de default
no lugar da taxa a termo."""
    ),
    code(
        """prazos = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
spreads_mercado = np.array([0.0045, 0.0078, 0.0110, 0.0135, 0.0160])

curva = bootstrap_hazards(prazos, spreads_mercado, TAXA, RECUPERACAO)

tabela = curva.como_tabela()
tabela["spread de entrada (bps)"] = spreads_mercado * 1e4
tabela["spread reprecificado (bps)"] = [curva.spread(p) * 1e4 for p in prazos]
tabela.round(6)"""
    ),
    md(
        """A reprecificação reproduz os spreads de entrada com precisão de máquina
— é a checagem mínima de qualquer bootstrap, e a que mais frequentemente falha
silenciosamente quando alguém troca a convenção de contagem de dias.

A curva de hazard é ascendente, o que é típico de crédito de boa qualidade: a
chance condicional de quebrar no próximo instante cresce com o horizonte, porque
há mais tempo para as coisas piorarem."""
    ),
    code(
        """t = np.linspace(0.05, 10, 300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

ax1.step(np.concatenate([[0], prazos]),
         np.concatenate([[curva.hazards[0]], curva.hazards]),
         where="pre", lw=2)
ax1.set_title("Curva de intensidade de default")
ax1.set_xlabel("anos")
ax1.set_ylabel("hazard (por ano)")

ax2.plot(t, curva.pd_acumulada(t) * 100, lw=2)
ax2.set_title("PD acumulada neutra ao risco")
ax2.set_xlabel("anos")
ax2.set_ylabel("%")
plt.show()

print(f"PD neutra ao risco em 5 anos:  {float(curva.pd_acumulada(5.0)[0]):.3%}")
print(f"PD neutra ao risco em 10 anos: {float(curva.pd_acumulada(10.0)[0]):.3%}")"""
    ),
    md(
        """## A regra de bolso, e quando ela falha

Toda mesa usa a aproximação

$$
s \\approx \\lambda\\,(1 - R),
$$

que lida de cabeça: spread de 120 bps com recuperação de 40% implica hazard de
2% ao ano.

Vale entender exatamente o quanto ela é boa, porque a resposta é mais
interessante do que "é uma aproximação grosseira"."""
    ),
    code(
        """linhas = []
for hazard in [0.005, 0.02, 0.10, 0.30]:
    exato = spread_justo(5.0, np.array([hazard]), np.array([5.0]),
                         TAXA, RECUPERACAO, por_ano=4)
    continuo = spread_justo(5.0, np.array([hazard]), np.array([5.0]),
                            TAXA, RECUPERACAO, por_ano=365)
    aprox = spread_aproximado(hazard, RECUPERACAO)
    linhas.append({
        "hazard": hazard,
        "exato trimestral (bps)": exato * 1e4,
        "exato contínuo (bps)": continuo * 1e4,
        "aproximação (bps)": aprox * 1e4,
        "erro vs trimestral": aprox / exato - 1,
    })
pd.DataFrame(linhas).set_index("hazard").round(4)"""
    ),
    md(
        """O erro é de meio por cento e **não cresce com o hazard**. Com pagamento
contínuo, some quase inteiramente.

A razão é bonita: com hazard constante, tanto a perna de proteção quanto a de
prêmio contêm o mesmo fator $\\frac{1}{r+\\lambda}\\big(1 - e^{-(r+\\lambda)T}\\big)$, que
se cancela na divisão. Sobra exatamente $\\lambda(1-R)$. A aproximação não é
aproximação — é **exata** em tempo contínuo com hazard constante, e o resíduo
observado é só a convenção de pagamento trimestral.

O que de fato quebra a regra é outra coisa: a **inclinação** da curva."""
    ),
    code(
        """prazos_inc = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
h_inclinada = np.array([0.004, 0.010, 0.020, 0.030, 0.045])

linhas = []
for i, prazo in enumerate(prazos_inc):
    exato = spread_justo(prazo, h_inclinada[: i + 1], prazos_inc[: i + 1],
                         TAXA, RECUPERACAO)
    larguras = np.diff(np.concatenate([[0.0], prazos_inc[: i + 1]]))
    h_medio = float(np.average(h_inclinada[: i + 1], weights=larguras))
    aprox = spread_aproximado(h_medio, RECUPERACAO)
    linhas.append({
        "prazo": prazo,
        "hazard médio": h_medio,
        "spread exato (bps)": exato * 1e4,
        "aproximação (bps)": aprox * 1e4,
        "erro": aprox / exato - 1,
    })
pd.DataFrame(linhas).set_index("prazo").round(4)"""
    ),
    md(
        """Com curva inclinada, a aproximação erra por dois dígitos no prazo longo
— e sempre para cima.

O motivo é que o spread não é uma média simples dos hazards: as intensidades dos
primeiros anos pesam mais, porque a probabilidade de sobrevivência ainda é alta
lá e o desconto é menor. A média simples superpondera os hazards distantes.

A lição prática: a regra de bolso é confiável para converter **um** spread em
hazard; é ruim para raciocinar sobre curva. E é exatamente para raciocinar sobre
curva que ela costuma ser usada.

## A diferença que importa: neutra ao risco versus física

Agora a pergunta do capítulo. A PD extraída do CDS pode substituir a PD estimada
dos capítulos 1 a 4?

Vamos comparar a PD implícita acima com uma PD física da ordem de grandeza de um
crédito corporativo de qualidade equivalente."""
    ),
    code(
        """pd_neutra_5a = float(curva.pd_acumulada(5.0)[0])

# PD física de referência: da ordem do que os capítulos anteriores estimariam
# para um crédito com este spread — cerca de 0,6% ao ano, acumulada em 5 anos.
pd_fisica_anual = 0.006
pd_fisica_5a = 1 - (1 - pd_fisica_anual) ** 5

comparacao = pd.Series({
    "PD neutra ao risco (5a)": pd_neutra_5a,
    "PD física estimada (5a)": pd_fisica_5a,
    "razão": pd_neutra_5a / pd_fisica_5a,
    "prêmio de risco implícito": premio_implicito(pd_neutra_5a, pd_fisica_5a, 5.0),
})
comparacao.round(4)"""
    ),
    md(
        """A probabilidade implícita no preço é substancialmente maior que a
frequência histórica esperada. A razão fica tipicamente entre duas e cinco vezes
em crédito corporativo, e sobe muito em períodos de estresse.

**A diferença não é erro de nenhum dos dois lados.** Quem vende proteção não
cobra apenas a perda esperada: cobra também por carregar risco não diversificável
— justamente o fator sistêmico dos capítulos 6 e 7 — além de liquidez e custo de
capital. Descontar tudo à taxa livre de risco empurra toda essa compensação para
dentro da probabilidade.

A consequência prática é uma regra simples e frequentemente violada:"""
    ),
    code(
        """usos = pd.DataFrame({
    "usar PD física": [
        "provisão contábil de perda esperada",
        "capital regulatório e econômico",
        "limite de crédito e alçada",
        "cálculo de perda esperada em RAROC",
    ],
    "usar PD neutra ao risco": [
        "precificação de CDS e derivativos de crédito",
        "marcação a mercado de posições",
        "ajuste de valor de crédito em derivativos",
        "avaliação relativa entre emissores",
    ],
})
usos"""
    ),
    md(
        """Usar PD neutra ao risco em provisão produz número conservador — e
errado. A provisão passa a embutir prêmio de risco de mercado, que não é perda
esperada, e passa a oscilar com o humor do mercado de crédito em vez de com a
qualidade dos devedores. O inverso — usar PD física para precificar — deixa
dinheiro na mesa e não fecha com o mercado.

## Extraindo o prêmio, não só a probabilidade

Há um uso melhor do CDS que substituir a PD física: **medir o prêmio de risco**.
Tendo estimativas independentes das duas probabilidades, a diferença entre elas é
informação sobre o apetite de risco do mercado."""
    ),
    code(
        """cenarios = pd.DataFrame({
    "regime": ["calmo", "normal", "estresse"],
    "spread 5a (bps)": [60, 110, 400],
})

linhas = []
for _, linha in cenarios.iterrows():
    c = bootstrap_hazards(np.array([5.0]),
                          np.array([linha["spread 5a (bps)"] / 1e4]),
                          TAXA, RECUPERACAO)
    neutra = float(c.pd_acumulada(5.0)[0])
    linhas.append({
        "regime": linha["regime"],
        "spread (bps)": linha["spread 5a (bps)"],
        "PD neutra (5a)": neutra,
        "PD física (5a)": pd_fisica_5a,
        "razão": neutra / pd_fisica_5a,
        "prêmio implícito": premio_implicito(neutra, pd_fisica_5a, 5.0),
    })
pd.DataFrame(linhas).set_index("regime").round(4)"""
    ),
    md(
        """Mantendo a PD física constante — a qualidade dos devedores não muda de
um mês para o outro —, a variação do spread é quase toda variação de **prêmio de
risco**.

Isso reenquadra a leitura de spreads de crédito. Um alargamento de 110 para 400
pontos-base raramente significa que o mercado passou a esperar quatro vezes mais
defaults. Significa, em boa medida, que o preço de carregar aquele risco subiu.
Ler alargamento de spread como piora de qualidade creditícia é o erro mais comum
na leitura de mercado de crédito.

Vale a ressalva honesta: separar as duas componentes exige uma estimativa
independente da PD física, e essa estimativa carrega a incerteza toda dos
capítulos 1 a 4. A decomposição é útil qualitativamente e frágil nos números.

## O que quebra fora do laboratório

**Recuperação é suposta, não observada.** Todo o cálculo usa $R = 40\\%$ por
convenção. Como só o produto $\\lambda(1-R)$ é identificado pelo spread, errar $R$
erra $\\lambda$ na mesma proporção. A PD implícita é tão firme quanto uma
convenção de mercado.

**Liquidez está dentro do spread.** Parte do spread paga iliquidez do próprio
contrato, não risco de crédito, e essa parcela é grande em nomes pouco
negociados.

**Risco de contraparte.** Quem vende a proteção pode quebrar junto com o
emissor de referência — e foi o que aconteceu em 2008. O spread observado embute
esse desconto.

**A curva raramente é limpa.** Poucos emissores têm CDS líquido em cinco prazos.
Interpolar entre pontos ilíquidos produz curvas de hazard com formatos
implausíveis, às vezes negativos.

**No Brasil, o mercado é raso.** CDS de emissores corporativos brasileiros é
escasso; a alternativa é extrair spread de títulos, o que traz junto risco de
taxa, liquidez e cláusulas contratuais.

## Ponte regulatória

**Provisão usa PD física.** A perda esperada contábil é expectativa de perda, não
preço de mercado do risco. Modelos alimentados por PD implícita de mercado
produzem provisão pró-cíclica e superestimada.

**Marcação a mercado usa PD neutra.** Instrumentos ao valor justo e ajustes de
valor de crédito em derivativos exigem a medida neutra ao risco — aqui a PD
histórica é que estaria errada.

**Coerência entre as duas.** Um banco que usa as duas medidas em áreas diferentes
deveria conseguir explicar a diferença entre elas, e essa explicação é o prêmio
de risco. Se a diferença muda sem explicação, alguma das duas está mal calibrada.

**Uso em validação.** Spread de mercado é uma referência externa útil para
avaliar se um rating interno está descolado — não porque a PD implícita seja a
verdadeira, mas porque a **ordenação** entre emissores deveria ser parecida.
Divergência sistemática de ordenação é sinal a investigar.

## Exercícios

1. Refaça o bootstrap supondo recuperação de 20% e de 60%. Quanto muda a PD
   implícita de cinco anos? Compare a sensibilidade a $R$ com a sensibilidade ao
   spread.

2. Construa uma curva de spreads decrescente (spread de 1 ano acima do de 10
   anos), típica de emissores em dificuldade. A curva de hazard resultante faz
   sentido? Em algum trecho fica negativa?

3. Usando a PD física do capítulo 3 por rating e spreads de mercado por rating,
   estime o prêmio de risco por grade. Ele é constante ao longo da escala?

4. Calcule a perda esperada de uma carteira de duas formas: com a PD física e com
   a PD neutra ao risco. Qual a diferença percentual na provisão? Escreva o
   parágrafo explicando ao controller por que a segunda está errada."""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap10_cds.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
