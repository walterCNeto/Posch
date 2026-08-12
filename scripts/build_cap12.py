"""Monta o notebook do capítulo 12 (Basileia e ratings internos)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 12 — Basileia e ratings internos

## O problema

Os onze capítulos anteriores construíram uma máquina: estimam PD, LGD,
correlação, agregam em distribuição de perda, validam cada peça. Este capítulo
responde à pergunta que motiva boa parte desse esforço em um banco real —
**quanto capital o regulador exige?**

A resposta tem uma propriedade que costuma surpreender quem chega à fórmula pela
primeira vez: ela não é uma convenção negociada em comitê. É o modelo do
capítulo 7, aplicado exposição por exposição, com duas modificações
deliberadas.

Este capítulo faz três coisas. Deriva a fórmula a partir do que já foi
construído, mostra por que ela tem a forma que tem, e mede onde ela erra."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.portfolio.montecarlo import (
    numero_efetivo_posicoes,
    quantil_vasicek,
    simular_perdas,
)
from credrisk.regcap.irb import (
    FATOR_RWA,
    ajuste_maturidade,
    correlacao_prescrita,
    curva_de_capital,
    efeito_da_granularidade,
    requerimento_capital,
    resumo_carteira,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

LGD = 0.45"""
    ),
    md(
        """## A fórmula, e de onde ela vem

O requerimento de capital por unidade de exposição é

$$
K = \\left[LGD \\cdot \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) + \\sqrt{R}\\,\\Phi^{-1}(0{,}999)}{\\sqrt{1-R}}\\right) - PD \\cdot LGD\\right] \\times \\text{ajuste de maturidade}.
$$

Compare com o capítulo 7. O termo dentro de $\\Phi$ é **exatamente** o quantil de
Vasicek para carteira homogênea infinitamente granular. O segundo termo subtrai
a perda esperada, porque ela é coberta por provisão — somar os dois seria contar
a mesma perda duas vezes, uma no resultado e outra no capital.

Não é analogia. É identidade, e dá para verificar."""
    ),
    code(
        """verificacao = []
for p in [0.0005, 0.001, 0.01, 0.05, 0.20]:
    R = float(correlacao_prescrita(p))
    do_capitulo_7 = (quantil_vasicek(p, R, 0.999) - p) * LGD
    do_irb = float(requerimento_capital(p, LGD)) / float(ajuste_maturidade(p, 2.5))
    verificacao.append({
        "PD": p,
        "ASRF (capítulo 7)": do_capitulo_7,
        "IRB sem maturidade": do_irb,
        "diferença": abs(do_capitulo_7 - do_irb),
    })
pd.DataFrame(verificacao).set_index("PD")"""
    ),
    md(
        """Idênticos até a precisão de máquina.

Isso significa que **todas as limitações do capítulo 7 são limitações do capital
regulatório**: fator único, carteira infinitamente granular, cauda gaussiana,
LGD constante. Não são simplificações que o regulador poderia ter evitado com
mais esforço — são o preço de uma propriedade que a fórmula precisa ter, e que
discutimos adiante.

## A correlação que o banco não estima

A primeira modificação em relação ao capítulo 7: a correlação **não é estimada**.
É dada por fórmula, decrescente na PD."""
    ),
    code(
        """pds = np.geomspace(0.0003, 0.30, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

for classe, rotulo in [("corporate", "corporativo"),
                       ("varejo_outros", "varejo — outros"),
                       ("varejo_hipotecario", "varejo — hipotecário"),
                       ("varejo_rotativo", "varejo — rotativo")]:
    ax1.plot(pds * 100, correlacao_prescrita(pds, classe), lw=2, label=rotulo)
ax1.set_xscale("log")
ax1.set_title("Correlação prescrita")
ax1.set_xlabel("PD (%, escala log)")
ax1.set_ylabel("R")
ax1.legend(fontsize=8)

curva = curva_de_capital(pds, LGD)
for classe, rotulo in [("corporate", "corporativo"),
                       ("varejo_outros", "varejo — outros"),
                       ("varejo_rotativo", "varejo — rotativo")]:
    ax2.plot(curva["PD"] * 100, curva[classe] * FATOR_RWA * 100, lw=2, label=rotulo)
ax2.set_xscale("log")
ax2.set_title("Ponderação de risco")
ax2.set_xlabel("PD (%, escala log)")
ax2.set_ylabel("RW (%)")
ax2.legend(fontsize=8)
plt.show()"""
    ),
    md(
        """A correlação corporativa interpola entre 24% para os melhores créditos e
12% para os piores. A justificativa declarada é econômica: devedores de pior
qualidade quebram mais por razões próprias — má gestão, fraude, um cliente
grande que sumiu — e menos por razões sistêmicas. Empresas sólidas só quebram
quando a economia inteira quebra.

Mas há uma segunda razão, e o capítulo 6 é ela. Estimar $\\rho$ livremente
produziria um intervalo de confiança em que o capital varia por um fator de
quase três — com vinte e cinco anos de dados. Deixar cada banco escolher o seu
número dentro dessa faixa criaria dispersão enorme entre instituições com
carteiras parecidas, e um incentivo evidente para escolher o extremo baixo.

**Prescrever a correlação não é desconfiança do regulador quanto à competência
dos bancos. É reconhecimento de que o parâmetro não é estimável com a precisão
que a decisão exige.**

Vale ver como o valor prescrito se compara ao que o capítulo 6 estimou."""
    ),
    code(
        """estimado_cap6 = {"pontual": 0.098, "IC inferior": 0.056, "IC superior": 0.177}
prescrito = float(correlacao_prescrita(0.015, "corporate"))

comparacao = pd.Series({**estimado_cap6, "prescrito pelo IRB (PD 1,5%)": prescrito})
print(comparacao.round(4).to_string())
print()
capital_por_rho = {
    nome: (quantil_vasicek(0.015, r, 0.999) - 0.015) * LGD
    for nome, r in {**estimado_cap6, "prescrito": prescrito}.items()
}
pd.Series(capital_por_rho, name="capital / EAD").round(5)"""
    ),
    md(
        """O valor prescrito fica no extremo superior do intervalo estimado — é
conservador, mas dentro da faixa que os dados admitem. Não é um número
arbitrário nem um número que os dados contradigam.

## O ajuste de maturidade

A segunda modificação cobre algo que o capítulo 7 ignora: o modelo de lá é de
**default puro**, e um crédito de prazo longo pode perder valor sem entrar em
default — basta ser rebaixado. É o risco de migração do capítulo 3."""
    ),
    code(
        """maturidades = np.linspace(1, 5, 60)

fig, ax = plt.subplots()
for p, rotulo in [(0.0005, "PD 0,05%"), (0.01, "PD 1%"), (0.10, "PD 10%")]:
    ax.plot(maturidades, ajuste_maturidade(p, maturidades), lw=2, label=rotulo)
ax.set_title("Multiplicador de maturidade")
ax.set_xlabel("maturidade efetiva (anos)")
ax.set_ylabel("multiplicador")
ax.legend()
plt.show()"""
    ),
    md(
        """Duas propriedades merecem atenção.

Em $M = 1$ o multiplicador vale exatamente 1, para qualquer PD — o prazo de um
ano é o caso base, e ali o modelo de default puro basta.

E o ajuste é **muito maior para PDs baixas**. Um crédito AAA de cinco anos tem
requerimento mais que dobrado; um CCC de cinco anos, pouco mais que um quarto a
mais. A lógica: um AAA tem enorme espaço para se deteriorar sem quebrar,
enquanto um CCC ou quebra ou não — não há muito para onde piorar.

É o mesmo formato das curvas do capítulo 3, onde a PD acumulada do AA era convexa
no horizonte e a do CCC, côncava.

## O que a fórmula acerta

A propriedade mais importante da fórmula não está escrita nela: o requerimento de
cada exposição **não depende das demais**. É por isso que o capital de uma
carteira é a soma simples dos capitais individuais.

Isso se chama invariância à carteira, e é o que torna a fórmula operacionalmente
viável. Sem ela, todo banco precisaria rodar um modelo de carteira completo para
saber o capital de uma operação nova — e o supervisor não teria como comparar
dois bancos.

Vale confirmar que a fórmula reproduz o capital econômico quando suas hipóteses
valem."""
    ),
    code(
        """PD_TESTE = 0.015
R_TESTE = float(correlacao_prescrita(PD_TESTE))
rng = np.random.default_rng(5)


def montar(n, sigma):
    ead = np.full(n, 1000.0) if sigma == 0 else rng.lognormal(0, sigma, n)
    ead = ead / ead.sum() * (n * 1000.0)
    return pd.DataFrame({
        "ID": np.arange(n), "EAD": ead,
        "PD": PD_TESTE, "LGD": LGD, "RHO": R_TESTE,
    })


linhas = []
for rotulo, n, sigma in [
    ("5.000 posições iguais", 5000, 0.0),
    ("500 posições iguais", 500, 0.0),
    ("500 com exposição desigual", 500, 1.5),
    ("100 posições iguais", 100, 0.0),
]:
    c = montar(n, sigma)
    economico = simular_perdas(c, 30_000, semente=3).capital(0.999)
    regulatorio = float(
        (c["EAD"].to_numpy() * requerimento_capital(
            c["PD"].to_numpy(), c["LGD"].to_numpy(), maturidade=1.0)).sum()
    )
    linhas.append({
        "carteira": rotulo,
        "nº efetivo": numero_efetivo_posicoes(c)["numero_efetivo"],
        "capital econômico": economico,
        "capital IRB": regulatorio,
        "IRB / econômico": regulatorio / economico,
    })
pd.DataFrame(linhas).set_index("carteira").round(3)"""
    ),
    md(
        """Na carteira grande e uniforme, os dois praticamente coincidem — a
fórmula está certa dentro das suas hipóteses.

À medida que a carteira concentra, o IRB fica **abaixo** do capital econômico. E
o efeito não vem de o número de posições cair: vem de o número **efetivo** cair.
Quinhentas posições com exposição desigual valem menos de cem posições iguais, e
o capital regulatório não recebe nenhum insumo capaz de perceber isso.

Este é o preço da invariância à carteira, e ele é estrutural. Uma fórmula que
somasse posição a posição **e** capturasse concentração não pode existir:
concentração é, por definição, propriedade do conjunto.

É por isso que risco de concentração é tratado em pilar separado, com avaliação
própria do banco e do supervisor. Não é lacuna esquecida — é a parte que a
fórmula não tem como cobrir, endereçada por outro instrumento.

## Quantas grades precisa ter o sistema de rating?

Uma pergunta prática de desenho que a fórmula responde sozinha. Sistemas de
rating não atribuem PD contínua: classificam em grades, e todo devedor da grade
recebe a mesma PD."""
    ),
    code(
        """rng2 = np.random.default_rng(2)
n = 1500
carteira_real = pd.DataFrame({
    "EAD": rng2.lognormal(6.2, 1.1, n),
    "PD": np.clip(rng2.lognormal(np.log(0.02), 0.95, n), 1e-4, 0.35),
    "LGD": np.full(n, LGD),
})

tabela = efeito_da_granularidade(carteira_real, grades=(1, 2, 3, 5, 7, 10, 15, 25))
tabela.set_index("grades").round(5)"""
    ),
    md(
        """O sinal é o que importa aqui, e ele não é óbvio: **menos grades exigem
mais capital**.

A razão é a concavidade. O requerimento cresce rápido no início da escala de PD
e achata na ponta ruim — veja o painel de ponderação de risco lá em cima, em
escala log. Por Jensen, substituir PDs distintas pela média da grade aumenta o
requerimento calculado, sem que o risco tenha mudado.

A consequência é um incentivo bem posto: um sistema de rating grosseiro custa
capital ao banco, e o custo é mensurável. Com poucas grades a penalidade é
relevante; a partir de dez ou quinze, o ganho de acrescentar mais vira ruído.

Repare que a redução **não é monótona** — em alguns pontos, acrescentar uma
grade piora. Isso não é erro de cálculo: as fronteiras aqui são recalculadas do
zero a cada número de grades, e não são aninhadas. Onde as fronteiras caem
importa tanto quanto quantas são, e é por isso que desenhar a escala de rating é
decisão substantiva, não escolha de um número redondo.

Isso dá um critério objetivo para uma decisão que costuma ser tomada por
tradição: a granularidade se justifica até onde a redução de capital compensa o
custo de operar mais grades e de estimar PD confiável em cada uma. E aqui os
capítulos 1 e 8 voltam com a ressalva — mais grades significam menos devedores
por grade, e menos devedores por grade significam PD pior estimada e
intestável.

## Uma carteira completa

Juntando tudo, o cálculo de ponta a ponta."""
    ),
    code(
        """resumo = resumo_carteira(carteira_real, classe="corporate", maturidade=2.5)
print(resumo.round(4).to_string())
print()
print(f"o capital cobre {resumo['capital exigido'] / resumo['perda esperada']:.1f}× "
      f"a perda esperada")"""
    ),
    md(
        """O capital exigido é várias vezes a perda esperada. Faz sentido: a
provisão cobre o ano típico, o capital cobre o ano ruim, e a distribuição de
perda de crédito é assimétrica — como o capítulo 7 mostrou, a maior parte dos
anos fica **abaixo** da média.

## O que quebra fora do laboratório

**Um fator só, e é global.** A fórmula supõe um único fator sistêmico comum a
todos os devedores do mundo. Concentração setorial e geográfica não aparecem.

**Correlação prescrita não é a sua correlação.** A fórmula usa uma correlação
calibrada para uma carteira internacional de referência. Não há razão para que
seja a da sua carteira, e o capítulo 6 mostra que estimar a sua não resolve — o
intervalo é largo demais.

**LGD constante.** O capítulo 5 mostrou LGD subindo justamente nos anos ruins. A
exigência de LGD de *downturn* é o remendo prudencial para isso, e é remendo
mesmo: entra como número fixo mais alto, não como variável correlacionada.

**Cauda gaussiana.** Vale aqui o mesmo do capítulo 11.

**Parâmetros são estimados, e a fórmula os trata como certos.** PD, LGD e EAD
entram como se fossem conhecidos. Toda a incerteza dos capítulos 1 a 5 desaparece
na conta. A margem de conservadorismo exigida na estimação é a resposta a isso —
e é por isso que ela não é opcional.

## Ponte regulatória

**No Brasil.** As abordagens IRB estão regulamentadas na Circular BCB 3.648 e nas
normas que a sucederam, e os requerimentos de capital consolidados na Resolução
BCB 265. A validação independente desses modelos é tratada na Resolução BCB 229 e
na estrutura de gerenciamento de riscos da Resolução CMN 4.557. Como o arcabouço
brasileiro é revisado com frequência, vale confirmar a redação vigente antes de
citar dispositivo específico — o que este capítulo ensina é a mecânica, que muda
muito mais devagar que a numeração.

**Piso de resultados.** Basileia III limita o quanto o capital calculado por
modelo interno pode ficar abaixo do calculado pela abordagem padronizada. É
reconhecimento explícito do risco de modelo: mesmo com validação, a liberdade de
parametrização produziu dispersão grande demais entre bancos.

**Piso de PD.** Existe justamente pelo problema do capítulo 3 — estimadores
ingênuos devolvem zero em grades superiores, e zero em provisão significa
provisão zero para sempre.

**Uso efetivo.** O arcabouço exige que os modelos usados para capital sejam os
mesmos usados na gestão. É uma exigência de coerência com dentes: um modelo
calibrado para minimizar capital tende a produzir decisões de crédito ruins, e a
exigência de uso força o banco a conviver com as consequências da própria
calibragem.

## O que este curso construiu

Fechando os doze capítulos, o encadeamento fica visível:

| capítulo | contribuição | entra em |
|---|---|---|
| 1, 2 | PD individual | $K$, provisão |
| 3, 4 | PD por horizonte e condicionada ao ciclo | perda esperada *lifetime* |
| 5 | LGD, e por que ela sobe em crise | $K$, provisão |
| 6 | correlação, e por que é prescrita | $R$ da fórmula |
| 7 | distribuição de perda e ASRF | a própria fórmula |
| 8, 9 | validação de cada peça e do todo | governança do modelo |
| 10 | preço versus frequência | marcação, não provisão |
| 11 | o que acontece com payoff convexo | securitização |
| 12 | como tudo vira capital | requerimento |

E o fio condutor, que apareceu em quase todos: **a incerteza dos parâmetros
costuma importar mais que a sofisticação do método.** O capítulo 1 achou 82
eventos para seis parâmetros; o 6 achou um intervalo de correlação que triplica
o capital; o 9 mostrou que o backtest não distingue essas hipóteses; o 11 mostrou
o que acontece quando um payoff convexo é construído sobre esse parâmetro.

A resposta correta a essa situação não é modelo melhor. É reportar a faixa em vez
do ponto, e escrever o que o teste não conseguiu descartar.

## Exercícios

1. Calcule o capital da mesma carteira pelas quatro classes de exposição. Quanto
   vale, em capital, classificar uma operação como varejo rotativo em vez de
   corporativo? Isso é incentivo a quê?

2. Aplique o ajuste de porte a uma carteira de PMEs e compare com o tratamento
   corporativo pleno. Para que faixa de faturamento o ajuste mais importa?

3. Simule o capital econômico da carteira do exercício anterior com correlação
   estimada dentro do intervalo do capítulo 6, e compare com o IRB. Em que ponto
   do intervalo os dois coincidem?

4. Monte a comparação entre capital IRB e capital econômico para uma carteira com
   concentração setorial — dois setores com fatores distintos. Quanto o IRB
   subestima? Escreva o parágrafo de avaliação de risco de concentração que
   acompanharia esse número."""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap12_basileia.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
