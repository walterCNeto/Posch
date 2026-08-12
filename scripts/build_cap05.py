"""Monta o notebook do capítulo 5 (perda dada o default)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 5 — Perda dada o default

## O problema

A perda esperada de uma operação é o produto de três coisas:

$$
PE = PD \\times LGD \\times EAD.
$$

Os quatro capítulos anteriores trataram do primeiro fator. Este trata do
segundo, que costuma receber uma fração da atenção — e não deveria, porque o
produto é simétrico: errar a LGD por 30% erra a perda esperada por 30%,
exatamente como errar a PD.

Há uma assimetria de tratamento que vale nomear. PD tem literatura, comitê,
backtest e um capítulo inteiro de validação. LGD frequentemente é uma tabela de
valores fixos por produto, revisada quando alguém lembra. A razão é histórica —
dados de recuperação são escassos e sujos, porque exigem acompanhar o processo
de cobrança até o fim, o que leva anos.

Mas há uma razão técnica também: a LGD é estatisticamente desagradável de um
jeito que a PD não é."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.data.generators import COEFS_LGD, gerar_lgd
from credrisk.lgd.fracionaria import (
    ajustar_fracionaria,
    ajustar_ols,
    fracao_fora_do_intervalo,
    lgd_downturn,
    lgd_media_por_ano,
    prever_lgd,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""
    ),
    md(
        """## Os dados

Observações de LGD por operação, com senioridade, alavancagem do emissor,
cobertura por garantia e um fator de ciclo anual. Dezoito anos."""
    ),
    code(
        """dados = gerar_lgd()
print(f"{len(dados):,} operações · {dados['Ano'].nunique()} anos · "
      f"LGD média {dados['LGD'].mean():.1%}")
dados.head()"""
    ),
    code(
        """fig, ax = plt.subplots()
ax.hist(dados["LGD"], bins=40, alpha=0.85)
ax.set_title("Distribuição da LGD")
ax.set_xlabel("LGD")
ax.set_ylabel("operações")
plt.show()

extremos = pd.Series({
    "abaixo de 5%": (dados["LGD"] < 0.05).mean(),
    "entre 40% e 60%": dados["LGD"].between(0.4, 0.6).mean(),
    "acima de 95%": (dados["LGD"] > 0.95).mean(),
})
extremos.round(3)"""
    ),
    md(
        """Aqui está a primeira dificuldade, e ela é visual antes de ser
estatística: **a distribuição é bimodal**. Há massa considerável perto de zero
— operações bem garantidas que recuperam quase tudo — e massa perto de um —
dívida subordinada sem garantia que não recupera nada. O meio é rarefeito.

A LGD média é 44%. Quase nenhuma operação tem LGD de 44%.

Isso mata a intuição de "estimar a média e usar". A média de uma distribuição
bimodal descreve mal qualquer observação individual, e um modelo que só acerta a
média da carteira pode errar sistematicamente em cada segmento dela.

## Por que regressão linear não serve

O reflexo é rodar um OLS da LGD sobre as características. Vejamos."""
    ),
    code(
        """ols = ajustar_ols(dados)
prev_ols = prever_lgd(ols, dados)

print(f"previsões abaixo de 0 ou acima de 1: {fracao_fora_do_intervalo(prev_ols):.2%}")
print(f"menor previsão: {prev_ols.min():.3f}")
print(f"maior previsão: {prev_ols.max():.3f}")"""
    ),
    md(
        """O modelo prevê **perda negativa** em uma parcela relevante das
operações. Perda negativa significa recuperar mais do que se emprestou.

Não é um caso patológico de laboratório: acontece justamente nas operações bem
garantidas, que é onde o banco mais precisa de estimativa confiável — são as que
recebem menos capital e menos provisão. O modelo erra pior exatamente onde o
erro custa mais.

E o problema não desaparece truncando em zero. Truncar conserta o sintoma e
mantém a causa: o OLS supõe que o efeito de uma variável é constante ao longo de
toda a escala, e num intervalo limitado isso não pode valer. O efeito de mais
garantia sobre quem já recupera 95% tem de ser menor que sobre quem recupera 50%.

## Regressão fracionária

A alternativa é modelar a **média condicional** com um link que respeita o
intervalo:

$$
E[LGD \\mid x] = \\Lambda(x'\\beta) = \\frac{1}{1 + e^{-x'\\beta}}.
$$

A estimação é por quase-máxima verossimilhança com família binomial — o
procedimento de Papke e Wooldridge. O nome assusta mais que o conceito: é o
mesmo maquinário do logit do capítulo 1, aplicado a uma variável contínua entre
0 e 1 em vez de a um indicador.

A propriedade que importa: a estimativa é consistente se a **média condicional**
estiver correta, mesmo que a distribuição não esteja. Não é preciso acreditar que
a LGD segue binomial — e ela obviamente não segue."""
    ),
    code(
        """frac = ajustar_fracionaria(dados)
prev_frac = prever_lgd(frac, dados)

print(f"previsões fora de [0,1]: {fracao_fora_do_intervalo(prev_frac):.2%}")
print(f"intervalo das previsões: [{prev_frac.min():.3f}, {prev_frac.max():.3f}]")
print()
print(frac.summary2().tables[1].round(4).to_string())"""
    ),
    code(
        """fig, ax = plt.subplots()
ax.scatter(prev_ols, prev_frac, s=10, alpha=0.35)
lim = [min(prev_ols.min(), 0) - 0.05, 1.0]
ax.plot(lim, lim, color="#7a8b8b", ls=":", lw=1.5)
ax.axvline(0, color="#c1553b", lw=1.5, label="limite inferior admissível")
ax.set_title("Onde os dois modelos discordam")
ax.set_xlabel("LGD prevista — OLS")
ax.set_ylabel("LGD prevista — fracionária")
ax.legend()
plt.show()"""
    ),
    md(
        """No miolo da carteira os dois modelos concordam quase perfeitamente. A
discordância se concentra na ponta esquerda, onde o OLS atravessa o zero e a
fracionária se achata em direção a ele.

Esse padrão é o mesmo do capítulo 3: o método melhor não ganha na média, ganha
na cauda. Se você comparar os dois por erro quadrático médio na carteira toda,
vai concluir que tanto faz.

## O que o modelo recuperou

Abrindo o gabarito, com uma sutileza que vale o parágrafo."""
    ),
    code(
        """completo = ajustar_fracionaria(dados, numericas=["LEV", "COB", "CICLO"])

nomes = ["CONST", "Sr. Unsec.", "Sub.", "LEV", "COB"]
comparacao = pd.DataFrame({
    "verdadeiro": pd.Series(COEFS_LGD)[nomes],
    "sem ciclo": frac.params[nomes],
    "com ciclo": completo.params[nomes],
})
comparacao.round(3)"""
    ),
    md(
        """O modelo que inclui o fator de ciclo recupera os coeficientes quase
exatamente. O que o omite **encolhe** os coeficientes das outras variáveis —
`COB` vai de −2,6 para −2,3.

Vale entender por que, porque a explicação usual não se aplica. Não é viés de
variável omitida no sentido clássico: o fator de ciclo é ortogonal à cobertura
por garantia de cada operação, então não há confusão entre os dois efeitos.

O que acontece é que, num modelo **não-linear**, a média das médias condicionais
não é a média condicional da média. Ao integrar sobre o ciclo não observado, a
curva resultante é mais achatada que a curva condicional a um ciclo fixo. O
coeficiente que se estima passa a ser marginal, não condicional — e o marginal é
sempre menor em magnitude.

Consequência prática: **o coeficiente estimado depende de quais fatores estão no
modelo**, mesmo que os fatores sejam independentes entre si. Comparar
elasticidades entre dois modelos com conjuntos diferentes de variáveis é
comparar coisas diferentes, ainda que ambos estejam corretos.

## A correlação que obriga o downturn

Anos ruins não têm só mais defaults. Têm recuperações piores — a garantia que
vale menos quando todo mundo está vendendo, o mercado secundário sem liquidez, o
processo de cobrança mais lento."""
    ),
    code(
        """por_ano = lgd_media_por_ano(dados)
rho = np.corrcoef(por_ano["LGD_media"], por_ano["taxa_default"])[0, 1]

fig, ax = plt.subplots()
ax.scatter(por_ano["taxa_default"] * 100, por_ano["LGD_media"] * 100, s=55)
for _, linha in por_ano.iterrows():
    ax.annotate(int(linha["Ano"]), (linha["taxa_default"] * 100,
                linha["LGD_media"] * 100),
                fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_title(f"Severidade e frequência andam juntas (ρ = {rho:.2f})")
ax.set_xlabel("taxa de default do ano (%)")
ax.set_ylabel("LGD média do ano (%)")
plt.show()"""
    ),
    md(
        """A correlação é forte e positiva. Isso tem uma consequência que a
fórmula de perda esperada esconde.

$PD \\times LGD$ trata os dois fatores como independentes. Quando eles são
positivamente correlacionados, o produto das médias **subestima** a média do
produto — e a diferença aparece justamente nos anos em que o banco menos pode
absorvê-la. É o mesmo mecanismo que faz uma carteira concentrada perder mais que
a soma dos riscos individuais sugere."""
    ),
    code(
        """downturn = lgd_downturn(dados, quantil=0.80)

print(f"LGD média de todo o período:        {downturn['media_geral']:.1%}")
print(f"LGD média nos {downturn['anos_downturn']} anos de estresse:      "
      f"{downturn['media_downturn']:.1%}")
print(f"diferença:                         {downturn['diferenca']:+.1%} "
      f"({downturn['razao']:.2f}×)")"""
    ),
    md(
        """Usar a LGD média de longo prazo em vez da LGD de estresse subestima a
perda em mais de um terço, precisamente nos anos em que ela se materializa.

É por isso que o arcabouço de capital exige LGD que reflita condições de
desaceleração econômica sempre que houver dependência entre frequência e
severidade. A exigência não é conservadorismo gratuito — é correção de um viés
mensurável, e o gráfico acima é como se demonstra que ele existe na sua carteira.

## O que quebra fora do laboratório

**Recuperação leva anos.** Um default de 2024 pode ter recuperação final
conhecida só em 2029. Estimar LGD com casos encerrados exclui os processos
longos, que costumam ser os de pior recuperação — é viés de seleção com direção
conhecida. Tratar os casos em aberto como censurados, e não descartá-los, é o
mínimo.

**Desconto importa.** LGD é valor presente de fluxos futuros. A taxa de desconto
usada muda o número materialmente e é escolha metodológica, não dado. Taxa
livre de risco, custo de capital e taxa contratual dão respostas bem diferentes.

**Custos administrativos.** Cobrança, honorários advocatícios e custas judiciais
entram na perda e frequentemente não estão no sistema de recuperação. Ignorá-los
subestima a LGD de forma sistemática, e mais nas operações pequenas.

**Cura não é recuperação.** Operações que entram em default e voltam a adimplir
têm LGD baixa ou nula, e a taxa de cura varia enormemente por produto. Misturar
curadas e não curadas numa média produz um número que não descreve nenhum dos
dois grupos — o mesmo problema da bimodalidade, em outra roupa.

## Ponte regulatória

**LGD de downturn é exigência, não opção.** Quando há correlação entre
frequência e severidade — e a demonstração acima é como se verifica — a
estimativa usada em capital tem de refletir condições adversas. A definição de
qual período é *downturn*, e como estimar a partir de poucos anos ruins, é onde
o julgamento entra e onde a validação precisa apertar.

**Margem de conservadorismo.** Com poucos anos de dados e poucos casos por
segmento, a incerteza da estimativa é grande. O arcabouço espera que essa
incerteza vire margem explícita, não que seja ignorada. Quantificá-la por
bootstrap, como no capítulo 3, é o caminho direto.

**Coerência com o uso contábil.** A LGD de provisão sob perda esperada e a de
capital não são o mesmo objeto: uma é neutra ao ciclo e prospectiva, a outra é de
*downturn*. Usar a mesma tabela para as duas é erro comum, e produz ou provisão
excessiva ou capital insuficiente.

**Segmentação.** A bimodalidade deste capítulo é argumento a favor de modelar por
segmento homogêneo, com e sem garantia, em vez de uma média única com dummies.
Qual granularidade se sustenta estatisticamente é pergunta de validação — e
depende de quantos casos há em cada célula.

## Exercícios

1. Estime a LGD média por senioridade e por faixa de cobertura, e compare com as
   previsões do modelo fracionário. Onde a tabela simples e o modelo mais
   discordam? Vale a complexidade do modelo?

2. Descarte aleatoriamente 60% das observações dos anos de estresse, simulando
   processos ainda não encerrados. Quanto a LGD de downturn é subestimada? Esse
   é o viés de censura em forma pura.

3. Compare o erro quadrático médio do OLS e do modelo fracionário na carteira
   inteira, e depois só nas operações com cobertura acima de 0,8. O que a
   diferença entre as duas comparações ensina sobre escolher modelo por métrica
   agregada?

4. Calcule a perda esperada da carteira de duas formas: $\\overline{PD} \\times
   \\overline{LGD}$ e a média anual de $PD_t \\times LGD_t$. Qual a diferença
   percentual? Você acabou de medir o custo de supor independência."""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap05_lgd.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
