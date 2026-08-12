"""Monta o notebook do capítulo 2 (modelo estrutural de Merton)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 2 — A abordagem estrutural

## O problema

O capítulo 1 estimou a probabilidade de default a partir de razões contábeis e
de um histórico de falências. Isso exige duas coisas que nem sempre existem:
demonstrações financeiras comparáveis e defaults suficientes para estimar.

A abordagem estrutural resolve o problema por outro caminho. Em vez de aprender
com defaults passados, ela **deduz** a probabilidade de default de uma teoria
sobre por que empresas quebram — e alimenta essa teoria com preço de ação, que é
observado diariamente e não espera o balanço fechar.

A intuição é de Merton (1974) e cabe em uma frase: o acionista de uma empresa
alavancada é dono de uma opção de compra sobre o ativo da empresa, com preço de
exercício igual à dívida. Se o ativo valer mais que a dívida no vencimento, ele
paga a dívida e fica com o resto; se valer menos, ele entrega a empresa aos
credores e perde o que investiu — nada mais, por responsabilidade limitada.

Se o capital próprio é uma opção, então o preço da ação carrega informação sobre
o valor e a volatilidade do ativo. E essas duas coisas determinam a chance de o
ativo cair abaixo da dívida."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.data.generators import (
    DIAS_UTEIS_ANO,
    PARAMS_MERTON,
    gerar_carteira_merton,
    gerar_serie_merton,
)
from credrisk.structural.merton import (
    distancia_ao_default,
    estimar_iterativo,
    pd_merton,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""
    ),
    md(
        """## A formulação

Suponha que o valor do ativo $V_t$ siga um movimento browniano geométrico

$$
dV_t = \\mu V_t\\,dt + \\sigma_V V_t\\,dW_t,
$$

e que a empresa tenha uma única dívida de valor de face $L$ com vencimento em
$T$. No vencimento, o acionista recebe $\\max(V_T - L,\\, 0)$ — o payoff exato de
uma call. Pela fórmula de Black-Scholes,

$$
E_t = V_t\\,\\Phi(d_1) - L e^{-r(T-t)}\\,\\Phi(d_2),
\\qquad
d_{1,2} = \\frac{\\ln(V_t/L) + (r \\pm \\tfrac12\\sigma_V^2)(T-t)}{\\sigma_V\\sqrt{T-t}}.
$$

O default ocorre se $V_T < L$. Sob a medida física, a probabilidade disso é

$$
\\Pr(V_T < L) = \\Phi\\!\\left(-\\underbrace{\\frac{\\ln(V_t/L) + (\\mu - \\tfrac12\\sigma_V^2)(T-t)}{\\sigma_V\\sqrt{T-t}}}_{\\text{distância ao default}}\\right).
$$

**A dificuldade prática está em uma linha:** as fórmulas acima dependem de $V_t$
e $\\sigma_V$, e nenhum dos dois é observável. O que se observa é $E_t$ — o valor
de mercado do capital próprio.

## Os dados

A série sintética tem 260 dias úteis de valor de mercado do capital próprio de
uma empresa cujo ativo foi simulado com parâmetros conhecidos. A coluna
`V_verdadeiro` é o gabarito; vamos ignorá-la até a hora da conferência."""
    ),
    code(
        """serie = gerar_serie_merton()
observado = serie[["Dia", "E", "L", "r"]]

print(f"{len(serie)} dias úteis · dívida L = {PARAMS_MERTON['L']:,.0f} · "
      f"r = {PARAMS_MERTON['r']:.1%}")
observado.head()"""
    ),
    md(
        """## A tentação: usar a volatilidade da ação

O primeiro reflexo é usar a volatilidade observada do capital próprio no lugar
da volatilidade do ativo. É errado, e o erro tem direção conhecida."""
    ),
    code(
        """vol_capital = np.std(np.diff(np.log(serie["E"])), ddof=1) * np.sqrt(DIAS_UTEIS_ANO)
print(f"volatilidade do capital próprio: {vol_capital:.2%}")
print(f"volatilidade verdadeira do ativo: {PARAMS_MERTON['sigma_V']:.2%}")"""
    ),
    md(
        """O capital próprio é mais que o dobro de volátil que o ativo. A razão é
a alavancagem da opção: uma variação de 1% no ativo produz variação maior que 1%
no capital próprio, porque a dívida é fixa. Formalmente,

$$
\\sigma_E = \\frac{V}{E}\\,\\Phi(d_1)\\,\\sigma_V,
$$

e o fator $\\frac{V}{E}\\Phi(d_1)$ é maior que 1 em qualquer empresa alavancada.

Usar $\\sigma_E$ no lugar de $\\sigma_V$ superestimaria a PD grosseiramente. Note
também que esse fator **muda quando o preço da ação muda** — a empresa fica mais
alavancada justamente quando cai. Volatilidade do capital próprio não é
parâmetro estável, o que já antecipa por que o procedimento correto precisa ser
iterativo.

## O procedimento iterativo

Temos uma equação — Black-Scholes — e duas incógnitas, $V_t$ e $\\sigma_V$. A
solução padrão, devida à KMV, explora o fato de termos uma série temporal
inteira em vez de uma observação só:

1. chute um $\\sigma_V$ (a volatilidade do capital próprio serve, é um limite
   superior);
2. inverta Black-Scholes em cada dia para obter a série $V_t$ implícita;
3. calcule a volatilidade dos log-retornos dessa série;
4. use-a como novo chute e repita até o ponto fixo."""
    ),
    code(
        """kmv = estimar_iterativo(
    serie["E"].to_numpy(), PARAMS_MERTON["L"], PARAMS_MERTON["r"], PARAMS_MERTON["T"]
)

print(f"convergiu em {kmv.iteracoes} iterações")
print(f"σ_V estimado:   {kmv.sigma_V:.4f}")
print(f"σ_V verdadeiro: {PARAMS_MERTON['sigma_V']:.4f}")

fig, ax = plt.subplots()
ax.plot(kmv.historico_sigma, marker="o", ms=4)
ax.axhline(PARAMS_MERTON["sigma_V"], color="#c1553b", ls="--", lw=1.5,
           label="σ_V verdadeiro")
ax.set_title("Convergência do procedimento iterativo")
ax.set_xlabel("iteração")
ax.set_ylabel("σ_V")
ax.legend()
plt.show()"""
    ),
    md(
        """A convergência é rápida e monótona: parte da volatilidade do capital
próprio, muito acima, e desce em poucos passos. O ponto fixo é único porque a
alavancagem da opção cai quando $\\sigma_V$ sobe — o mapa é uma contração.

E a série do ativo recuperada?"""
    ),
    code(
        """erro = np.abs(kmv.V - serie["V_verdadeiro"]) / serie["V_verdadeiro"]
print(f"erro relativo máximo na série do ativo: {erro.max():.3%}")

fig, ax = plt.subplots()
ax.plot(serie["Dia"], serie["V_verdadeiro"], label="ativo verdadeiro", lw=2.5)
ax.plot(serie["Dia"], kmv.V, label="ativo estimado", ls="--", lw=1.5)
ax.plot(serie["Dia"], serie["E"], label="capital próprio observado", lw=1.2)
ax.axhline(PARAMS_MERTON["L"], color="#7a8b8b", ls=":", label="dívida L")
ax.set_title("O que se observa e o que se infere")
ax.set_xlabel("dia útil")
ax.legend()
plt.show()"""
    ),
    md(
        """A recuperação do **nível** do ativo é quase exata — erro máximo abaixo
de meio por cento. Isso não deveria surpreender: dado $\\sigma_V$, a inversão de
Black-Scholes é determinística e bem-condicionada.

Toda a incerteza do método está no $\\sigma_V$, e é ali que vale olhar.

## Do parâmetro à probabilidade

O erro em $\\sigma_V$ foi de poucos pontos percentuais. A pergunta que importa é
o que ele vira depois de passar por $\\Phi(-DD)$."""
    ),
    code(
        """V_final_est = kmv.V[-1]
V_final_verd = serie["V_verdadeiro"].iloc[-1]
L, mu, r = PARAMS_MERTON["L"], PARAMS_MERTON["mu"], PARAMS_MERTON["r"]

comparacao = pd.DataFrame({
    "estimado": {
        "σ_V": kmv.sigma_V,
        "V": V_final_est,
        "DD": float(distancia_ao_default(V_final_est, L, mu, kmv.sigma_V)),
        "PD": float(pd_merton(V_final_est, L, mu, kmv.sigma_V)),
    },
    "verdadeiro": {
        "σ_V": PARAMS_MERTON["sigma_V"],
        "V": V_final_verd,
        "DD": float(distancia_ao_default(V_final_verd, L, mu, PARAMS_MERTON["sigma_V"])),
        "PD": float(pd_merton(V_final_verd, L, mu, PARAMS_MERTON["sigma_V"])),
    },
})
comparacao["erro relativo"] = (
    comparacao["estimado"] / comparacao["verdadeiro"] - 1
)
comparacao"""
    ),
    md(
        """Aqui está o resultado desconfortável do capítulo: um erro de poucos
por cento em $\\sigma_V$ vira um erro **muito maior** na PD.

A causa é a não-linearidade de $\\Phi(-DD)$ combinada com a de $DD$ em
$\\sigma_V$ — a volatilidade aparece no numerador e no denominador da distância
ao default. Vale medir a amplificação em vez de estimá-la de olho."""
    ),
    code(
        """replicas = []
for semente in range(200, 250):
    d = gerar_serie_merton(semente=semente)
    est = estimar_iterativo(d["E"].to_numpy(), L, r, PARAMS_MERTON["T"])
    replicas.append({
        "σ_V estimado": est.sigma_V,
        "PD estimada": float(pd_merton(est.V[-1], L, mu, est.sigma_V)),
        "PD verdadeira": float(
            pd_merton(d["V_verdadeiro"].iloc[-1], L, mu, PARAMS_MERTON["sigma_V"])
        ),
    })

replicas = pd.DataFrame(replicas)
erro_sigma = (replicas["σ_V estimado"] / PARAMS_MERTON["sigma_V"] - 1).abs()
erro_pd = (replicas["PD estimada"] / replicas["PD verdadeira"] - 1).abs()

vies = replicas["σ_V estimado"].mean() - PARAMS_MERTON["sigma_V"]
print(f"viés de σ_V:                     {vies:+.4f}")
print(f"erro relativo mediano em σ_V:    {erro_sigma.median():.1%}")
print(f"erro relativo mediano na PD:     {erro_pd.median():.1%}")
print(f"fator de amplificação:           {erro_pd.median() / erro_sigma.median():.1f}×")"""
    ),
    md(
        """O estimador de $\\sigma_V$ é essencialmente não-enviesado — o
procedimento iterativo funciona como anunciado. Mas o erro que sobra é
amplificado por um fator de cerca de cinco quando vira probabilidade: um erro de
3% na volatilidade vira um erro de 15% na PD.

Isso tem consequência direta em validação: **um modelo estrutural pode ter
parâmetros bem estimados e PDs ruins**, e nenhum diagnóstico sobre a estimação
de $\\sigma_V$ revelaria isso. A precisão que interessa é a da quantidade final,
não a dos insumos — e ela precisa ser reportada com intervalo, não como ponto.

## Distância ao default e a tradução em probabilidade

A distância ao default é a saída robusta do modelo; a PD é a saída frágil.
Vale ver por quê, numa seção transversal de empresas."""
    ),
    code(
        """carteira = gerar_carteira_merton()
carteira["DD"] = distancia_ao_default(
    carteira["V"], carteira["L"], mu, carteira["sigma_V"]
)
carteira["PD"] = pd_merton(carteira["V"], carteira["L"], mu, carteira["sigma_V"])

fig, ax = plt.subplots()
ordem = carteira.sort_values("DD")
ax.plot(ordem["DD"], ordem["PD"] * 100, lw=2)
ax.set_title("A tradução de distância em probabilidade é fortemente não-linear")
ax.set_xlabel("distância ao default (desvios-padrão)")
ax.set_ylabel("PD (%)")
plt.show()

print(carteira[["DD", "PD"]].describe().round(4))"""
    ),
    md(
        """Na região de DD entre 1 e 3 — onde vive a maior parte de uma carteira
corporativa real — a curva é íngreme: pequenas diferenças de distância viram
grandes diferenças de PD. Acima de DD = 4 ela é rasa, e a normal esmaga tudo
para perto de zero.

É por isso que a KMV, no produto comercial, **não usa $\\Phi(-DD)$**. Ela mapeia
a distância ao default para uma frequência empírica de default, estimada de um
histórico de empresas com distâncias parecidas. A normal é uma hipótese sobre a
cauda que os dados não sustentam: defaults reais são mais frequentes do que ela
prevê para DD alto.

Guarde essa distinção — modelo bem especificado no meio da distribuição e
péssimo na cauda é um padrão que reaparece nos capítulos 7 e 11.

## Física ou neutra ao risco?

A mesma fórmula com $r$ no lugar de $\\mu$ devolve outra coisa."""
    ),
    code(
        """pd_fisica = float(pd_merton(V_final_verd, L, mu, PARAMS_MERTON["sigma_V"]))
pd_neutra = float(pd_merton(V_final_verd, L, r, PARAMS_MERTON["sigma_V"]))

print(f"PD física (deriva μ = {mu:.0%}):        {pd_fisica:.4%}")
print(f"PD neutra ao risco (r = {r:.0%}):      {pd_neutra:.4%}")
print(f"razão:                                {pd_neutra / pd_fisica:.2f}×")"""
    ),
    md(
        """A PD neutra ao risco é sistematicamente maior. A diferença não é erro
de modelo: é prêmio de risco. Investidores exigem retorno acima da taxa livre
de risco para carregar risco de default, e essa exigência aparece como uma
probabilidade inflada quando se desconta tudo à taxa livre de risco.

As duas têm usos distintos e não intercambiáveis:

- **PD física** — provisão contábil, capital regulatório, limite de crédito.
  Quer responder "com que frequência isso quebra?".
- **PD neutra ao risco** — precificação de CDS, títulos, derivativos de crédito.
  Quer responder "quanto vale hoje esse fluxo incerto?".

Usar uma no lugar da outra é um erro grave e comum. Uma provisão calculada com
PD neutra ao risco é conservadora por construção — e errada. O capítulo 10
retoma isso pelo lado do CDS.

## O que quebra fora do laboratório

A série que usamos foi gerada pelo próprio modelo. Isso torna o exercício
honesto quanto à estimação e silencioso quanto à especificação. Em dado real:

**A estrutura de capital não é uma dívida com um vencimento.** É dívida curta,
longa, bancária, mercado de capitais, com covenants e garantias. A prática de
mercado — usar dívida de curto prazo mais metade da de longo — é uma convenção,
não um resultado. Vale testar a sensibilidade a ela.

**O default não espera o vencimento.** Empresas quebram por liquidez antes de
ficarem com ativo abaixo da dívida. Modelos de barreira permitem default a
qualquer momento, ao custo de mais parâmetros.

**Ativo não segue browniano geométrico.** Retornos reais têm caudas pesadas e
volatilidade que muda com o tempo. Justamente na cauda, onde o modelo é usado.

**Só funciona com empresa de capital aberto.** O que exclui a maior parte de
qualquer carteira de crédito brasileira.

## Ponte regulatória

O modelo estrutural é a base conceitual do arcabouço de capital, e isso costuma
passar despercebido. A fórmula de requerimento de capital do IRB parte de um
modelo de valor do ativo com um fator sistêmico — a mesma ideia deste capítulo,
estendida a uma carteira. O capítulo 7 constrói isso, e o 12 fecha.

Duas implicações práticas para validação:

**PD point-in-time.** A PD estrutural reage a preço de ação, portanto oscila com
o mercado e é fortemente pró-cíclica. O arcabouço de capital espera estimativas
com característica de ciclo mais longo. Usar saída estrutural direto como PD
regulatória exige uma etapa de suavização ou mapeamento que precisa ser
documentada e validada — não é detalhe de implementação.

**Poder discriminante versus nível.** Distância ao default costuma discriminar
muito bem, e por isso passa com folga em teste de AR/Gini. O nível de PD é
frágil, como este capítulo mediu. Um relatório que valida o modelo estrutural
apenas por poder discriminante repete o erro apontado no capítulo 1.

## Exercícios

1. Refaça a estimação supondo que a dívida foi medida com erro de ±20%. Quanto
   muda a PD? Compare a sensibilidade a esse erro com a sensibilidade ao erro em
   $\\sigma_V$. Qual dos dois insumos merece mais atenção do validador?

2. Estime $\\sigma_V$ com janelas de 60, 130 e 520 dias. Como o erro se comporta?
   Existe tensão entre janela longa (menos ruído) e janela curta (mais
   atualizada)? Como você escolheria em um comitê de modelos?

3. Substitua a normal por uma $t$ de Student com 4 graus de liberdade na
   tradução de DD em PD, mantendo a mesma distância. Quanto sobe a PD? Isso
   sugere o que sobre a prática da KMV de usar frequência empírica?

4. Gere uma empresa com dívida crescente ao longo do ano e verifique o que o
   procedimento iterativo, que supõe $L$ constante, devolve. O viés tem direção
   previsível?"""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap02_merton.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
