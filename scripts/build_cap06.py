"""Monta o notebook do capítulo 6 (estimação de correlação de ativos)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 6 — Correlação de ativos

## O problema

Suponha uma carteira de dez mil operações, cada uma com PD de 1,5%. Se os
defaults fossem independentes, a lei dos grandes números resolveria o problema
do banco: a taxa realizada ficaria colada em 1,5% todo ano, o desvio-padrão
seria de 0,12 ponto percentual, e não haveria por que manter capital contra
risco de crédito — bastaria provisionar a média.

Não é o que acontece. Bancos quebram por crédito, e quebram todos juntos, nos
mesmos anos. A razão é que os defaults **não** são independentes: existe um
fator comum — a economia — que empurra todos os devedores na mesma direção ao
mesmo tempo.

A correlação de ativos $\\rho$ é o parâmetro que mede a intensidade desse fator
comum. É ela, e não a PD, que determina o tamanho da cauda da distribuição de
perda. E, portanto, o capital.

Este capítulo estima $\\rho$. A conclusão antecipada, para você ler o resto com a
expectativa certa: **é o parâmetro pior estimado de todo o curso**, e a
consequência disso em capital é grande."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.correlation.vasicek import (
    densidade_vasicek,
    estimar_ml,
    estimar_momentos,
    intervalo_perfil,
    log_verossimilhanca,
    log_verossimilhanca_simples,
    taxa_condicional,
)
from credrisk.data.generators import PARAMS_FATOR, gerar_taxas_vasicek
from credrisk.portfolio.montecarlo import quantil_vasicek
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.5f}")

PD_V, RHO_V = PARAMS_FATOR["PD"], PARAMS_FATOR["RHO"]"""
    ),
    md(
        """## A formulação

O modelo de fator único, devido a Vasicek, descreve o valor do ativo
padronizado do devedor $i$ no ano $t$ como

$$
Z_{it} = \\sqrt{\\rho}\\,X_t + \\sqrt{1-\\rho}\\,\\varepsilon_{it},
$$

com $X_t$ o fator sistêmico e $\\varepsilon_{it}$ o risco próprio do devedor,
ambos normais padrão independentes. Há default quando $Z_{it} < \\Phi^{-1}(PD)$.

A correlação entre os valores de ativo de dois devedores quaisquer é exatamente
$\\rho$ — daí o nome. Condicionando ao fator, os defaults voltam a ser
independentes, e a taxa de default condicional é

$$
p(x) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) - \\sqrt{\\rho}\\,x}{\\sqrt{1-\\rho}}\\right).
$$

Essa condicionalidade é o que torna tudo tratável: a integral que aparece
adiante é unidimensional, por mais devedores que a carteira tenha."""
    ),
    code(
        """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

x = np.linspace(-3.2, 3.2, 300)
for rho in [0.03, 0.12, 0.30]:
    ax1.plot(x, taxa_condicional(x, PD_V, rho) * 100, label=f"ρ = {rho:.2f}")
ax1.axhline(PD_V * 100, ls=":", color="#7a8b8b")
ax1.set_title("Taxa de default condicional ao fator")
ax1.set_xlabel("fator sistêmico X (alto = bom estado)")
ax1.set_ylabel("taxa condicional (%)")
ax1.legend()

t = np.linspace(0.0005, 0.10, 400)
for rho in [0.03, 0.12, 0.30]:
    ax2.plot(t * 100, densidade_vasicek(t, PD_V, rho), label=f"ρ = {rho:.2f}")
ax2.axvline(PD_V * 100, ls=":", color="#7a8b8b", label="PD média")
ax2.set_title("Densidade da taxa de default")
ax2.set_xlabel("taxa de default (%)")
ax2.legend()
plt.show()"""
    ),
    md(
        """Os dois painéis dizem a mesma coisa de formas diferentes. Correlação
maior deixa a taxa condicional mais sensível ao fator, e a densidade resultante
mais assimétrica: mais massa concentrada abaixo da média e uma cauda direita mais
longa.

Isso tem uma consequência que vale reter, porque contraria a intuição: **na
maioria dos anos a taxa de default fica abaixo da PD média**. A média é puxada
por poucos anos muito ruins. Quem calibra PD pela mediana de uma série curta
subestima sistematicamente — e nem percebe, porque a série "parece" bem
comportada.

## Os dados

Vinte e cinco anos de taxa de default de uma carteira de mil devedores. É uma
base generosa para padrões reais."""
    ),
    code(
        """dados = gerar_taxas_vasicek(n_anos=25, n_obrigados=1000)
observado = dados[["Ano", "N", "Defaults", "taxa_observada"]]

fig, ax = plt.subplots()
ax.bar(dados["Ano"], dados["taxa_observada"] * 100, width=0.7)
ax.axhline(PD_V * 100, color="#c1553b", ls="--", lw=1.5, label="PD verdadeira")
ax.set_title("Taxa de default anual observada")
ax.set_xlabel("ano")
ax.set_ylabel("%")
ax.legend()
plt.show()

print(f"média {dados['taxa_observada'].mean():.2%} · "
      f"mínima {dados['taxa_observada'].min():.2%} · "
      f"máxima {dados['taxa_observada'].max():.2%}")"""
    ),
    md(
        """Toda a informação sobre $\\rho$ está na **variação entre as barras**.
Não no número de devedores — em quantos anos foram observados.

Vale internalizar isso antes de seguir: uma carteira de dez milhões de contratos
observada por vinte anos tem, para efeito de estimar correlação, **vinte
observações**. Aumentar a carteira reduz o ruído de cada barra, mas não cria
barras novas.

## Estimador de momentos

O caminho mais curto: a média das taxas estima a PD, e a variância delas contém
$\\rho$. Igualando a variância observada à teórica e invertendo numericamente,
sai uma estimativa."""
    ),
    code(
        """mm = estimar_momentos(dados["taxa_observada"].to_numpy())
pd.DataFrame({
    "verdadeiro": {"PD": PD_V, "ρ": RHO_V},
    "momentos": {"PD": mm["PD"], "ρ": mm["RHO"]},
})"""
    ),
    md(
        """Razoável, mas o método joga fora informação: usa apenas dois momentos,
e trata a taxa observada como se fosse a taxa verdadeira — ignorando que ela
carrega ruído binomial, o que infla artificialmente a variância e portanto
$\\rho$.

## Máxima verossimilhança, e a armadilha da quadratura

O estimador correto trata a carteira como finita. A probabilidade de observar
$d$ defaults entre $N$ devedores num ano é

$$
\\Pr(D = d) = \\int \\binom{N}{d}\\,p(x)^d\\,\\big(1-p(x)\\big)^{N-d}\\,\\phi(x)\\,dx,
$$

integral unidimensional que separa corretamente a variação vinda do fator da
variação vinda do azar binomial.

A integral não tem forma fechada e precisa ser resolvida numericamente. O
instrumento natural é a quadratura de Gauss-Hermite, que aproxima integrais
contra a densidade normal por uma soma ponderada em nós fixos.

**E é aqui que este capítulo quase deu errado.** Vale ver por quê, porque o modo
de falha é instrutivo."""
    ),
    code(
        """d, n = np.array([25.0]), np.array([1000.0])

comparacao = []
for rho in [0.05, 0.12, 0.30, 0.50]:
    comparacao.append({
        "ρ": rho,
        "nós fixos (64)": log_verossimilhanca_simples(d, n, PD_V, rho, n_nos=64),
        "adaptativa (30)": log_verossimilhanca(d, n, PD_V, rho, n_nos=30),
    })
comparacao = pd.DataFrame(comparacao).set_index("ρ")
comparacao["diferença"] = comparacao["nós fixos (64)"] - comparacao["adaptativa (30)"]
comparacao.round(6)"""
    ),
    md(
        """Com correlação baixa as duas concordam. Com $\\rho = 0{,}50$ diferem na
primeira casa decimal — e a versão com nós fixos, que usa **mais que o dobro de
nós**, é a errada.

A causa é geométrica. Com mil devedores, apenas uma faixa muito estreita de $x$
é compatível com observar exatamente 25 defaults: o integrando é um pico agudo.
Os nós de Gauss-Hermite ficam espalhados sobre toda a normal padrão, e quando o
pico é mais estreito que o espaçamento entre nós, a quadratura simplesmente
passa por cima dele.

A correção é deslocar os nós para o pico — localizar o modo do integrando, medir
sua curvatura e concentrar a quadratura ali. É o mesmo procedimento usado em
modelos mistos generalizados, e resolve com trinta nós o que centenas de nós
fixos não resolvem.

O que torna esse erro perigoso não é a magnitude. É que **nada sinaliza**. A
função devolve um número finito, plausível, com a mesma cara de sempre, e o
otimizador converge alegremente para um $\\rho$ errado. Descobri isso conferindo
contra integração numérica direta; sem essa conferência, o capítulo inteiro
estaria errado e pareceria certo.

Vale registrar um detalhe adicional, que reforça o ponto: a própria integração
numérica de referência do SciPy, chamada sobre o intervalo $[-8, 8]$, também
perde o pico em carteiras grandes e devolve um valor errado por várias unidades.
Precisão numérica não é automática só porque a biblioteca é boa."""
    ),
    code(
        """ml = estimar_ml(dados["Defaults"].to_numpy(), dados["N"].to_numpy())

pd.DataFrame({
    "verdadeiro": {"PD": PD_V, "ρ": RHO_V},
    "momentos": {"PD": mm["PD"], "ρ": mm["RHO"]},
    "máx. verossimilhança": {"PD": ml.PD, "ρ": ml.RHO},
})"""
    ),
    md(
        """A estimativa de $\\rho$ erra por uma margem considerável. E, ao
contrário dos capítulos anteriores, aqui a estimativa pontual é quase
irrelevante — o que importa é quanto ela poderia ter sido diferente.

## O intervalo é o resultado

A verossimilhança em $\\rho$ é bastante assimétrica com poucos anos, então um
intervalo simétrico do tipo estimativa ± 1,96 erro-padrão dá resposta ruim, e
chega a incluir valores negativos. O caminho correto é perfilar: para cada
$\\rho$ da grade, maximizar sobre a PD, e reter os valores cuja perda de
verossimilhança seja pequena."""
    ),
    code(
        """inf, sup, grade, perfil = intervalo_perfil(
    dados["Defaults"].to_numpy(), dados["N"].to_numpy()
)

fig, ax = plt.subplots()
ax.plot(grade, perfil - perfil.max(), lw=2)
ax.axhline(-1.92, color="#7a8b8b", ls=":", label="corte de 95%")
ax.axvline(RHO_V, color="#c1553b", lw=2, label="ρ verdadeiro")
ax.axvspan(inf, sup, alpha=0.12, color="#1f4e5f", label="IC 95%")
ax.set_title("Verossimilhança perfilada em ρ")
ax.set_xlabel("ρ")
ax.set_ylabel("log-verossimilhança relativa")
ax.set_ylim(-8, 0.5)
ax.legend()
plt.show()

print(f"ρ estimado: {ml.RHO:.4f}")
print(f"IC de 95%:  [{inf:.4f}, {sup:.4f}]   (razão superior/inferior = {sup/inf:.1f}×)")"""
    ),
    md(
        """Vinte e cinco anos de dados não determinam $\\rho$. O intervalo cobre
uma faixa em que o limite superior é várias vezes o inferior, e a curva é
visivelmente assimétrica — cai rápido à esquerda e devagar à direita.

Isso não é defeito do estimador: é o teto de informação da amostra.

## Quanto isso custa em capital

A pergunta que interessa a quem decide não é o intervalo de $\\rho$. É o
intervalo de **capital** que ele implica."""
    ),
    code(
        """# Capital como fração da exposição, pela fórmula de Vasicek.
def capital_por_rho(rho: float, lgd: float = 0.45, nivel: float = 0.999) -> float:
    return (quantil_vasicek(PD_V, rho, nivel) - PD_V) * lgd


faixa = pd.DataFrame({
    "ρ": [inf, ml.RHO, RHO_V, sup],
    "cenário": ["limite inferior do IC", "estimativa pontual",
                "valor verdadeiro", "limite superior do IC"],
})
faixa["capital (% da exposição)"] = [capital_por_rho(r) * 100 for r in faixa["ρ"]]
faixa.set_index("cenário").round(4)"""
    ),
    code(
        """grade_rho = np.linspace(0.02, 0.30, 200)
fig, ax = plt.subplots()
ax.plot(grade_rho, [capital_por_rho(r) * 100 for r in grade_rho], lw=2)
ax.axvspan(inf, sup, alpha=0.12, color="#1f4e5f")
ax.axvline(RHO_V, color="#c1553b", lw=1.8, ls="--", label="ρ verdadeiro")
ax.set_title("Capital exigido em função da correlação")
ax.set_xlabel("ρ")
ax.set_ylabel("capital (% da exposição)")
ax.legend()
plt.show()

razao = capital_por_rho(sup) / capital_por_rho(inf)
print(f"dentro do IC de 95%, o capital varia por um fator de {razao:.1f}×")"""
    ),
    md(
        """Esta é a mensagem do capítulo, e ela é desconfortável.

Com vinte e cinco anos de dados — mais do que a maioria das carteiras tem — a
incerteza sobre $\\rho$ se traduz num intervalo de capital cujo extremo superior
é quase o triplo do inferior. Não é diferença de segunda casa decimal: é a
diferença entre um banco confortável e um banco em dificuldade.

E o número reportado ao comitê é um só, sem intervalo.

Isso reenquadra uma crítica comum. Muita gente reclama que o arcabouço
regulatório **fixa** as correlações por classe de ativo em vez de deixar cada
banco estimar. À luz do gráfico acima, a fixação parece menos arbitrariedade e
mais reconhecimento de que a estimação livre desse parâmetro produziria
dispersão enorme entre bancos com carteiras parecidas — e incentivo óbvio para
escolher o extremo baixo do intervalo.

## O que quebra fora do laboratório

**Um fator não bastam.** O modelo supõe um único fator comum. Carteiras reais
têm estrutura setorial e geográfica: construção civil e varejo não respondem ao
mesmo choque na mesma intensidade. Modelos multifatoriais captam isso, ao custo
de estimar uma matriz inteira com os mesmos vinte anos.

**A correlação não é constante.** Há evidência de que a dependência aumenta
justamente em crises — exatamente quando o modelo mais importa. A cópula
gaussiana implícita aqui tem dependência de cauda nula, uma propriedade que os
dados de 2008 contradizem, e que reaparece com força no capítulo 11.

**A carteira muda de composição.** A série de vinte anos mistura carteiras
diferentes. Parte da variação atribuída ao fator sistêmico pode ser mudança de
perfil de originação.

**Rho e PD são estimados juntos e se confundem.** Uma série com poucos anos ruins
é compatível tanto com PD baixa e $\\rho$ alto quanto com PD média e $\\rho$
moderado. A verossimilhança perfilada mostra essa troca; o número pontual, não.

## Ponte regulatória

**Correlações prescritas.** No arcabouço de capital de risco de crédito, as
correlações são dadas por fórmula, geralmente decrescentes na PD e diferenciadas
por classe de exposição. Este capítulo explica por que não são estimadas
livremente.

**Validação de modelo interno de capital econômico.** Quando o banco usa modelo
próprio para capital econômico ou para testes de estresse, $\\rho$ volta a ser
estimado — e aí o intervalo deste capítulo é obrigação de reporte, não
refinamento. Um modelo de capital econômico sem análise de sensibilidade a
$\\rho$ está incompleto.

**Margem de conservadorismo.** Diante de incerteza dessa magnitude, o
conservadorismo não é preferência de gosto: é a resposta correta a um parâmetro
que a amostra não determina.

**Consistência entre usos.** A correlação usada em capital econômico, em teste de
estresse e em precificação deveria ser a mesma, ou as diferenças deveriam ser
justificadas. Na prática, cada área calibra a sua e ninguém compara.

## Exercícios

1. Refaça a estimação com 10, 25, 50 e 100 anos de dados. Como a largura do
   intervalo de $\\rho$ encolhe? A que taxa? Quantos anos seriam necessários para
   um intervalo de largura aceitável?

2. Fixe os anos em 25 e varie o número de devedores: 200, 1.000, 50.000. Quanto
   melhora a estimativa de $\\rho$? Confirme numericamente a afirmação de que
   tamanho de carteira quase não ajuda.

3. Gere dados com $\\rho = 0{,}12$ mas estime supondo carteira infinitamente
   granular (use a densidade de Vasicek em vez da verossimilhança binomial). O
   viés tem direção previsível? Por quê?

4. Simule uma carteira em que metade dos devedores tem $\\rho = 0{,}05$ e a outra
   metade $\\rho = 0{,}25$, e estime um $\\rho$ único. O valor estimado fica no
   meio? O que isso diz sobre agregar segmentos heterogêneos num modelo de fator
   único?"""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap06_correlacao.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
