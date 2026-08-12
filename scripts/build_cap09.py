"""Monta o notebook do capítulo 9 (validação de modelos de carteira)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 9 — Validação de modelos de carteira

## O problema

O capítulo 8 validou um sistema de rating com oito mil observações por ano. Este
capítulo valida um modelo de carteira, e a diferença é aritmética antes de ser
metodológica.

O modelo de carteira produz uma **distribuição** de perda. Dessa distribuição,
observa-se **uma** realização por ano: a perda que de fato ocorreu. Vinte anos
de histórico dão vinte pontos para julgar se a cauda de 99,9% está correta.

Pense no que isso significa. O percentil 99,9% é, por definição, o evento que
ocorre uma vez a cada mil anos. Ninguém tem mil anos de dados. Ninguém terá.

Este capítulo faz duas coisas. Primeiro, apresenta o instrumento correto — a
transformada integral de probabilidade e o teste de Berkowitz, que extraem o
máximo de informação dos poucos pontos disponíveis. Depois, mede **quanto** eles
conseguem detectar, que é a pergunta que raramente se faz."""
    ),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credrisk.correlation.vasicek import taxa_condicional
from credrisk.portfolio.montecarlo import quantil_vasicek
from credrisk.validation.carteira import (
    poder_do_teste,
    teste_berkowitz,
    teste_excedencias,
    transformada_pit,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

PD_INC = 0.015
RHO_VERDADEIRO = 0.12"""
    ),
    md(
        """## A transformada integral de probabilidade

O instrumento resolve um problema aparentemente insolúvel: como testar uma
distribuição que muda todo ano — porque a carteira muda — com uma observação
anual?

A saída é olhar a **posição percentual** da perda realizada dentro da
distribuição prevista para aquele ano:

$$
u_t = \\hat F_t(L_t).
$$

Se o modelo está correto, os $u_t$ são uniformes em $(0,1)$ e independentes
entre si — qualquer que seja o formato de cada $\\hat F_t$, e mesmo que ele mude
todo ano. Essa invariância é o que torna a transformada útil: ela converte "a
distribuição está certa?" em "esta amostra é uniforme?".

A leitura dos $u_t$ é direta:

* valores concentrados perto de 1 — as perdas realizadas caem sistematicamente
  alto na distribuição prevista, ou seja, **o modelo subestima o risco**;
* valores concentrados no meio — o modelo é largo demais, superestima a
  dispersão;
* valores autocorrelacionados — sobra dependência temporal não modelada."""
    ),
    code(
        """def gerar_pit(n_anos, rho_real, rho_modelo, semente, n_cenarios=4000):
    # Perdas reais vêm de rho_real; o modelo do banco supõe rho_modelo.
    rng = np.random.default_rng(semente)
    perdas = taxa_condicional(rng.normal(0, 1, n_anos), PD_INC, rho_real)
    cenarios = [
        taxa_condicional(rng.normal(0, 1, n_cenarios), PD_INC, rho_modelo)
        for _ in range(n_anos)
    ]
    return transformada_pit(perdas, cenarios)


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
casos = [
    ("modelo correto", RHO_VERDADEIRO, RHO_VERDADEIRO),
    ("subestima o risco (ρ=0,04)", RHO_VERDADEIRO, 0.04),
    ("superestima o risco (ρ=0,35)", RHO_VERDADEIRO, 0.35),
]
for ax, (titulo, rv, rm) in zip(axes, casos, strict=True):
    u = gerar_pit(400, rv, rm, semente=0)
    ax.hist(u, bins=15, alpha=0.85)
    ax.axhline(400 / 15, ls="--", color="#c1553b", lw=1.5)
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlabel("u")
axes[0].set_ylabel("anos")
plt.show()"""
    ),
    md(
        """Com quatrocentos anos simulados, os três casos são inconfundíveis. O
modelo correto produz histograma plano; o que subestima o risco empilha massa à
direita; o que superestima empilha no meio.

Guarde essa nitidez, porque ela vai desaparecer.

## O teste de Berkowitz

Testar uniformidade diretamente funciona, mas desperdiça poder. Berkowitz
propôs transformar para a escala normal, $z_t = \\Phi^{-1}(u_t)$, e testar por
razão de verossimilhanças a hipótese conjunta

$$
\\mu = 0, \\qquad \\sigma^2 = 1, \\qquad \\varphi = 0
$$

em um AR(1) ajustado aos $z_t$. Trabalhar na escala normal dá mais peso às
observações de cauda, que é onde um modelo de capital erra e importa.

Os três parâmetros têm leitura de negócio imediata: $\\mu$ mede erro de nível,
$\\sigma$ mede erro de dispersão, e $\\varphi$ denuncia dependência temporal que o
modelo não captura."""
    ),
    code(
        """for titulo, rv, rm in casos:
    u = gerar_pit(400, rv, rm, semente=0)
    r = teste_berkowitz(u)
    print(f"{titulo}")
    print(f"   μ = {r.media:+.3f} · σ = {r.desvio:.3f} · φ = {r.autocorrelacao:+.3f}"
          f" · p-valor = {r.p_valor:.4f} · rejeita = {r.rejeita}")"""
    ),
    md(
        """O diagnóstico é preciso, e cada parâmetro aponta **qual** é o defeito, não
apenas que existe um. O modelo que subestima o risco tem $\\sigma$ bem acima de 1,
porque as perdas reais se espalham mais do que ele prevê; o que superestima tem
$\\sigma$ perto da metade, porque as perdas reais se concentram no meio de uma
distribuição larga demais. O modelo correto fica com $\\mu$ perto de zero e
$\\sigma$ perto de um, como manda a teoria.

Nada disso é garantido em uma realização específica: mesmo com quatrocentos
anos, o teste rejeita o modelo correto em torno de uma vez a cada quinze. A
tabela de erro tipo I adiante mede exatamente isso.

## Agora com o número real de anos

Tudo acima usou quatrocentos anos. Vamos repetir com vinte — um histórico
excelente para padrões reais."""
    ),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
for ax, (titulo, rv, rm) in zip(axes, casos, strict=True):
    u = gerar_pit(20, rv, rm, semente=7)
    ax.hist(u, bins=8, alpha=0.85)
    ax.axhline(20 / 8, ls="--", color="#c1553b", lw=1.5)
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlabel("u")
axes[0].set_ylabel("anos")
plt.show()

for titulo, rv, rm in casos:
    u = gerar_pit(20, rv, rm, semente=7)
    r = teste_berkowitz(u)
    print(f"{titulo:32s} p-valor = {r.p_valor:.3f}  rejeita = {r.rejeita}")"""
    ),
    md(
        """Os três histogramas são visualmente indistinguíveis. Vinte pontos não
desenham uma distribuição.

## Quanto o teste realmente detecta

A pergunta que decide se a validação tem valor: com o número de anos que existe,
qual a probabilidade de detectar um modelo errado?

Primeiro é preciso confirmar que o teste não rejeita modelos corretos. Depois,
medir o poder contra erros de magnitude conhecida."""
    ),
    code(
        """linhas = []
for n_anos in [10, 20, 50]:
    erro_tipo_1 = poder_do_teste(
        lambda k, n=n_anos: gerar_pit(n, RHO_VERDADEIRO, RHO_VERDADEIRO, 100 + k),
        n_repeticoes=150,
    )
    poder_baixo = poder_do_teste(
        lambda k, n=n_anos: gerar_pit(n, RHO_VERDADEIRO, 0.06, 500 + k),
        n_repeticoes=150,
    )
    poder_alto = poder_do_teste(
        lambda k, n=n_anos: gerar_pit(n, RHO_VERDADEIRO, 0.24, 900 + k),
        n_repeticoes=150,
    )
    linhas.append({
        "anos": n_anos,
        "rejeita modelo correto": erro_tipo_1,
        "detecta ρ=0,06 (metade)": poder_baixo,
        "detecta ρ=0,24 (dobro)": poder_alto,
    })
pd.DataFrame(linhas).set_index("anos").round(3)"""
    ),
    md(
        """A primeira coluna é boa notícia: o teste respeita o nível nominal, não
sai rejeitando modelos corretos. Ao contrário do que vimos no capítulo 8, aqui o
instrumento está bem construído.

As outras duas são o problema. Com vinte anos, um modelo que usa **metade** da
correlação verdadeira escapa em parcela substancial das vezes. Com dez anos, é
mais provável escapar do que ser pego.

E convém saber quanto custa esse erro que passa despercebido."""
    ),
    code(
        """def capital(rho, lgd=0.45):
    return (quantil_vasicek(PD_INC, rho, 0.999) - PD_INC) * lgd


tabela = pd.DataFrame({
    "ρ": [0.06, RHO_VERDADEIRO, 0.24],
    "cenário": ["modelo usa metade", "verdadeiro", "modelo usa o dobro"],
})
tabela["capital (% da exposição)"] = [capital(r) * 100 for r in tabela["ρ"]]
tabela["erro vs verdadeiro"] = tabela["capital (% da exposição)"] / (
    capital(RHO_VERDADEIRO) * 100
) - 1
tabela.set_index("cenário").round(4)"""
    ),
    md(
        """Um modelo que subestima o capital em quase metade tem chance
substancial de passar no backtest com vinte anos de dados.

Isso não é falha do teste de Berkowitz — ele é o melhor instrumento disponível
para o problema. É limite de informação: vinte observações não distinguem essas
duas distribuições, e nenhum teste estatístico contorna isso.

## Comparando os testes disponíveis

Vale ver o que se perde usando instrumentos mais simples. O mais comum na
prática é contar violações do quantil."""
    ),
    code(
        """def gerar(k):
    return gerar_pit(20, RHO_VERDADEIRO, 0.06, semente=1500 + k)


comparacao = pd.Series({
    "Berkowitz (verossimilhança)": poder_do_teste(gerar, 150, teste="berkowitz"),
    "Kolmogorov-Smirnov": poder_do_teste(gerar, 150, teste="ks"),
    "contagem de violações": poder_do_teste(gerar, 150, teste="excedencias"),
}, name="poder com 20 anos")
comparacao.round(3)"""
    ),
    code(
        """u = gerar_pit(20, RHO_VERDADEIRO, 0.06, semente=3)
r = teste_excedencias(u, nivel=0.99)
print(f"anos observados: {r['anos']}")
print(f"violações do percentil 99%: {r['violacoes']}")
print(f"violações esperadas sob o modelo: {r['esperadas']:.2f}")"""
    ),
    md(
        """A contagem de violações é o teste de menor poder dos três, e a razão é
aritmética: com vinte anos e nível de 99%, o número esperado de violações é
0,2. Observar zero violações é perfeitamente compatível com um modelo bom **e**
com um modelo péssimo. O teste quase não tem como rejeitar.

Ainda assim, é o teste mais reportado, porque é o mais fácil de explicar. Vale
guardar a assimetria: **passar num teste sem poder não é evidência de que o
modelo está certo.** É evidência de que o teste não consegue distinguir — o que
é uma afirmação bem diferente, e frequentemente apresentada ao comitê como se
fosse aprovação.

## O que fazer diante disso

Se o backtest direto tem pouco poder, a validação não pode se apoiar só nele.
O que resta, em ordem de utilidade:

**Validar os componentes.** PD, LGD e correlação têm cada um sua evidência
própria, com muito mais observações que a distribuição agregada. É o conteúdo
dos capítulos 1 a 6.

**Validar a implementação contra respostas fechadas.** O capítulo 7 mostrou que
o simulador reproduz a fórmula de Vasicek na carteira homogênea granular. Isso
não valida as premissas, mas descarta erro de código — e erro de código em
simulador de crédito é silencioso.

**Análise de sensibilidade.** Se o teste não distingue $\\rho = 0{,}06$ de
$\\rho = 0{,}12$, reporte o capital nas duas hipóteses. A incerteza que o teste
não resolve deve aparecer no número apresentado, não ser escondida por ele.

**Teste de estresse com cenários nomeados.** Em vez de perguntar se a cauda de
99,9% está certa, pergunte quanto o modelo prevê de perda em um cenário
específico e compare com a experiência histórica ou com pares.

**Reconhecer o limite por escrito.** Um relatório que diz "o modelo não foi
rejeitado, e o teste tem poder de 60% contra erro de metade na correlação" é
honesto. Um que diz apenas "o modelo não foi rejeitado" induz o comitê a erro.

## O que quebra fora do laboratório

**A carteira muda todo ano.** A distribuição prevista para 2015 é de outra
carteira que a de 2025. A transformada lida com isso por construção, mas a
hipótese de independência entre anos fica mais frágil quando há tendência de
composição.

**A perda observada não é limpa.** Recuperações se estendem por anos, e a perda
"do ano" depende de convenções contábeis. Comparar uma perda contábil com uma
distribuição econômica simulada mistura dois objetos.

**O modelo é recalibrado no meio.** Se os parâmetros mudam ao longo da janela, os
$u_t$ não vêm todos do mesmo modelo, e o teste perde sentido formal.

**Um ano ruim domina tudo.** Com vinte pontos, uma única crise no meio da amostra
determina o resultado do teste. O intervalo de confiança do próprio teste é
largo, e raramente reportado.

## Ponte regulatória

**Backtesting é exigido; poder é raramente reportado.** O arcabouço de validação
espera comparação sistemática entre previsto e realizado. Reportar apenas o
resultado do teste, sem a análise de poder deste capítulo, transmite mais
segurança do que a evidência sustenta.

**Uso do modelo além do backtest.** Justamente porque o backtest agregado é
fraco, a validação de modelo de carteira se apoia mais em validação de
componentes, verificação de implementação e análise de sensibilidade. Isso não é
contorno: é reconhecimento de onde a informação está.

**Capital econômico e teste de estresse.** Se o backtest não distingue hipóteses
de correlação que mudam o capital em quase metade, o resultado do teste de
estresse depende de premissa que o dado não determina — e a governança sobre
essa premissa importa mais que o teste.

**Documentar a limitação é parte do trabalho.** Um relatório que quantifica o
poder do próprio teste é mais defensável perante o supervisor que um que reporta
apenas "não rejeitado".

## Exercícios

1. Refaça a tabela de poder variando o número de cenários simulados por ano
   (500, 4.000, 50.000). O poder muda? O que isso diz sobre onde está o gargalo
   — no modelo ou nos dados?

2. Introduza autocorrelação nas perdas reais (fator sistêmico AR(1)) mantendo a
   distribuição marginal correta. O componente $\\varphi$ do Berkowitz detecta? Em
   quantos anos?

3. Simule um modelo que acerta o corpo da distribuição e erra só a cauda além de
   99%. Qual teste detecta com vinte anos? Alguma coisa detecta?

4. Escreva o parágrafo de conclusão de um relatório de validação para um modelo
   que não foi rejeitado com quinze anos de dados, incluindo a análise de poder.
   Compare com o parágrafo que você escreveria sem ela."""
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
    destino = (
        pathlib.Path(__file__).resolve().parents[1]
        / "book"
        / "cap09_validacao_carteira.ipynb"
    )
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
