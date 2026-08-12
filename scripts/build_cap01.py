"""Monta o notebook do capítulo 1 a partir de células declaradas em Python.

Manter o notebook como código gerado tem duas vantagens práticas: o diff no git
fica legível e o texto pode ser revisado sem abrir o Jupyter. Rode com:

    python scripts/build_cap01.py
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 1 — Escore de crédito com logit

## O problema

Um banco precisa decidir, hoje, quanto de capital e de provisão alocar a um
devedor cujo default só se revelará no futuro. Toda a máquina de risco de
crédito — provisão contábil, capital regulatório, precificação, limite de
alçada — parte de um número: a probabilidade de que aquele devedor entre em
default dentro de um horizonte definido.

Estimar esse número a partir de características observáveis do devedor é o
problema deste capítulo. A escolha da regressão logística para resolvê-lo não é
histórica nem estética: o alvo é binário, queremos uma probabilidade no
intervalo $(0,1)$, e queremos coeficientes interpretáveis o bastante para
sustentar uma discussão com o regulador e com a área de negócio.

O que este capítulo persegue não é ajustar o modelo — isso são três linhas de
código. É responder: **o modelo acertou?** E, mais desconfortável: **como você
saberia se não tivesse acertado?**"""
    ),
    code(
        """import pandas as pd

from credrisk.data.generators import (
    COEFS_VERDADEIROS,
    PREDITORES_CAP01,
    gerar_painel_scoring,
)
from credrisk.data.registry import carregar
from credrisk.scoring.logit import (
    ajustar,
    ajustar_firth,
    auc,
    comparar_com_verdadeiro,
    prever_pd,
    razao_de_acuracia,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""
    ),
    md(
        """## Os dados

A base é um painel empresa-ano com cinco razões financeiras no espírito do
índice de Altman, e um indicador de default. É sintética, gerada por
`credrisk.data.generators` — e essa é a peça central do curso, não um contorno.

O processo gerador tem três camadas deliberadas:

1. **efeito de empresa** — cada empresa tem seu nível próprio e persistente de
   cada razão;
2. **fator sistêmico anual** — um choque macro comum a todas em cada ano;
3. **ruído idiossincrático com memória** — variação própria, autocorrelacionada.

Além disso, a empresa **sai do painel ao entrar em default**. O painel é
desbalanceado, como o de qualquer base de falência real.

Cada uma dessas escolhas tem consequência estatística que aparecerá adiante.
Nenhuma é decorativa."""
    ),
    code(
        """dados = carregar("cap01_scoring")
print(f"{len(dados):,} observações · {dados['ID'].nunique()} empresas · "
      f"{dados['Ano'].min()}–{dados['Ano'].max()}")
print(f"defaults: {int(dados['Default'].sum())} "
      f"({dados['Default'].mean():.2%} das observações)")
dados.head()"""
    ),
    code(
        """import matplotlib.pyplot as plt

por_ano = dados.groupby("Ano")["Default"].agg(["mean", "size"])

fig, ax = plt.subplots()
ax.bar(por_ano.index, por_ano["mean"] * 100, width=0.65)
ax.axhline(dados["Default"].mean() * 100, ls="--", lw=1.2, color="#c1553b",
           label="média do período")
ax.set_title("Taxa de default anual")
ax.set_ylabel("% das empresas ativas")
ax.set_xlabel("")
ax.legend()
plt.show()"""
    ),
    md(
        """A taxa de default não é constante: varia de menos de 1% a mais de 5%
conforme o ano. Isso não é ruído amostral — é o fator sistêmico do gerador
aparecendo. Guarde essa figura: ela é a razão de existirem os capítulos 6 e 7.
Se os defaults fossem independentes entre devedores, a perda de uma carteira
grande seria praticamente determinística e não haveria capital econômico a
discutir.

## A formulação

Seja $y_{it} \\in \\{0,1\\}$ o indicador de default da empresa $i$ no ano $t$, e
$x_{it}$ o vetor de razões financeiras. O modelo logit postula

$$
\\Pr(y_{it}=1 \\mid x_{it}) = \\Lambda(x_{it}'\\beta) = \\frac{1}{1+e^{-x_{it}'\\beta}}.
$$

A estimação por máxima verossimilhança maximiza

$$
\\ell(\\beta) = \\sum_{i,t} \\Big[ y_{it}\\log \\Lambda(x_{it}'\\beta)
+ (1-y_{it})\\log\\big(1-\\Lambda(x_{it}'\\beta)\\big) \\Big].
$$

A log-verossimilhança é globalmente côncava, então o ótimo é único e o
Newton-Raphson converge sem drama. A interpretação do coeficiente é em log-odds:
$\\beta_k$ é a variação no log da razão de chances por unidade de $x_k$.

Duas propriedades da MLE que o capítulo vai testar, em vez de assumir:

- ela é **consistente**, mas apenas **assintoticamente** não-enviesada;
- os erros-padrão da matriz de informação supõem observações **independentes**."""
    ),
    code(
        """mle = ajustar(dados, PREDITORES_CAP01)
mle.summary2().tables[1].round(3)"""
    ),
    md(
        """## O modelo acertou?

Aqui o curso faz o que nenhum curso com dado real consegue: abrir o gabarito.
Os coeficientes que geraram os defaults estão em `COEFS_VERDADEIROS`."""
    ),
    code(
        """comparacao = comparar_com_verdadeiro(mle, COEFS_VERDADEIROS)
comparacao.round(3)"""
    ),
    md(
        """Olhe a coluna `estimado` contra `verdadeiro` antes de olhar qualquer
outra coisa.

O coeficiente de `EBIT/TA` foi plantado em $-8{,}00$ e estimado em algo perto de
metade disso. O de `S/TA` praticamente desapareceu. E a coluna `ic95_cobre`
mostra que o intervalo de confiança de 95% **não cobre o valor verdadeiro** em
três dos seis coeficientes.

Se este fosse um modelo em produção, você teria concluído que a rentabilidade
operacional importa metade do que de fato importa, e que giro do ativo não
importa. Nenhum diagnóstico usual denunciaria isso: o modelo converge, os sinais
estão corretos, e as variáveis relevantes são significativas. O modelo está
errado de um jeito que o relatório de ajuste não mostra.

Antes de acusar o estimador, a pergunta certa é: **isso é viés ou é azar?** A
distinção é decisiva, porque viés se corrige com método e azar só se corrige com
mais dados. Com base sintética, dá para responder: basta reestimar o modelo em
muitas amostras diferentes do mesmo processo gerador e olhar a distribuição."""
    ),
    code(
        """replicas = []
for semente in range(1000, 1050):
    amostra = gerar_painel_scoring(semente=semente)
    replicas.append(ajustar(amostra, PREDITORES_CAP01).params)

replicas = pd.DataFrame(replicas)
verdade = pd.Series(COEFS_VERDADEIROS)[replicas.columns]

resumo_mc = pd.DataFrame({
    "verdadeiro": verdade,
    "média das réplicas": replicas.mean(),
    "desvio entre réplicas": replicas.std(),
    "viés": replicas.mean() - verdade,
})
resumo_mc.round(3)"""
    ),
    code(
        """fig, ax = plt.subplots()
ax.hist(replicas["EBIT/TA"], bins=18, alpha=0.85)
ax.axvline(COEFS_VERDADEIROS["EBIT/TA"], color="#c1553b", lw=2,
           label="verdadeiro (−8,00)")
ax.axvline(mle.params["EBIT/TA"], color="#1f4e5f", lw=2, ls="--",
           label=f"nossa amostra ({mle.params['EBIT/TA']:.2f})")
ax.set_title("Coeficiente de EBIT/TA em 50 amostras do mesmo processo")
ax.set_xlabel("coeficiente estimado")
ax.set_ylabel("réplicas")
ax.legend()
plt.show()"""
    ),
    md(
        """O veredito é claro: em média o estimador acerta o alvo — o viés é
pequeno perto do desvio entre réplicas. O problema não é o método. O problema é
que **uma amostra só não é suficiente**, e a nossa caiu na cauda esquerda da
distribuição.

Esse é o resultado mais importante do capítulo, e ele é desconfortável. O banco
tem uma amostra. Não tem cinquenta. O histograma acima é a incerteza que existe
mas que o relatório de estimação não mostra — porque o relatório reporta o
ponto, e o ponto é um sorteio.

A largura daquela distribuição tem uma causa mensurável:"""
    ),
    code(
        """n_eventos = int(dados["Default"].sum())
k = len(PREDITORES_CAP01) + 1
print(f"eventos (defaults): {n_eventos}")
print(f"parâmetros estimados: {k}")
print(f"eventos por parâmetro: {n_eventos / k:.1f}")"""
    ),
    md(
        """A regra de bolso usual pede pelo menos 10 eventos por parâmetro, e há
literatura defendendo bem mais. Aqui estamos perto do limite inferior — com
4.130 linhas de dados. **O tamanho da amostra em risco de crédito não é o número
de linhas, é o número de defaults.** Uma base de dez milhões de contratos com
oitenta defaults é uma base pequena.

Este é o primeiro ponto de validação do curso, e ele não envolve nenhuma técnica
sofisticada: contar eventos.

## A hipótese de independência

Os erros-padrão acima supõem que as 4.130 observações são independentes. À
primeira vista não são: são cerca de 250 empresas observadas repetidamente, e
razões financeiras são persistentes — a empresa alavancada em 1990 provavelmente
ainda estava alavancada em 1991.

O reflexo treinado é agrupar os erros-padrão por empresa. Vamos fazer isso e
olhar o que muda — o resultado não é o esperado."""
    ),
    code(
        """agrupado = ajustar(dados, PREDITORES_CAP01, cluster="ID")

lado_a_lado = pd.DataFrame({
    "coef": mle.params,
    "ep_ingenuo": mle.bse,
    "ep_por_empresa": agrupado.bse,
})
lado_a_lado["inflacao"] = lado_a_lado["ep_por_empresa"] / lado_a_lado["ep_ingenuo"]
lado_a_lado.round(3)"""
    ),
    md(
        """Os coeficientes são idênticos — o agrupamento nunca muda o ponto
estimado, muda a incerteza declarada. Mas a coluna `inflacao` fica em torno de
1,0: **o erro-padrão praticamente não mudou.**

Se você esperava inflação, vale entender por que ela não veio, porque o motivo é
mais instrutivo que o resultado esperado. O agrupamento corrige correlação nos
**resíduos** dentro do grupo. Aqui cada empresa entra em default no máximo uma
vez e sai do painel em seguida — não há como os resíduos de uma mesma empresa se
correlacionarem, porque só existe um evento por empresa. A persistência do
painel está nas **covariáveis**, não nos resíduos, e covariável persistente não
viola a hipótese que o erro-padrão usa.

Registre o resultado negativo: aplicar a correção certa para o problema errado
não custa nada e não resolve nada. Onde o agrupamento morde de verdade é quando
o mesmo devedor pode entrar em default mais de uma vez na janela, ou quando ele
aparece com várias operações simultâneas. Uma carteira de varejo com múltiplos
contratos por CPF é o caso canônico: tratar contrato como observação
independente pode inflar a amostra efetiva em uma ordem de grandeza. A escolha
da chave de agrupamento — contrato, devedor, grupo econômico — é decisão de
modelagem, não detalhe de implementação, e ela decide quais variáveis passam no
corte de significância.

## Viés de evento raro: a correção de Firth

A teoria diz que a MLE do logit, embora consistente, é **enviesada em amostra
finita**, e que o viés cresce quando os eventos são raros. Firth (1993) propôs
penalizar a verossimilhança pelo *prior* de Jeffreys, o que reduz o viés a uma
ordem menor e, de quebra,
produz estimativa finita mesmo sob separação completa — situação em que a MLE
diverge para o infinito e o software devolve coeficientes absurdos com
erros-padrão gigantescos."""
    ),
    code(
        """firth = ajustar_firth(dados, PREDITORES_CAP01)

tres = pd.DataFrame({
    "verdadeiro": pd.Series(COEFS_VERDADEIROS),
    "mle": mle.params,
    "firth": firth.params,
}).loc[["CONST", *PREDITORES_CAP01]]
tres["|erro| mle"] = (tres["mle"] - tres["verdadeiro"]).abs()
tres["|erro| firth"] = (tres["firth"] - tres["verdadeiro"]).abs()
tres.round(3)"""
    ),
    md(
        """Com 82 eventos, a correção de Firth muda quase nada — coerente com o
Monte Carlo, que já havia mostrado viés pequeno. O problema dominante desta
amostra é **variância, não viés**, e nenhuma penalização conserta variância.

Vale registrar o resultado negativo: **nem toda correção sofisticada resolve o
seu problema.** Diagnosticar qual patologia domina a sua amostra é mais útil do
que aplicar todas as correções disponíveis e reportar que foram aplicadas.

Onde o Firth é decisivo é sob separação, comum em carteiras de baixo default
(*low default portfolios*) — soberanos, grandes corporativos, project finance.
Em um segmento com três defaults em quinze anos, a MLE simplesmente não existe."""
    ),
    code(
        """separado = pd.DataFrame({
    "alavancagem": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
    "Default": [0, 0, 0, 1, 1, 1],
})
f = ajustar_firth(separado, ["alavancagem"])
print("Firth sob separação completa:")
print(f.resumo().round(3))"""
    ),
    md(
        """## Poder discriminante

Coeficiente correto e poder discriminante são coisas diferentes. Um modelo pode
ordenar bem os devedores e ainda assim ter coeficientes errados — e é
exatamente o que acontece aqui."""
    ),
    code(
        """pd_prevista = prever_pd(mle, dados, PREDITORES_CAP01)
alvo = dados["Default"].to_numpy()

print(f"AUC = {auc(pd_prevista, alvo):.4f}")
print(f"AR  = {razao_de_acuracia(pd_prevista, alvo):.4f}")

# Se o modelo tivesse sido estimado com os coeficientes verdadeiros:
class Oraculo:
    params = pd.Series(COEFS_VERDADEIROS)[["CONST", *PREDITORES_CAP01]]

pd_oraculo = prever_pd(Oraculo, dados, PREDITORES_CAP01)
print(f"\\nAUC do modelo com os coeficientes verdadeiros = "
      f"{auc(pd_oraculo, alvo):.4f}")"""
    ),
    md(
        """O modelo estimado discrimina quase tão bem quanto o modelo com os
coeficientes verdadeiros, apesar de errar `EBIT/TA` pela metade.

Isso não é curiosidade: é a razão pela qual **validação por poder discriminante
não detecta erro de calibração**. AUC e AR são invariantes a qualquer
transformação monótona da PD. Um modelo que multiplica todas as PDs por três tem
exatamente o mesmo AUC — e triplica a sua provisão. Os capítulos 8 e 9
desenvolvem a separação entre discriminação e calibração; aqui basta ver que ela
existe e que é grande.

## Ponte regulatória

Três coisas deste capítulo reaparecem literalmente no dia a dia de um validador
independente no Brasil:

**Suficiência de dados.** O arcabouço de ratings internos condiciona o uso de
estimativas próprias a histórico e representatividade suficientes. A leitura
usual foca no número de anos; este capítulo mostra que a métrica que morde é o
número de eventos por parâmetro. Uma janela longa com poucos defaults não
resolve o problema — só o disfarça.

**Carteiras de baixo default.** Segmentos com pouquíssimos defaults exigem
tratamento explícito: margem de conservadorismo, estimadores enviesados para
cima, ou métodos que sobrevivam à separação. A correção de Firth e os
estimadores de intervalo superior são a resposta técnica a uma exigência que é,
antes de tudo, prudencial.

**Discriminação não é calibração.** Um relatório de validação que reporta apenas
AR/Gini e declara o modelo adequado está incompleto, e a demonstração acima
mostra por quê. Backtesting de nível de PD é obrigação separada — e é o objeto
do capítulo 8.

Para provisionamento sob o arcabouço contábil vigente, a PD deste capítulo é
insumo, não resultado: ela precisa ser condicionada a informação prospectiva e
estendida ao horizonte relevante. É o capítulo 4.

## Exercícios

1. Regenere a base com `gerar_painel_scoring(n_empresas=2000)` e refaça a
   comparação com os coeficientes verdadeiros. A que altura o `EBIT/TA` passa a
   ser recuperado dentro de um erro-padrão? Quantos defaults isso exigiu?

2. Estime o modelo separadamente em cada metade do período. Compare os
   coeficientes. O que a diferença entre as duas metades diz sobre a
   estabilidade que você reportaria em um relatório de monitoramento?

3. Construa uma variável irrelevante (ruído puro, sem relação com o default) e
   inclua-a no modelo. Com que frequência ela sai significativa a 5% se você
   repetir o exercício com 200 sementes diferentes? Compare o resultado usando
   erro-padrão ingênuo e agrupado por empresa.

4. Agrupe as PDs previstas em dez faixas e compare, por faixa, a PD média
   prevista com a taxa de default observada. O modelo discrimina bem — ele
   calibra bem? Guarde a figura: ela é o ponto de partida do capítulo 8."""
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
    destino = pathlib.Path(__file__).resolve().parents[1] / "book" / "cap01_logit.ipynb"
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
