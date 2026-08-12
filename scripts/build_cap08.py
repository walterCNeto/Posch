"""Monta o notebook do capítulo 8 (validação de sistemas de rating)."""

from __future__ import annotations

import pathlib

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELULAS = [
    md(
        """# Capítulo 8 — Validação de sistemas de rating

## O problema

Um sistema de rating está pronto e em produção. Ele classifica devedores em
grades, e cada grade carrega uma PD. A pergunta da validação independente é
aparentemente simples: **está funcionando?**

A pergunta se desdobra em duas, que são independentes uma da outra e exigem
instrumentos diferentes:

**Discriminação** — o sistema ordena bem? Os devedores que quebraram tinham,
antes de quebrar, notas piores que os que não quebraram?

**Calibração** — os níveis estão certos? A grade que promete 1,1% de default
entrega 1,1%?

Que sejam independentes não é detalhe técnico. O capítulo 2 já mostrou um modelo
com discriminação quase perfeita e PDs erradas pela metade. O inverso também
existe: um sistema que acerta a média da carteira e não distingue nada dentro
dela. **Aprovar um sistema medindo só uma das duas é o erro mais comum da
validação de crédito.**"""
    ),
    code(
        """import matplotlib.pyplot as plt
import pandas as pd

from credrisk.data.generators import gerar_carteira_rating
from credrisk.validation.rating import (
    auc,
    brier,
    curva_cap,
    curva_roc,
    erro_padrao_auc,
    hosmer_lemeshow,
    razao_de_acuracia,
    tabela_por_grade,
    teste_binomial,
    teste_binomial_com_correlacao,
    teste_binomial_por_grade,
)
from credrisk.viz import estilo

estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.5f}")"""
    ),
    md(
        """## Os dados

Uma carteira de oito mil devedores classificados em sete grades, com os defaults
do ano observados. Por enquanto **sem correlação** — defaults independentes dado
o rating. É a hipótese que os testes clássicos assumem, e vamos honrá-la antes
de derrubá-la."""
    ),
    code(
        """carteira = gerar_carteira_rating(rho=0.0, n_obrigados=8000, semente=1)
tabela = tabela_por_grade(carteira)
tabela[["Grade", "n", "defaults", "pd_prevista", "taxa_observada"]].round(5)"""
    ),
    md(
        """## Discriminação

A curva CAP ordena a carteira do pior escore para o melhor e pergunta: quantos
dos defaults já foram capturados ao examinar cada fração da carteira? A curva
ROC faz a mesma pergunta em outro par de eixos.

A razão de acurácia é a área entre a curva CAP e a diagonal, normalizada pela
área do sistema perfeito. Vale a identidade `AR = 2 × AUC − 1`, o que significa
que **as duas métricas carregam exatamente a mesma informação** — reportar as
duas não acrescenta nada além de simetria visual no relatório."""
    ),
    code(
        """cap = curva_cap(carteira["escore"], carteira["Default"])
roc = curva_roc(carteira["escore"], carteira["Default"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

ax1.plot(cap["fracao_carteira"], cap["fracao_defaults"], lw=2, label="sistema")
ax1.plot([0, 1], [0, 1], ls=":", color="#7a8b8b", label="aleatório")
taxa = carteira["Default"].mean()
ax1.plot([0, taxa, 1], [0, 1, 1], ls="--", color="#d9a441", label="perfeito")
ax1.set_title("Curva CAP")
ax1.set_xlabel("fração da carteira (pior para melhor)")
ax1.set_ylabel("fração dos defaults capturada")
ax1.legend()

ax2.plot(roc["falsos_positivos"], roc["verdadeiros_positivos"], lw=2)
ax2.plot([0, 1], [0, 1], ls=":", color="#7a8b8b")
ax2.set_title("Curva ROC")
ax2.set_xlabel("falsos positivos")
ax2.set_ylabel("verdadeiros positivos")
plt.show()

a = auc(carteira["escore"], carteira["Default"])
ep = erro_padrao_auc(carteira["escore"], carteira["Default"])
print(f"AUC = {a:.4f}  (erro-padrão {ep:.4f})")
print(f"AR  = {razao_de_acuracia(carteira['escore'], carteira['Default']):.4f}")"""
    ),
    md(
        """Uma AR nessa faixa seria considerada boa em qualquer comitê de modelos.

Agora o teste que separa discriminação de calibração de forma inequívoca: um
sistema com todas as PDs multiplicadas por um quarto — grosseiramente
descalibrado — tem exatamente a mesma discriminação."""
    ),
    code(
        """descalibrado = gerar_carteira_rating(
    rho=0.0, vies_pd=0.25, n_obrigados=8000, semente=1
)

pd.DataFrame({
    "sistema calibrado": {
        "AUC": auc(carteira["escore"], carteira["Default"]),
        "AR": razao_de_acuracia(carteira["escore"], carteira["Default"]),
        "Brier": brier(carteira["PD_atribuida"], carteira["Default"]),
        "PD média prometida": carteira["PD_atribuida"].mean(),
        "taxa observada": carteira["Default"].mean(),
    },
    "PDs divididas por 4": {
        "AUC": auc(descalibrado["escore"], descalibrado["Default"]),
        "AR": razao_de_acuracia(descalibrado["escore"], descalibrado["Default"]),
        "Brier": brier(descalibrado["PD_atribuida"], descalibrado["Default"]),
        "PD média prometida": descalibrado["PD_atribuida"].mean(),
        "taxa observada": descalibrado["Default"].mean(),
    },
})"""
    ),
    md(
        """AUC e AR **idênticos**, até a última casa. E não é coincidência: as duas
métricas dependem apenas da ordenação dos escores, e multiplicar todos por uma
constante não altera ordenação alguma.

O sistema descalibrado provisiona um quarto do que deveria. Sua validação por
poder discriminante o aprova sem ressalvas.

O escore de Brier capta a diferença, porque penaliza o erro quadrático da
probabilidade — mas é uma medida agregada difícil de interpretar isoladamente, e
mistura calibração com discriminação. Para calibração é preciso teste próprio.

## Calibração grade a grade

O instrumento clássico é o teste binomial: sob a hipótese de que a PD prometida
está correta e de que os defaults são independentes, o número de defaults segue
uma binomial, e rejeita-se quando o observado passa do quantil superior."""
    ),
    code(
        """teste_binomial_por_grade(carteira).round(5)"""
    ),
    md(
        """Nesta carteira, uma grade aparece rejeitada — apesar de o modelo estar
**correto por construção**. É a grade de melhor qualidade, com PD de 0,04% e
poucas centenas de devedores: bastam dois defaults por azar para o teste
disparar.

Isso ilustra duas armadilhas de uma vez. A primeira é que grades de alta
qualidade são praticamente não-testáveis: com PD de 0,04% e quatrocentos
devedores, o número esperado de defaults é 0,16, e qualquer resultado diferente
de zero parece anômalo. A segunda é **múltiplas comparações**: sete testes a 5%
cada produzem probabilidade bem maior que 5% de ao menos uma rejeição espúria.

Um relatório que aponta "a grade AAA falhou no teste binomial" sem discutir esses
dois pontos está reportando ruído como se fosse achado.

## O problema sério: correlação

Até aqui os defaults foram simulados independentes. O capítulo 6 mostrou que não
são: existe um fator sistêmico que os empurra juntos, com correlação de ativos
da ordem de 0,12.

O teste binomial não sabe disso. E a consequência é grave."""
    ),
    code(
        """com_correlacao = gerar_carteira_rating(rho=0.12, n_obrigados=8000, semente=1)

linha = tabela_por_grade(com_correlacao)
linha = linha[linha["Grade"] == "BB"].iloc[0]

sem = teste_binomial(int(linha["n"]), int(linha["defaults"]),
                     float(linha["pd_prevista"]))
com = teste_binomial_com_correlacao(int(linha["n"]), int(linha["defaults"]),
                                    float(linha["pd_prevista"]), rho=0.12)

pd.DataFrame({
    "binomial (supõe independência)": sem,
    "integrado sobre o fator (ρ = 0,12)": com,
})"""
    ),
    md(
        """O limite crítico praticamente dobra quando a correlação é reconhecida.

A razão é a mesma do capítulo 7: com um fator comum, a distribuição do número de
defaults tem cauda muito mais gorda que a binomial. Anos ruins produzem muitos
defaults simultâneos, e isso é comportamento **esperado** do modelo, não
evidência contra ele.

Vale medir o estrago de forma sistemática. O experimento: gerar muitas carteiras
com o modelo **correto** e contar quantas vezes cada teste rejeita. Toda
rejeição é, por construção, um falso positivo."""
    ),
    code(
        """def taxa_de_falsos_positivos(rho: float, n_carteiras: int = 120) -> dict:
    binomial, integrado, hl = 0, 0, 0
    for s in range(n_carteiras):
        d = gerar_carteira_rating(rho=rho, n_obrigados=4000, semente=5000 + s)
        t = tabela_por_grade(d)
        linha = t[t["Grade"] == "BB"].iloc[0]
        binomial += teste_binomial(
            int(linha["n"]), int(linha["defaults"]), float(linha["pd_prevista"])
        )["rejeita"]
        integrado += teste_binomial_com_correlacao(
            int(linha["n"]), int(linha["defaults"]), float(linha["pd_prevista"]),
            rho=max(rho, 1e-9),
        )["rejeita"]
        hl += hosmer_lemeshow(d["PD_atribuida"], d["Default"]).rejeita
    return {
        "binomial (grade BB)": binomial / n_carteiras,
        "binomial com ρ (grade BB)": integrado / n_carteiras,
        "Hosmer-Lemeshow": hl / n_carteiras,
    }


resultado = pd.DataFrame({
    "defaults independentes (ρ = 0)": taxa_de_falsos_positivos(0.0),
    "com correlação (ρ = 0,12)": taxa_de_falsos_positivos(0.12),
})
print("taxa de rejeição com o modelo CORRETO — nível nominal = 5%\\n")
resultado.round(3)"""
    ),
    md(
        """Este é o resultado central do capítulo, e ele deveria mudar como muito
relatório de validação é lido.

Sem correlação, os testes se comportam: rejeitam perto dos 5% nominais. Com
correlação de 0,12 — um valor conservador diante do intervalo estimado no
capítulo 6 — o teste binomial rejeita várias vezes mais que o nominal, e o
Hosmer-Lemeshow rejeita um modelo **perfeitamente calibrado** na larga maioria
das carteiras.

A versão que integra sobre o fator sistêmico restaura o nível nominal.

Traduzindo para a prática: se a sua área de validação roda teste binomial anual
por grade e reporta as grades que "falharam", ela está produzindo uma lista de
falsos positivos com regularidade — e o custo disso não é acadêmico. Modelos são
recalibrados sem necessidade, o comitê perde confiança nos testes, e quando um
modelo de fato quebra o alarme já virou ruído de fundo.

## Um ano ruim não é um modelo ruim

O ponto merece ser dito de outro jeito, porque é contraintuitivo. Em um ano de
recessão, **todas** as grades vão exceder suas PDs simultaneamente. Isso não é
evidência de que o sistema está descalibrado: é exatamente o que um sistema
correto faz num ano ruim, porque a PD é uma média de longo prazo e o ano ruim é
uma realização da cauda.

O erro simétrico também existe: numa sequência de anos bons, todas as grades
ficam abaixo da PD, e alguém conclui que o sistema é conservador demais e propõe
afrouxar. É assim que se desmonta um sistema de rating em plena expansão de
crédito.

**Calibração de PD é afirmação sobre a média de um ciclo, e só pode ser testada
ao longo de um ciclo.**

## O que quebra fora do laboratório

**A janela de observação é curta.** Testar PD média de ciclo exige um ciclo.
Poucas carteiras têm dez anos de histórico consistente com definição de default
estável.

**A definição de default muda.** Alterações regulatórias e de sistema mudam o que
conta como default no meio da série, e isso aparece nos testes como
descalibragem.

**A carteira muda.** O sistema de rating é aplicado a uma população que muda de
composição; parte do desvio observado é mudança de mix, não erro do modelo.

**Filtragem por aprovação.** Só se observa o default de quem foi aprovado. As
grades ruins da carteira são as sobreviventes de um processo de seleção, e sua
taxa observada não é a da população que recebeu aquela nota.

## Ponte regulatória

**Discriminação e calibração são requisitos separados.** O arcabouço de validação
espera evidência das duas. Um relatório com AR e sem teste de nível de PD está
incompleto, e este capítulo mostra por quê: as duas propriedades são
independentes e o AR é cego para a segunda.

**Correlação nos testes de calibração.** Testes que supõem independência entre
devedores produzem falsos positivos em série. Reconhecer a correlação — como
faz a versão integrada aqui — é ajuste técnico simples com efeito grande. Vale
a pena documentar essa escolha, porque a diferença entre os dois limites
críticos é da ordem de duas vezes.

**Estabilidade ao longo do tempo.** Monitorar AR ano a ano é prática comum e
útil, mas AR também flutua com a composição da carteira. Queda de AR não é
necessariamente degradação do modelo.

**Carteiras de baixo default.** Grades superiores são estatisticamente
intestáveis em qualquer janela realista. É a mesma limitação do capítulo 1, agora
do lado da validação, e a resposta continua sendo margem de conservadorismo em
vez de teste.

## Exercícios

1. Repita o experimento de falsos positivos variando ρ de 0 a 0,25. A partir de
   que correlação o teste binomial deixa de ser utilizável?

2. Simule dez anos de carteira com o modelo correto e aplique o teste binomial
   ano a ano na grade BB. Em quantos anos ele rejeita? Agora agregue os dez anos
   num único teste. O que muda, e por quê?

3. Construa um sistema com AR alto e PDs multiplicadas por três. Qual teste
   detecta? Qual não detecta? Escreva o parágrafo de conclusão que você mandaria
   ao comitê.

4. Compare AR calculado na carteira inteira com AR calculado apenas nas grades
   BBB a B. Por que o segundo é menor? O que isso diz sobre usar AR agregado
   para comparar dois modelos aplicados a carteiras diferentes?"""
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
        pathlib.Path(__file__).resolve().parents[1] / "book" / "cap08_validacao_rating.ipynb"
    )
    nbf.write(nb, destino)
    print(f"escrito: {destino} ({len(CELULAS)} células)")


if __name__ == "__main__":
    main()
