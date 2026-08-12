# Curso de Risco de Crédito em Python

Curso em 12 capítulos que cobre o ciclo completo de modelagem de risco de
crédito — do escore individual ao capital regulatório — com implementações em
Python, bases sintéticas reprodutíveis e notebooks explicativos.

A estrutura de capítulos segue o roteiro clássico de Löffler e Posch
(*Credit Risk Modeling using Excel and VBA*). As implementações são próprias,
escritas a partir da formulação matemática dos métodos, e as bases são geradas
por processos sintéticos documentados — nenhum dado proprietário é redistribuído.

## Por que base sintética

Cada capítulo tem um processo gerador de dados com parâmetros verdadeiros
expostos no código. Isso permite três coisas que dado real não permite:

1. comparar o que o estimador **recuperou** com o que foi **plantado**;
2. escrever teste automatizado sobre o resultado do modelo, não só sobre o código;
3. variar o processo gerador e observar o estimador quebrar — que é, no fim,
   o objeto de estudo da validação de modelos.

## Instalação

```bash
conda create -n credrisk python=3.11 -y
conda activate credrisk
pip install -e ".[dev,book]"
pytest
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `credrisk/` | biblioteca: toda a matemática, testada |
| `credrisk/data/` | geradores sintéticos e registro de bases |
| `tests/` | suíte pytest, com fechamento analítico onde existe |
| `book/` | notebooks e texto do curso (Jupyter Book) |
| `data/gerado/` | cache das bases, reconstruível, fora do versionamento |

Os notebooks não definem funções: importam de `credrisk` e cuidam da narrativa,
dos gráficos e da interpretação.

## Roteiro

| Cap. | Tema | Módulo |
|---|---|---|
| 1 | Escore de crédito com logit | `scoring` |
| 2 | Abordagem estrutural (Merton) | `structural` |
| 3 | Matrizes de transição | `transition` |
| 4 | Previsão de taxas de default e transição | `forecast` |
| 5 | Previsão de LGD | `lgd` |
| 6 | Estimação de correlação de ativos | `correlation` |
| 7 | Risco de carteira pela abordagem de valor do ativo | `portfolio` |
| 8 | Validação de sistemas de rating | `validation` |
| 9 | Validação de modelos de carteira | `validation` |
| 10 | CDS e probabilidades de default neutras ao risco | `pricing` |
| 11 | Crédito estruturado: CDOs | `structured` |
| 12 | Basileia II e ratings internos | `regcap` |

Cada capítulo encerra com uma nota de ponte regulatória para o arcabouço
brasileiro (CMN 4.966, CMN 4.557, Circular 3.648, BCB 265).

## Licença

Código sob MIT. Textos e notebooks sob CC BY 4.0.
