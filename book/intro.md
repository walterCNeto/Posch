# Apresentação

Este curso percorre o ciclo completo de modelagem de risco de crédito: começa na
probabilidade de default de um devedor individual, passa pela perda dada o
default, agrega tudo em uma distribuição de perda de carteira, submete cada peça
a validação estatística e termina no capital regulatório que sai disso.

## Como o curso está organizado

Cada capítulo é um notebook autocontido com quatro camadas:

**O problema.** Qual decisão de negócio ou exigência regulatória motiva o
modelo. Sem isso, o método vira exercício de álgebra.

**A formulação.** A matemática do estimador, escrita por extenso — não como
citação, mas na forma em que ela é efetivamente implementada.

**A implementação.** Código que chama a biblioteca `credrisk`. Os notebooks não
definem funções; a matemática mora no pacote, onde pode ser testada.

**A crítica.** O que quebra o modelo, o que o dado real faz que o dado simulado
não faz, e o que um validador independente perguntaria. Cada capítulo termina
com uma nota de ponte para o arcabouço regulatório brasileiro.

## A opção pela base sintética

Nenhum capítulo usa dado proprietário. Todas as bases vêm de processos
geradores documentados em `credrisk.data.generators`, com semente fixa e
parâmetros verdadeiros expostos no código.

Isso não é uma limitação contornada, é uma escolha pedagógica. Com dado real
você nunca sabe se o modelo errou porque o estimador é ruim, porque a amostra é
pequena ou porque o mundo mudou. Com dado sintético você sabe exatamente qual
era a resposta certa, e pode perguntar coisas que dado real não responde: o que
acontece com o estimador de correlação quando a amostra cai pela metade? Quanto
viés a censura por saída da carteira introduz? Quando o intervalo de confiança
deixa de cobrir o parâmetro?

Essa é, no fundo, a mentalidade de validação de modelos — e é por isso que ela
aparece desde o primeiro capítulo, não só nos capítulos 8 e 9.

## Pré-requisitos

Estatística inferencial, regressão e Python intermediário (`numpy`, `pandas`).
Não é necessário conhecimento prévio de risco de crédito nem de Basileia.

## Ambiente

```bash
conda create -n credrisk python=3.11 -y
conda activate credrisk
pip install -e ".[dev,book]"
pytest
```
