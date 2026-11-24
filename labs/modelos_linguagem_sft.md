# Trabalho prático em grupo — Modelos de Linguagem: SFT (Seções 1–8)

Este lab foi planejado para grupos e para uma aula de **1h50min**. Seu objetivo
é consolidar os conceitos centrais das Seções 1–8 da nota de aula sobre modelos
de linguagem, usando o notebook:

`notebooks/GCC1734_08b_ajuste_supervisionado.ipynb`

O foco prático é o **ajuste supervisionado** (*Supervised Fine-Tuning*, SFT),
mas as respostas devem relacioná-lo às demais etapas: representação por tokens,
modelagem autoregressiva, pré-treinamento e alinhamento por preferências.

Respondam de maneira objetiva e justifiquem com observações do notebook quando
solicitado. Não é necessário obter resultados idênticos aos de outros grupos:
o checkpoint de pré-treinamento e o treinamento podem levar a pequenas
diferenças nas saídas.

---

## Preparação e execução

1. Ativem o ambiente da disciplina e abram o Jupyter dentro da pasta
   `notebooks/`. Os notebooks usam caminhos relativos para os dados e os
   checkpoints:

   ```bash
   conda activate gcc1734
   cd notebooks
   jupyter notebook
   ```

2. Antes de executar este notebook, é necessário executar o Companion 01
   (`notebooks/GCC1734_08a_pre_treinamento.ipynb`) ao menos até a célula que
   salva o modelo e o tokenizador. Isso cria:

   ```text
   notebooks/artefatos_pretraining/modelo/
   notebooks/artefatos_pretraining/tokenizador/
   ```

3. No Jupyter, abram `GCC1734_08b_ajuste_supervisionado.ipynb` e executem suas
   células **na ordem**, sem alterar o código inicialmente. O arquivo de pares
   para SFT é lido de `corpora/sft_pares.csv`.

4. Na célula de treinamento, mantenham inicialmente `MAX_STEPS = 180`. Se o
   computador for lento, usem `MAX_STEPS = 60`; registrem esse valor na entrega.

> Se aparecer erro relacionado a `Trainer`, confirmem que o kernel selecionado é
> **gcc1734**. O ambiente deve conter `torch`, `transformers` e `accelerate`.

### Sugestão de tempo

| Etapa | Tempo sugerido |
|---|---:|
| Preparação e primeira leitura do notebook | 15 min |
| Execução e perguntas dos blocos 1–3 | 40 min |
| Experimento curto e perguntas dos blocos 4–5 | 35 min |
| Síntese e organização da entrega | 20 min |

---

## Bloco 1 — Etapas de treinamento e representação

**1.** A nota distingue pré-treinamento, SFT e alinhamento por preferências.
Complete a tabela com o objetivo principal de cada etapa e o tipo de dado usado
como sinal de treinamento.

| Etapa | Objetivo principal | Sinal/dado de treinamento |
|---|---|---|
| Pré-treinamento |  |  |
| SFT |  |  |
| Alinhamento por preferências |  |  |

**2.** O notebook carrega tanto um modelo quanto um tokenizador produzidos no
Companion 01. Por que não seria adequado treinar um novo tokenizador no SFT?
Mencione explicitamente os IDs dos tokens e a matriz de embeddings.

**3.** Na primeira seção do notebook, observem as colunas `instrucao`,
`resposta`, `tarefa` e `split` de `sft_pares.csv`. Escolham uma linha e
identifiquem:

- a instrução fornecida ao modelo;
- a resposta demonstrada;
- a tarefa a que ela pertence;
- se ela foi usada para treino ou validação.

---

## Bloco 2 — Formato autoregressivo e função de perda

**4.** Escrevam o formato geral de um exemplo de SFT usado no notebook, incluindo
`<bos>`, o texto `Resposta:` e `<eos>`. Na geração, qual parte desse formato é
fornecida ao modelo e qual parte ele deve produzir?

**5.** Na tabela produzida pela função `preparar_exemplo`, o que significa o
valor `-100` na coluna `label`? Por que os tokens da instrução devem estar
mascarados dessa forma na perda de SFT?

**6.** A célula seguinte mostra que a perda calculada pelo modelo coincide com
a entropia cruzada calculada manualmente após um deslocamento de uma posição.
Expliquem qual token é previsto pelo último token do contexto imediatamente
antes da resposta. Não é necessário reproduzir a conta numérica.

---

## Bloco 3 — Dados, lotes e validação

**7.** O notebook separa frases de treino e validação, e verifica que os dois
conjuntos são disjuntos. Por que separar somente pares de instrução-resposta,
mas deixar a mesma frase aparecer nos dois conjuntos, seria uma avaliação menos
confiável?

**8.** Os exemplos que ultrapassam 64 posições são excluídos, em vez de serem
truncados. Qual problema poderia ocorrer se o final da sequência fosse cortado?
Relacionem a resposta à posição dos tokens de resposta e à função de perda.

**9.** Observem o lote criado pelo `SFTCollator`. Para que servem, respectivamente,
`input_ids`, `attention_mask` e `labels`? Por que os tokens de preenchimento
também recebem `-100` em `labels`?

---

## Bloco 4 — Efeito do ajuste supervisionado

**10.** Comparem a tabela de respostas geradas antes e depois do SFT. Escolham
um dos quatro *prompts* e relatem:

- a instrução;
- a resposta antes do SFT;
- a resposta depois do SFT;
- uma evidência de que o comportamento se aproximou (ou não) do formato pedido.

**11.** Observem a curva de perdas. O que uma queda na perda de treino indica?
E o que a perda de validação acrescenta a essa observação? Expliquem por que
uma perda de validação menor, por si só, não demonstra que o modelo se tornou um
assistente geral confiável.

**12.** O notebook calcula acerto exato por tarefa (`tema`, `inicio` e `final`).
Por que acerto exato é útil nesse cenário sintético? Dê um exemplo de situação
realista em que essa métrica seria limitada ou injusta.

---

## Bloco 5 — Experimento curto

**13.** Alterem apenas `MAX_STEPS` para um valor menor ou maior que o usado na
primeira execução (sugestão: `60` ou `240`), executem novamente a partir da
célula de treinamento e preencham a tabela.

| Execução | `MAX_STEPS` | Loss de validação final | Uma observação sobre as respostas geradas |
|---|---:|---:|---|
| Inicial |  |  |  |
| Alterada |  |  |  |

Em duas ou três frases, expliquem se a mudança trouxe evidência de melhoria,
piora ou resultado inconclusivo. Não é necessário repetir experimentos até
encontrar um resultado específico.

**14.** Na seção de inspeção de probabilidades, o notebook compara a perda média
de respostas candidatas para a instrução sobre o tema de uma frase. Qual resposta
deve ter menor perda após um SFT bem-sucedido? Por que essa comparação é feita
condicionada à instrução, e não como uma probabilidade isolada da palavra?

---

## Bloco 6 — Limites do SFT e conexão com alinhamento

**15.** Suponha que, para uma instrução, existam duas respostas plausíveis, mas
uma seja mais clara e segura que a outra. Por que uma coleção de exemplos
`instrução → resposta demonstrada` não expressa diretamente essa preferência?

**16.** Descrevam, em uma frase cada, as duas ideias abaixo:

- como dados de preferência podem complementar o SFT;
- por que alinhar respostas a preferências humanas não garante que elas sejam
  factualmente corretas ou livres de vieses.

---

## Entrega esperada

Cada grupo deve entregar um único arquivo Markdown ou PDF contendo:

- identificação dos integrantes;
- respostas dos itens **1 a 16**;
- a tabela e a breve análise do experimento do item 13;
- o valor de `MAX_STEPS` utilizado na execução inicial e na alterada.

Cada integrante deve submeter a mesma versão produzida pelo grupo.
