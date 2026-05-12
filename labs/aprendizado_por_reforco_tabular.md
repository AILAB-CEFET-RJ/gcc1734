# Trabalho prático em grupo — Aprendizado por Reforço (Seções 1–6)

Este trabalho foi planejado para ser realizado em grupo durante os dois tempos
de aula de hoje. O foco é aplicar, manualmente e com justificativas claras, os
conceitos das notas de aula **apenas até a Seção 6**:

- formulação de problemas como PDM
- política, utilidade e fator de desconto
- diferença temporal
- Q-learning e SARSA tabulares
- exploração versus aproveitamento

Respondam de forma objetiva, mas mostrando o raciocínio. Quando houver contas,
não apresentem apenas o resultado final.

---

## Cenário base

Considere o seguinte **Mundo Grade 3 x 3**:

- estado inicial: `(1,1)`
- estado terminal positivo: `(3,3)`, com recompensa `+1`
- estado terminal negativo: `(3,2)`, com recompensa `-1`
- estados não terminais: living reward `-0,04`
- ações disponíveis: `Norte`, `Sul`, `Leste`, `Oeste`
- para as questões numéricas, assuma inicialmente **transições determinísticas**
- use `gamma = 0,9` e `alpha = 0,5`, salvo indicação em contrário
- inicialmente, todos os valores `Q(s,a)` valem `0`

Convenção: `(1,1)` é a célula inferior esquerda; aumentar a segunda coordenada
significa subir na grade.

---

## Bloco 1 — Modelagem do problema

**1.** Escreva os componentes do PDM correspondente ao cenário base:

- conjunto de estados `S`
- conjunto de ações `A`
- regra de transição `T`
- função de recompensa `R`
- fator de desconto `gamma`

Não é necessário listar todas as transições uma a uma, mas a descrição deve ser
precisa.

**2.** Explique por que esse problema pode ser tratado como um processo com
propriedade de Markov. Em seguida, dê um exemplo de informação histórica que
**não** precisaria ser armazenada no estado para o agente agir.

**3.** Compare este problema com um problema clássico de busca monoagente.
Indique:

- o que a formulação PDM representa que a busca clássica não representa bem
- por que, no aprendizado por reforço, o agente não pode simplesmente planejar
  a política ótima antes de interagir com o ambiente

---

## Bloco 2 — Retorno, política e valores

Considere a trajetória:

`(1,1) --Norte--> (1,2) --Leste--> (2,2) --Norte--> (2,3) --Leste--> (3,3)`

As três primeiras transições têm recompensa `-0,04`, e a última leva ao estado
terminal com recompensa `+1`.

**4.** Calcule o retorno descontado dessa trajetória a partir do estado inicial.

**5.** Suponha que, em um certo momento do treinamento, o agente tenha as
seguintes estimativas no estado `(2,3)`:

- `Q((2,3), Norte) = 0,10`
- `Q((2,3), Sul) = -0,20`
- `Q((2,3), Leste) = 0,80`
- `Q((2,3), Oeste) = 0,05`

Qual seria a ação escolhida por uma política gulosa nesse estado? Qual seria o
valor estimado `V((2,3))` sob essa política?

**6.** Explique, sem fazer novas contas, como o valor de uma mesma trajetória
mudaria se:

- `gamma = 0`
- `gamma` fosse muito próximo de `1`

Relacionem a resposta com a importância relativa de recompensas imediatas e
futuras.

---

## Bloco 3 — Diferença temporal e Q-learning

**7.** Um agente observa a transição

`((2,3), Leste, +1, terminal)`

Calcule a atualização de `Q((2,3), Leste)` por Q-learning. Mostrem:

- o alvo
- o erro de TD
- o novo valor de `Q((2,3), Leste)`

**8.** Agora considere a transição

`((2,2), Norte, -0,04, (2,3))`

Use o valor atualizado no item anterior e assuma que, no estado `(2,3)`, a
melhor ação continua sendo `Leste`. Calcule a atualização de
`Q((2,2), Norte)` por Q-learning.

**9.** Com base nos itens 7 e 8, expliquem a ideia de **propagação de valor**.
Por que estados mais próximos do terminal positivo tendem a receber boas
estimativas antes dos estados mais distantes?

---

## Bloco 4 — Q-learning versus SARSA

Considere agora a transição:

`((2,2), Leste, -1, (3,2))`

Suponha que:

- `Q((2,2), Leste) = 0,20`
- no próximo estado, `max_a Q((3,2), a) = 0`
- em uma execução SARSA, a ação sorteada em `(3,2)` também teria valor `0`

**10.** Calcule a atualização de `Q((2,2), Leste)` por Q-learning.

**11.** Calcule a atualização correspondente por SARSA. Neste caso, os
resultados coincidem ou diferem? Expliquem por quê.

**12.** Agora alterem apenas a situação a seguir: em um estado não terminal
`s'`, a melhor ação tem valor `0,60`, mas a política exploratória escolhe uma
ação com valor `0,10`. Sem refazer todas as contas, expliquem qual algoritmo
tenderá a produzir a estimativa mais otimista para a ação anterior e por quê.

---

## Bloco 5 — Exploração versus aproveitamento

**13.** Suponha que, no estado `(1,1)`, os valores atuais sejam:

- `Q((1,1), Norte) = 0,30`
- `Q((1,1), Leste) = 0,28`
- `Q((1,1), Sul) = -0,10`
- `Q((1,1), Oeste) = -0,05`

Descrevam o que significa, nesse contexto:

- agir por **aproveitamento**
- agir por **exploração**

Em seguida, expliquem o efeito de usar uma política `epsilon-greedy` com
`epsilon = 0,1`.

**14.** Um grupo afirma: "Se uma ação parece a melhor agora, o agente deveria
sempre escolhê-la". Construam um contra-argumento baseado no dilema
exploração-aproveitamento. A resposta deve mencionar explicitamente o risco de
convergir para uma política ruim por falta de experiência.

**15.** Fechamento. Considerando o cenário base, respondam:

- Q-learning ou SARSA parece mais prudente perto de regiões perigosas?
- uma política muito exploratória seria desejável o tempo todo?
- qual seria uma escolha razoável de estratégia de exploração no início e no
  fim do treinamento?

Justifiquem conceitualmente; não é preciso propor fórmulas novas.

---

## Entrega esperada

Ao final do tempo, cada grupo deve entregar:

- as respostas dos itens `1` a `15`
- as contas dos itens numéricos
- uma justificativa curta, mas clara, para as perguntas conceituais

Se sobrar tempo, comparem as respostas entre grupos e identifiquem em quais
itens houve maior divergência de interpretação.
