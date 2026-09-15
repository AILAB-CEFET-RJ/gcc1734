# Perguntas de discussão — `busca_adversarial_minimax.py`

Formem grupos de 2 a 4 componentes. Cada grupo deve elaborar respostas justificadas, referenciando partes do código sempre que possível.

Este roteiro cobre o conteúdo da Nota 04 desde a formulação de jogos adversariais até **Funções de Avaliação Heurística**: árvore de jogo, Minimax, poda alfa-beta, H-Minimax e avaliação de estados. Não abordem, neste laboratório, as opções ou funções relacionadas a busca quiescente e efeito de horizonte.

O arquivo acompanha uma implementação jogável de um **Pacman simplificado**, com alternância de turnos entre Pacman (MAX) e fantasma (MIN). O script também contém uma implementação de busca quiescente, mas ela está fora do escopo deste roteiro.

## Como executar o script

O laboratório requer Python 3. A partir da raiz do repositório, entre na pasta do laboratório e consulte primeiro as opções disponíveis:

```bash
cd labs
python3 busca_adversarial_minimax.py --help
```

O comando sem opções usa busca quiescente por padrão. Como esse conteúdo ainda não foi abordado, use explicitamente um dos algoritmos abaixo. Para começar com uma árvore pequena, execute Minimax no grid `mini` e limite a partida a poucos plies:

```bash
python3 busca_adversarial_minimax.py --grid mini --algorithm minimax --max-ply 6
```

Em cada turno, observem o agente que jogou, a ação escolhida, o valor retornado pela busca e o número de nós examinados. Também é possível executar o script sem mudar de diretório, usando `python3 labs/busca_adversarial_minimax.py ...` a partir da raiz do repositório.

A interface também mostra a linha `Não quiescente?`. Ignore esse diagnóstico neste laboratório: ele pertence à implementação de busca quiescente, que será estudada em outro momento.

Os parâmetros mais relevantes neste roteiro são `--grid`, `--algorithm`, `--depth`, `--max-ply`, `--pacman-mode` e `--ghost-mode`. Use `minimax`, `alphabeta` ou `hminimax` em `--algorithm`; não use `quiescent` neste laboratório.

---

## Bloco 1 — Formulação e representação do jogo

**1.** Quais elementos do problema adversarial aparecem explicitamente na classe `PacmanAdversarialProblem`? Identifique no código os equivalentes de:

- estado (`State`)
- ações (`ACTIONS`)
- transição (`RESULT`)
- jogador da vez (`PLAYER`)
- teste terminal (`TERMINAL-TEST`)
- utilidade (`UTILITY`)

**2.** A classe `GameState` armazena `pacman`, `ghost`, `foods`, `turn`, `score` e `ply`. Qual é o papel de cada atributo? Por que `turn` é necessário para representar uma árvore de jogo, e não apenas um caminho de busca monoagente?

**3.** Explique como `legal_actions()` e `result()` implementam, respectivamente, as funções `ACTIONS` e `RESULT`. Quais componentes do estado podem mudar após uma ação de Pacman? Quais podem mudar após uma ação do fantasma?

---

## Bloco 2 — Estados terminais e utilidade

**4.** O método `terminal_test()` considera colisão, vitória e limite de plies. Explique o papel de cada condição e por que o limite de plies é necessário neste domínio.

**5.** Como `utility()` atribui valores aos estados terminais? Explique por que os valores devem ser interpretados da perspectiva de MAX, mesmo quando o estado terminal é alcançado após uma ação de MIN.

---

## Bloco 3 — Minimax

**6.** Explique o papel das funções internas de `minimax_decision()`:

- `value`
- `max_value`
- `min_value`

Como elas implementam diretamente a definição recursiva do Minimax?

**7.** Em `max_value()` e `min_value()`, por que os acumuladores começam, respectivamente, em `-math.inf` e `math.inf`? O que poderia dar errado se começassem em zero?

**8.** A classe `SearchResult` armazena `action`, `value` e `nodes`. Qual é o significado de cada campo? Por que o contador de nós é útil ao comparar Minimax com poda alfa-beta?

**9.** Execute o comando abaixo e acompanhe pelo menos um turno de Pacman e um turno do fantasma no código:

```bash
python3 busca_adversarial_minimax.py --grid mini --algorithm minimax --max-ply 6
```

Identifique a sequência de chamadas desde `choose_action_ai()` até `minimax_decision()`. Em que ponto a busca interrompe a descida recursiva e começa a retornar valores?

---

## Bloco 4 — Poda alfa-beta

**10.** Em `alphabeta_decision()`, o que representam `alpha` e `beta`? Explique em termos de limites já conhecidos por MAX e MIN, sem recorrer apenas às fórmulas.

**11.** Onde ocorrem os cortes em `max_value()` e `min_value()`? Relacione as condições `v >= beta` e `v <= alpha` ao fato de que um ancestral já possui uma alternativa que não será superada.

**12.** A função `ordered_actions()` usa `eval()` para ordenar os filhos antes de expandi-los. Por que essa ordenação pode aumentar o número de podas sem modificar o valor minimax nem a ação ótima retornada pelo algoritmo?

**13.** Compare as execuções abaixo no mesmo grid e com o mesmo limite de plies:

```bash
python3 busca_adversarial_minimax.py --grid mini --algorithm minimax --max-ply 6
python3 busca_adversarial_minimax.py --grid mini --algorithm alphabeta --max-ply 6
```

As ações e os valores retornados devem coincidir? Compare os números de nós examinados e explique qualquer diferença observada.

---

## Bloco 5 — H-Minimax e funções de avaliação

**14.** O que muda quando `alphabeta_decision()` é chamado com `depth_limit` definido, como ocorre no modo `hminimax`? Descreva a condição de corte e o papel de `eval()` nas folhas de corte que não são estados terminais.

**15.** Quais componentes contribuem para `eval()`? Explique a intuição de cada um deles:

- score acumulado
- distância à comida mais próxima
- distância ao fantasma
- mobilidade
- penalidade por beco ou corredor sob pressão

**16.** Execute os comandos abaixo e compare o que é informado a cada turno:

```bash
python3 busca_adversarial_minimax.py --grid beco --algorithm hminimax --depth 3
python3 busca_adversarial_minimax.py --grid duas_comidas --algorithm hminimax --depth 3
```

Escolha um estado exibido em uma das execuções e justifique, usando os termos de `eval()`, por que ele parece favorável ou desfavorável para Pacman. Explique também por que esse valor é uma estimativa, e não uma utilidade terminal garantida.

---

## Comandos úteis

```bash
# Minimax completo: use somente no grid pequeno.
python3 busca_adversarial_minimax.py --grid mini --algorithm minimax --max-ply 6

# Alfa-beta completo no mesmo cenário, para comparar nós examinados.
python3 busca_adversarial_minimax.py --grid mini --algorithm alphabeta --max-ply 6

# H-Minimax: profundidade limitada e avaliação heurística.
python3 busca_adversarial_minimax.py --grid beco --algorithm hminimax --depth 3
python3 busca_adversarial_minimax.py --grid duas_comidas --algorithm hminimax --depth 3

# Pacman humano contra fantasma controlado por H-Minimax.
python3 busca_adversarial_minimax.py --grid mini --pacman-mode human --ghost-mode ai --algorithm hminimax --depth 3
```
