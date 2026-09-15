# Perguntas de discussão — `busca_adversarial_minimax.py`

Formem grupos de 2 a 4 componentes. Cada grupo deve elaborar respostas justificadas, referenciando partes do código sempre que possível.

O arquivo acompanha uma implementação jogável de um **Pacman simplificado**,
com alternância de turnos entre Pacman (MAX) e fantasma (MIN), além de suporte
para **Minimax**, **poda alfa-beta**, **H-Minimax** e uma versão com **busca
quiescente**.

## Como executar o script

O laboratório requer Python 3. A partir da raiz do repositório, entre na pasta
do laboratório e consulte primeiro as opções disponíveis:

```bash
cd labs
python3 busca_adversarial_minimax.py --help
```

Para iniciar a execução padrão, use:

```bash
python3 busca_adversarial_minimax.py
```

Por padrão, o jogo usa o grid `lab`, profundidade 4 e busca quiescente. Pacman
e fantasma são controlados por IA; portanto, a partida avança automaticamente.
Em cada turno, observem o agente que jogou, a ação escolhida, o valor estimado
e o número de nós examinados pela busca.

Também é possível executar o script sem mudar de diretório, usando
`python3 labs/busca_adversarial_minimax.py ...` a partir da raiz do repositório.
Os parâmetros mais importantes são `--grid`, `--algorithm`, `--depth`,
`--pacman-mode` e `--ghost-mode`. Os comandos ao final deste roteiro apresentam
configurações adequadas para comparar os algoritmos e os modos de controle.

---

## Bloco 1 — Representação do jogo

**1.** Quais elementos do problema adversarial aparecem explicitamente na
classe `PacmanAdversarialProblem`? Identifique no código os equivalentes de:

- estado (`State`)
- ações (`ACTIONS`)
- transição (`RESULT`)
- jogador da vez (`PLAYER`)
- teste terminal (`TERMINAL-TEST`)
- utilidade (`UTILITY`)

**2.** A classe `GameState` armazena `pacman`, `ghost`, `foods`, `turn`,
`score` e `ply`. Qual é o papel de cada um desses atributos? Por que guardar
`turn` explicitamente simplifica o código?

---

## Bloco 2 — Regras do jogo e topologia do mapa

**3.** O método `legal_actions()` consulta a posição do jogador ativo. Por que
isso é suficiente para definir as ações legais neste domínio?

**4.** A função `neighbors()` é usada em vários pontos do código. Onde ela é
usada apenas para movimentação imediata e onde ela é usada para raciocínio mais
estrutural sobre o mapa?

---

## Bloco 3 — Terminal, utilidade e avaliação

**5.** O método `terminal_test()` considera três condições: colisão, vitória e
limite de plies. Explique o papel de cada uma delas.

**6.** Quais componentes contribuem para `eval()`? Explique a intuição de cada
um deles:

- score acumulado
- distância à comida mais próxima
- distância ao fantasma
- mobilidade
- penalidade por beco/corredor sob pressão

---

## Bloco 4 — Minimax

**7.** Explique o papel das funções internas de `minimax_decision()`:

- `value`
- `max_value`
- `min_value`

Como elas implementam diretamente a definição recursiva do Minimax?

**8.** A função `minimax_decision()` retorna uma `SearchResult` com três
campos. O que significa cada um deles?

---

## Bloco 5 — Poda alfa-beta

**9.** Em `alphabeta_decision()`, o que representam `alpha` e `beta`? Explique
sem usar fórmulas, apenas em termos de “melhor valor já garantido”.

**10.** Onde exatamente ocorre a poda em `max_value()` e `min_value()`? Explique
por que a decisão de interromper a expansão é segura.

---

## Bloco 6 — H-Minimax e profundidade limitada

**11.** O que muda quando passamos de `minimax` para `hminimax`?

**12.** O método `value()` em `alphabeta_decision()` usa `eval()` quando o
limite de profundidade é atingido. Por que isso é necessário?


---

## Bloco 7 — Busca quiescente

**13.** O método `non_quiescent()` tenta detectar estados taticamente instáveis.
Quais sinais locais ele usa?

**14.** Por que `non_quiescent()` não tenta provar formalmente que existe uma
sequência forçada de derrota?

---

## Bloco 8 — Interação e modos de execução

**15.** O comando abaixo roda Pacman humano contra fantasma controlado por IA:

```bash
python3 busca_adversarial_minimax.py --pacman-mode human --ghost-mode ai --algorithm quiescent
```

Explique passo a passo o que ocorrerá em cada turno.

---

## Comandos úteis

```bash
python3 busca_adversarial_minimax.py
python3 busca_adversarial_minimax.py --grid beco --algorithm hminimax --depth 3
python3 busca_adversarial_minimax.py --grid beco --algorithm quiescent --depth 2
python3 busca_adversarial_minimax.py --pacman-mode human --ghost-mode ai --algorithm quiescent
python3 busca_adversarial_minimax.py --pacman-mode ai --ghost-mode random --algorithm hminimax
```
