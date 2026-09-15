from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Pos = Tuple[int, int]


# ============================================================================
# Estado do jogo
# ============================================================================

@dataclass(frozen=True)
class GameState:
    """
    Representa um instantâneo completo do jogo em um dado momento.

    Atributos
    ---------
    pacman : Pos
        Posição atual do Pacman no grid, como tupla (linha, coluna).
    ghost : Pos
        Posição atual do fantasma no grid, como tupla (linha, coluna).
    foods : frozenset[Pos]
        Conjunto das células que ainda contêm comida.
        Usar frozenset garante imutabilidade e permite usar GameState como
        chave em dicionários ou conjuntos (útil para detecção de ciclos).
    turn : str
        Indica de quem é a vez: "MAX" para Pacman, "MIN" para o fantasma.
        Segue a convenção do Minimax: MAX quer maximizar a utilidade,
        MIN quer minimizá-la.
    score : int
        Pontuação acumulada — cada comida coletada vale 10 pontos.
    ply : int
        Número de jogadas já realizadas desde o estado inicial.
        Um "ply" corresponde ao turno de *um* jogador (≠ "round", que
        engloba os dois jogadores).
    """
    pacman: Pos
    ghost: Pos
    foods: frozenset[Pos]
    turn: str          # "MAX" (Pacman) ou "MIN" (Fantasma)
    score: int = 0     # comida coletada até aqui
    ply: int = 0       # número de jogadas já realizadas


# ============================================================================
# Problema: Pacman simplificado em grade
# ============================================================================

class PacmanAdversarialProblem:
    """
    Cenário simplificado para estudo de busca adversarial.

    Convenções do mapa:
    - '#' = parede
    - 'P' = posição inicial do Pacman
    - 'X' = posição inicial do fantasma
    - '.' = comida
    - ' ' = célula livre

    Regras resumidas:
    - Pacman (MAX) e fantasma (MIN) alternam turnos.
    - Pacman coleta comida ao entrar numa célula com '.'.
    - Se Pacman e fantasma ocupam a mesma célula, Pacman perde.
    - Se toda a comida for coletada, Pacman vence.
    - Há um limite máximo de plies para evitar jogos infinitos.
    """

    ACTIONS = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1),
        "S": (0, 0),   # ficar parado, útil para inspeção didática
    }

    def __init__(self, grid: List[str], max_ply: int = 40):
        """
        Inicializa o problema a partir de um grid textual.

        Parâmetros
        ----------
        grid : List[str]
            Lista de strings representando as linhas do mapa.
            Cada caractere é uma célula; todas as linhas devem ter o mesmo
            comprimento (grid retangular).
        max_ply : int
            Número máximo de jogadas antes de o jogo ser declarado empate.
            Evita que partidas AI vs AI rodem indefinidamente.

        Efeitos colaterais
        ------------------
        Preenche os atributos de leitura rápida:
        - self.walls   : conjunto de posições intransponíveis
        - self.pacman0 : posição inicial do Pacman
        - self.ghost0  : posição inicial do fantasma
        - self.foods0  : conjunto inicial de comidas
        - self.s0      : estado inicial completo (GameState)
        """
        self.grid = grid
        self.R = len(grid)
        self.C = len(grid[0]) if self.R else 0
        self.max_ply = max_ply

        self._validate_grid()
        self.walls = {
            (r, c)
            for r in range(self.R)
            for c in range(self.C)
            if self.grid[r][c] == "#"
        }
        self.pacman0 = self._find("P")
        self.ghost0 = self._find("X")
        self.foods0 = frozenset(
            (r, c)
            for r in range(self.R)
            for c in range(self.C)
            if self.grid[r][c] == "."
        )
        self.s0 = GameState(
            pacman=self.pacman0,
            ghost=self.ghost0,
            foods=self.foods0,
            turn="MAX",
            score=0,
            ply=0,
        )

    # ------------------------------------------------------------------
    # Infraestrutura básica
    # ------------------------------------------------------------------

    def _validate_grid(self) -> None:
        """
        Verifica a integridade estrutural do grid antes de qualquer uso.

        Lança ValueError se:
        - o grid estiver vazio ou tiver linhas de comprimento diferente;
        - houver caracteres não reconhecidos;
        - não houver exatamente um 'P' e um 'X';
        - não houver ao menos uma comida '.'.
        """
        if not self.grid:
            raise ValueError("Grid vazio.")
        cols = len(self.grid[0])
        if cols == 0:
            raise ValueError("Grid com linha vazia.")
        valid = {"#", "P", "X", ".", " "}
        for i, row in enumerate(self.grid, start=1):
            if len(row) != cols:
                raise ValueError(
                    f"Grid não retangular: linha {i} tem tamanho {len(row)}, esperado {cols}."
                )
            for ch in row:
                if ch not in valid:
                    raise ValueError(f"Caractere inválido no grid: {ch!r}")
        if sum(row.count("P") for row in self.grid) != 1:
            raise ValueError("Grid deve conter exatamente um 'P'.")
        if sum(row.count("X") for row in self.grid) != 1:
            raise ValueError("Grid deve conter exatamente um 'X'.")
        if sum(row.count(".") for row in self.grid) == 0:
            raise ValueError("Grid deve conter ao menos uma comida '.'.")

    def _find(self, ch: str) -> Pos:
        """
        Retorna a posição (linha, coluna) da primeira ocorrência do caractere ch no grid.

        Lança ValueError se ch não for encontrado.
        """
        for r in range(self.R):
            for c in range(self.C):
                if self.grid[r][c] == ch:
                    return (r, c)
        raise ValueError(f"Caractere {ch!r} não encontrado.")

    def in_bounds(self, pos: Pos) -> bool:
        """Retorna True se pos está dentro dos limites do grid."""
        r, c = pos
        return 0 <= r < self.R and 0 <= c < self.C

    def passable(self, pos: Pos) -> bool:
        """Retorna True se pos não é uma parede (célula transitável)."""
        return pos not in self.walls

    def neighbors(self, pos: Pos, include_stop: bool = False) -> List[Pos]:
        """
        Lista as posições vizinhas alcançáveis a partir de pos em um passo.

        Parâmetros
        ----------
        pos : Pos
            Posição de origem.
        include_stop : bool
            Se True, inclui a ação "S" (ficar parado), útil para BFS
            e análises que precisam considerar quietude.

        Retorna
        -------
        List[Pos]
            Posições vizinhas que estão dentro do grid e não são paredes.
        """
        out: List[Pos] = []
        for a, (dr, dc) in self.ACTIONS.items():
            if a == "S" and not include_stop:
                continue
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.in_bounds(nxt) and self.passable(nxt):
                out.append(nxt)
        return out

    def legal_actions(self, state: GameState) -> List[str]:
        """
        Retorna a lista de ações legais para o jogador que tem a vez em state.

        O jogador ativo é Pacman se state.turn == "MAX", fantasma se "MIN".
        A ação "S" (parar) é excluída das ações legais para forçar movimento
        contínuo — isso evita que agentes fujam de decisões difíceis ficando parados.
        """
        actions: List[str] = []
        pos = state.pacman if state.turn == "MAX" else state.ghost
        for a, (dr, dc) in self.ACTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.in_bounds(nxt) and self.passable(nxt):
                actions.append(a)
        return actions

    def result(self, state: GameState, action: str) -> GameState:
        """
        Aplica action ao state e retorna o novo GameState resultante.

        A função é pura: não modifica state; sempre cria um novo objeto.

        Semântica por turno
        -------------------
        - Turno MAX (Pacman): move Pacman para a nova célula; se houver
          comida nela, remove do conjunto e incrementa o score em 10.
          O turno passa para "MIN" e ply é incrementado.
        - Turno MIN (Fantasma): move o fantasma para a nova célula.
          O turno passa para "MAX" e ply é incrementado.

        Parâmetros
        ----------
        state : GameState
            Estado atual.
        action : str
            Uma das chaves de ACTIONS ("U", "D", "L", "R", "S").

        Retorna
        -------
        GameState
            Estado resultante da aplicação da ação.
        """
        dr, dc = self.ACTIONS[action]
        if state.turn == "MAX":
            new_pac = (state.pacman[0] + dr, state.pacman[1] + dc)
            foods = set(state.foods)
            score = state.score
            if new_pac in foods:
                foods.remove(new_pac)
                score += 10
            return GameState(
                pacman=new_pac,
                ghost=state.ghost,
                foods=frozenset(foods),
                turn="MIN",
                score=score,
                ply=state.ply + 1,
            )
        else:
            new_ghost = (state.ghost[0] + dr, state.ghost[1] + dc)
            return GameState(
                pacman=state.pacman,
                ghost=new_ghost,
                foods=state.foods,
                turn="MAX",
                score=state.score,
                ply=state.ply + 1,
            )

    # ------------------------------------------------------------------
    # Testes terminal / utilidade / avaliação
    # ------------------------------------------------------------------

    def collision(self, state: GameState) -> bool:
        """
        Retorna True se Pacman e o fantasma estão na mesma célula.

        Colisão é a condição de derrota para Pacman (MAX).
        """
        return state.pacman == state.ghost

    def win(self, state: GameState) -> bool:
        """
        Retorna True se Pacman venceu: coletou toda a comida sem colidir com o fantasma.
        """
        return len(state.foods) == 0 and not self.collision(state)

    def draw(self, state: GameState) -> bool:
        """
        Retorna True se o jogo terminou em empate por limite de jogadas.

        Ocorre quando ply >= max_ply sem vitória nem derrota.
        Essa condição é necessária para garantir que o algoritmo Minimax
        termine mesmo em grids onde nenhum lado consegue vencer.
        """
        return state.ply >= self.max_ply and not self.collision(state) and not self.win(state)

    def terminal_test(self, state: GameState) -> bool:
        """
        Retorna True se state é um nó terminal da árvore de busca.

        Um estado é terminal quando:
        - há colisão (Pacman perdeu),
        - Pacman coletou toda a comida (Pacman ganhou), ou
        - o limite de jogadas foi atingido (empate).
        """
        return self.collision(state) or self.win(state) or self.draw(state)

    def utility(self, state: GameState) -> float:
        """
        Retorna o valor de utilidade de um estado terminal, na perspectiva de MAX.

        Valores:
        - Colisão (derrota de MAX): -1000.0
        - Vitória de MAX: +1000.0 + pontuação acumulada (incentiva coletar
          comida rapidamente, não apenas "não perder")
        - Empate / estado não terminal: delega para eval() como fallback seguro.

        Nota: utility() só deve ser chamada em estados terminais; para estados
        intermediários, use eval().
        """
        if self.collision(state):
            return -1000.0
        if self.win(state):
            return 1000.0 + state.score
        return self.eval(state)

    def manhattan(self, a: Pos, b: Pos) -> int:
        """
        Retorna a distância de Manhattan entre duas posições.

        d_Manhattan(a, b) = |a.linha - b.linha| + |a.col - b.col|

        É uma heurística admissível para grids sem paredes — subestima
        (ou iguala) a distância real quando há paredes no caminho.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def shortest_path_distance(self, src: Pos, dst: Pos) -> Optional[int]:
        """
        Calcula a distância real (em número de passos) entre src e dst via BFS.

        Ao contrário da distância de Manhattan, leva em conta as paredes do mapa.
        Retorna None se dst não for alcançável a partir de src.

        Complexidade: O(R × C) no pior caso, onde R e C são as dimensões do grid.
        """
        if src == dst:
            return 0
        q = deque([(src, 0)])
        seen = {src}
        while q:
            cur, d = q.popleft()
            for nxt in self.neighbors(cur, include_stop=False):
                if nxt == dst:
                    return d + 1
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return None

    def mobility(self, pos: Pos) -> int:
        """
        Retorna o número de direções livres a partir de pos (grau de liberdade local).

        Valores típicos:
        - 1 → beco sem saída (dead end)
        - 2 → corredor
        - 3 ou 4 → cruzamento / espaço aberto
        """
        return len(self.neighbors(pos, include_stop=False))

    def dead_end(self, pos: Pos) -> bool:
        """
        Retorna True se pos é um beco sem saída (apenas 1 vizinho livre).

        Becos são perigosos para o Pacman: se o fantasma está próximo,
        Pacman pode ficar encurralado sem opção de fuga.
        """
        return self.mobility(pos) <= 1

    def corridor_like(self, pos: Pos) -> bool:
        """
        Retorna True se pos é um corredor (exatamente 2 vizinhos livres).

        Em corredores, Pacman só pode avançar ou recuar — mobilidade reduzida
        aumenta o risco quando o fantasma está próximo.
        """
        return self.mobility(pos) == 2

    def ghost_can_reach_pacman_soon(self, state: GameState, radius: int = 2) -> bool:
        """
        Retorna True se o fantasma pode alcançar o Pacman em até `radius` passos.

        Usa a distância real (BFS), não a de Manhattan. Parâmetro `radius`
        controla a definição de "proximidade imediata".
        """
        d = self.shortest_path_distance(state.ghost, state.pacman)
        return d is not None and d <= radius

    def non_quiescent(self, state: GameState) -> bool:
        """
        Detecta se state está em uma posição taticamente instável ("não-quiescente").

        Um estado quiescente é aquele onde a avaliação estática (eval) é confiável
        porque não há ação imediata capaz de alterar drasticamente o valor posicional.
        Estados não-quiescentes precisam de busca extra (extensão de quiescência)
        para evitar o "efeito de horizonte" — a ilusão de segurança logo antes
        de uma sequência de jogadas forçadas desfavoráveis.

        Sinais de não-quiescência detectados (heurística local):
        - Fantasma a ≤ 1 passo: colisão iminente.
        - Fantasma a ≤ 2 passos e Pacman está em beco: fuga impossível.
        - Fantasma a ≤ 2 passos, Pacman em corredor, fantasma com mobilidade
          igual ou maior: perseguição em corredor sem rota de escape.

        Nota: a função propositalmente *não* faz busca profunda; é um filtro
        rápido baseado em propriedades locais do estado.
        """
        d = self.shortest_path_distance(state.ghost, state.pacman)
        if d is None:
            return False
        if d <= 1:
            return True
        if d <= 2 and self.dead_end(state.pacman):
            return True
        if d <= 2 and self.corridor_like(state.pacman) and self.mobility(state.ghost) >= self.mobility(state.pacman):
            return True
        return False

    def eval(self, state: GameState) -> float:
        """
        Função de avaliação heurística para estados não-terminais.

        Estima o quão bom o estado é para MAX (Pacman), combinando vários
        componentes em uma soma ponderada. Valores positivos favorecem MAX;
        negativos favorecem MIN.

        Componentes
        -----------
        score : float
            Pontuação já acumulada (comidas coletadas).

        food_bonus : float  (+)
            12 / (dist_comida_mais_próxima + 1)
            Incentiva Pacman a se aproximar de comidas.

        mobility_bonus : float  (+)
            2 × número de vizinhos livres do Pacman.
            Recompensa posições abertas (mais opções de movimento).

        danger_penalty : float  (-)
            18 / (dist_fantasma + 1)
            Penaliza estar perto do fantasma; cresce rapidamente quando
            a distância diminui.

        remaining_food_penalty : float  (-)
            4 × quantidade de comidas restantes.
            Incentiva Pacman a reduzir o número de comidas pendentes.

        trap_penalty : float  (-)
            Penalidade adicional quando Pacman está em posição desfavorável
            com o fantasma próximo:
            - Beco com fantasma a ≤ 2 passos: -40.
            - Corredor com fantasma a ≤ 2 passos: -18.

        Nota sobre calibração: os pesos foram escolhidos empiricamente para
        o conjunto de grids do catálogo. Para outros mapas, pode ser necessário
        reajustá-los — explore isso como exercício!
        """
        if self.collision(state):
            return -1000.0
        if self.win(state):
            return 1000.0 + state.score

        score = float(state.score)

        # Comida mais próxima
        if state.foods:
            food_dist = min(self.shortest_path_distance(state.pacman, f) or 999 for f in state.foods)
        else:
            food_dist = 0

        ghost_dist = self.shortest_path_distance(state.pacman, state.ghost)
        if ghost_dist is None:
            ghost_dist = 999

        mobility_bonus = 2.0 * self.mobility(state.pacman)
        food_bonus = 12.0 / (food_dist + 1)
        danger_penalty = 18.0 / (ghost_dist + 1)
        remaining_food_penalty = 4.0 * len(state.foods)

        trap_penalty = 0.0
        if self.dead_end(state.pacman) and ghost_dist <= 2:
            trap_penalty += 40.0
        elif self.corridor_like(state.pacman) and ghost_dist <= 2:
            trap_penalty += 18.0

        return score + food_bonus + mobility_bonus - danger_penalty - remaining_food_penalty - trap_penalty

    # ------------------------------------------------------------------
    # Renderização
    # ------------------------------------------------------------------

    def render(self, state: GameState) -> str:
        """
        Converte state em uma representação textual do grid para exibição no terminal.

        Símbolos usados:
        - 'P' : Pacman
        - 'X' : fantasma
        - '!' : colisão (Pacman e fantasma na mesma célula)
        - '#' : parede
        - '.' : comida
        - ' ' : célula vazia

        Retorna uma string com linhas separadas por '\\n'.
        """
        out: List[str] = []
        for r in range(self.R):
            row_chars: List[str] = []
            for c in range(self.C):
                pos = (r, c)
                if pos == state.pacman == state.ghost:
                    row_chars.append("!")
                elif pos == state.pacman:
                    row_chars.append("P")
                elif pos == state.ghost:
                    row_chars.append("X")
                elif pos in self.walls:
                    row_chars.append("#")
                elif pos in state.foods:
                    row_chars.append(".")
                else:
                    row_chars.append(" ")
            out.append("".join(row_chars))
        return "\n".join(out)


# ============================================================================
# Minimax / alfa-beta / busca com avaliação / quiescência
# ============================================================================

@dataclass
class SearchResult:
    """
    Empacota o resultado de uma chamada ao algoritmo de busca adversarial.

    Atributos
    ---------
    action : str
        Ação escolhida pelo agente para o estado raiz.
    value : float
        Valor minimax (ou heurístico) associado a essa ação.
    nodes : int
        Número total de nós expandidos durante a busca.
        Útil para comparar a eficiência dos algoritmos — compare
        minimax vs. alfa-beta para o mesmo grid e profundidade!
    """
    action: str
    value: float
    nodes: int


class Counter:
    """Contador mutável de nós expandidos, compartilhado entre chamadas recursivas."""
    def __init__(self) -> None:
        self.nodes = 0


def minimax_decision(problem: PacmanAdversarialProblem, state: GameState) -> SearchResult:
    """
    Implementa o algoritmo Minimax puro (sem poda, sem limite de profundidade).

    Expande a árvore de jogo *completamente* até estados terminais e escolhe
    a ação que maximiza (ou minimiza) a utilidade, assumindo que o adversário
    joga de forma ótima.

    Adequado apenas para grids muito pequenos, pois o custo cresce
    exponencialmente com a profundidade da árvore. Use --algorithm minimax
    apenas com o grid "mini" ou configurações de max_ply muito pequenas para
    não travar.

    Parâmetros
    ----------
    problem : PacmanAdversarialProblem
        Definição do problema (regras, grid, funções de utilidade).
    state : GameState
        Estado raiz a partir do qual a busca começa.

    Retorna
    -------
    SearchResult
        Ação ótima, valor minimax e contagem de nós expandidos.
    """
    counter = Counter()

    def value(s: GameState) -> float:
        counter.nodes += 1
        if problem.terminal_test(s):
            return problem.utility(s)
        if s.turn == "MAX":
            return max_value(s)
        return min_value(s)

    def max_value(s: GameState) -> float:
        v = -math.inf
        for a in problem.legal_actions(s):
            v = max(v, value(problem.result(s, a)))
        return v

    def min_value(s: GameState) -> float:
        v = math.inf
        for a in problem.legal_actions(s):
            v = min(v, value(problem.result(s, a)))
        return v

    best_action = None
    best_value = -math.inf if state.turn == "MAX" else math.inf
    for a in problem.legal_actions(state):
        child_val = value(problem.result(state, a))
        if state.turn == "MAX":
            if child_val > best_value:
                best_value = child_val
                best_action = a
        else:
            if child_val < best_value:
                best_value = child_val
                best_action = a

    assert best_action is not None
    return SearchResult(best_action, best_value, counter.nodes)


def alphabeta_decision(
    problem: PacmanAdversarialProblem,
    state: GameState,
    depth_limit: Optional[int] = None,
    use_quiescence: bool = False,
    quiescence_limit: int = 4,
) -> SearchResult:
    """
    Implementa Minimax com poda alfa-beta, limite de profundidade opcional
    e extensão de quiescência opcional.

    Modos de operação (controlados pelos parâmetros):
    -------------------------------------------------
    1. **Alfa-beta puro** (depth_limit=None, use_quiescence=False):
       Equivalente ao minimax, mas com poda. Expande até estados terminais;
       não usa eval(). Mesma qualidade de decisão que minimax_decision,
       mas com menos nós — demonstra a eficiência da poda.

    2. **H-Minimax** (depth_limit=d, use_quiescence=False):
       Expande até profundidade d e aplica eval() nos nós de corte.
       Permite jogar em grids grandes onde o minimax completo é inviável.
       Sujeito ao efeito de horizonte.

    3. **Quiescent search** (depth_limit=d, use_quiescence=True):
       Como H-Minimax, mas estados "não-quiescentes" na fronteira recebem
       plies extras (até quiescence_limit) antes de serem avaliados
       estaticamente. Reduz o efeito de horizonte em situações de perigo
       iminente.

    Ordenação de ações
    ------------------
    Antes de expandir, as ações são ordenadas pela avaliação heurística
    dos estados filhos (maiores primeiro para MAX, menores para MIN).
    Isso concentra as podas no início da busca, aumentando o speedup
    da poda alfa-beta em relação ao minimax ingênuo.

    Parâmetros
    ----------
    problem : PacmanAdversarialProblem
        Definição do problema.
    state : GameState
        Estado raiz.
    depth_limit : Optional[int]
        Profundidade máxima de busca. None = sem limite (alfa-beta puro).
    use_quiescence : bool
        Se True, ativa a extensão de quiescência.
    quiescence_limit : int
        Número máximo de plies extras permitidos pela extensão de quiescência.

    Retorna
    -------
    SearchResult
        Melhor ação encontrada, seu valor estimado e o número de nós expandidos.
    """
    counter = Counter()

    def cutoff(s: GameState, depth: int) -> bool:
        if problem.terminal_test(s):
            return True
        if depth_limit is None:
            return False
        if depth < depth_limit:
            return False
        if use_quiescence and problem.non_quiescent(s):
            return False
        return depth >= depth_limit

    def value(s: GameState, depth: int, alpha: float, beta: float, q_extra: int) -> float:
        counter.nodes += 1
        if problem.terminal_test(s):
            return problem.utility(s)
        if depth_limit is not None and depth >= depth_limit:
            if use_quiescence and problem.non_quiescent(s) and q_extra < quiescence_limit:
                # Continua apenas por estar em estado taticamente instável.
                pass
            else:
                return problem.eval(s)
        if s.turn == "MAX":
            return max_value(s, depth, alpha, beta, q_extra)
        return min_value(s, depth, alpha, beta, q_extra)

    def ordered_actions(s: GameState) -> List[str]:
        """
        Retorna as ações legais de s ordenadas pela avaliação heurística dos filhos.

        MAX prefere filhos com maior eval (ordem decrescente).
        MIN prefere filhos com menor eval (ordem crescente).
        A ordenação antecipa cortes alfa-beta mais cedo na busca.
        """
        acts = problem.legal_actions(s)
        scored = []
        for a in acts:
            child = problem.result(s, a)
            scored.append((problem.eval(child), a))
        reverse = s.turn == "MAX"
        scored.sort(reverse=reverse)
        return [a for _, a in scored]

    def max_value(s: GameState, depth: int, alpha: float, beta: float, q_extra: int) -> float:
        v = -math.inf
        for a in ordered_actions(s):
            child = problem.result(s, a)
            child_q = q_extra + 1 if depth_limit is not None and depth >= depth_limit else q_extra
            v = max(v, value(child, depth + 1, alpha, beta, child_q))
            if v >= beta:
                return v          # Corte beta: MIN nunca escolheria este ramo
            alpha = max(alpha, v)
        return v

    def min_value(s: GameState, depth: int, alpha: float, beta: float, q_extra: int) -> float:
        v = math.inf
        for a in ordered_actions(s):
            child = problem.result(s, a)
            child_q = q_extra + 1 if depth_limit is not None and depth >= depth_limit else q_extra
            v = min(v, value(child, depth + 1, alpha, beta, child_q))
            if v <= alpha:
                return v          # Corte alfa: MAX nunca escolheria este ramo
            beta = min(beta, v)
        return v

    best_action = None
    if state.turn == "MAX":
        best_value = -math.inf
        alpha, beta = -math.inf, math.inf
        for a in ordered_actions(state):
            child_val = value(problem.result(state, a), 1, alpha, beta, 0)
            if child_val > best_value:
                best_value = child_val
                best_action = a
            alpha = max(alpha, best_value)
    else:
        best_value = math.inf
        alpha, beta = -math.inf, math.inf
        for a in ordered_actions(state):
            child_val = value(problem.result(state, a), 1, alpha, beta, 0)
            if child_val < best_value:
                best_value = child_val
                best_action = a
            beta = min(beta, best_value)

    assert best_action is not None
    return SearchResult(best_action, best_value, counter.nodes)


# ============================================================================
# Catálogo de grids
# ============================================================================

GRID_CATALOG = {
    "mini": [
        "#########",
        "#P .   X#",
        "# ### # #",
        "#   .   #",
        "#########",
    ],
    "beco": [
        "###########",
        "# P# .    #",
        "#  #      #",
        "# X       #",
        "###########",
    ],
    "duas_comidas": [
        "###########",
        "#P .   # X#",
        "# ###    .#",
        "#         #",
        "###########",
    ],
    "lab": [
        "#############",
        "#P .   #   X#",
        "# ### ## #  #",
        "# .     .   #",
        "#############",
    ],
}


# ============================================================================
# Execução jogável
# ============================================================================

def clear_screen() -> None:
    """Limpa o terminal usando código de escape ANSI (compatível com Linux/macOS)."""
    print("\033[H\033[J", end="")


def choose_action_ai(
    problem: PacmanAdversarialProblem,
    state: GameState,
    algorithm: str,
    depth: int,
    quiescent: bool,
) -> SearchResult:
    """
    Despacha a chamada ao algoritmo de busca correto conforme o argumento algorithm.

    Parâmetros
    ----------
    problem : PacmanAdversarialProblem
        Definição do problema.
    state : GameState
        Estado atual a partir do qual a IA decidirá.
    algorithm : str
        Um dos valores: "minimax", "alphabeta", "hminimax", "quiescent".
    depth : int
        Limite de profundidade para buscas heurísticas (hminimax e quiescent).
        Ignorado por minimax e alphabeta (que expandem até folhas terminais).
    quiescent : bool
        Parâmetro legacy; a ativação da quiescência é inferida de algorithm.

    Retorna
    -------
    SearchResult
        Melhor ação e estatísticas da busca.

    Lança
    -----
    ValueError se algorithm for um valor não reconhecido.
    """
    if algorithm == "minimax":
        return minimax_decision(problem, state)
    if algorithm == "alphabeta":
        return alphabeta_decision(problem, state, depth_limit=None)
    if algorithm == "hminimax":
        return alphabeta_decision(problem, state, depth_limit=depth, use_quiescence=False)
    if algorithm == "quiescent":
        return alphabeta_decision(problem, state, depth_limit=depth, use_quiescence=True)
    raise ValueError(f"Algoritmo desconhecido: {algorithm}")


def print_state(problem: PacmanAdversarialProblem, state: GameState) -> None:
    """
    Exibe o estado atual do jogo no terminal: grid visual e métricas principais.

    Informações exibidas:
    - Representação visual do grid (render)
    - Turno atual (MAX / MIN)
    - Ply atual e limite máximo
    - Pontuação acumulada
    - Quantidade de comidas restantes
    - Valor da função de avaliação heurística no estado atual
    - Se o estado é considerado não-quiescente (instável taticamente)
    """
    print(problem.render(state))
    print()
    print(f"Turno : {state.turn}")
    print(f"Ply   : {state.ply}/{problem.max_ply}")
    print(f"Score : {state.score}")
    print(f"Comidas restantes: {len(state.foods)}")
    print(f"EVAL(state): {problem.eval(state):.2f}")
    print(f"Não quiescente? {'sim' if problem.non_quiescent(state) else 'não'}")


def parse_human_action(problem: PacmanAdversarialProblem, state: GameState) -> str:
    """
    Solicita e valida a ação do jogador humano via entrada no terminal.

    Mapeia teclas simples (c/b/e/d/Enter) para os códigos internos de ação
    (U/D/L/R/S) e verifica se a ação escolhida é legal no estado atual.
    Repete a solicitação até receber uma entrada válida.

    Mapeamento de teclas:
    - c → U (cima / Up)
    - b → D (baixo / Down)
    - e → L (esquerda / Left)
    - d → R (direita / Right)
    - Enter (vazio) → S (parar / Stop)

    Parâmetros
    ----------
    problem : PacmanAdversarialProblem
        Definição do problema (para consultar ações legais).
    state : GameState
        Estado atual (define quais ações são legais).

    Retorna
    -------
    str
        Código de ação legal escolhido pelo usuário ("U", "D", "L", "R" ou "S").
    """
    mapping = {
        "c": "U",  # cima
        "b": "D",  # baixo
        "e": "L",  # esquerda
        "d": "R",  # direita
        "": "S",   # Enter = parar
    }
    
    action_descriptions = {
        "U": "cima",
        "D": "baixo", 
        "L": "esquerda",
        "R": "direita",
        "S": "parar"
    }
    
    legal = set(problem.legal_actions(state))
    legal_descriptions = [action_descriptions[a] for a in sorted(legal)]
    
    print()
    print("Instruções de controle:")
    print("  c = cima (Up)")
    print("  b = baixo (Down)")
    print("  e = esquerda (Left)")
    print("  d = direita (Right)")
    print("  Enter = parar (Stop)")
    print(f"Ações legais neste turno: {legal_descriptions}")
    while True:
        raw = input("Sua ação [c/b/e/d ou Enter=parar]: ").strip().lower()
        if raw in mapping and mapping[raw] in legal:
            return mapping[raw]
        print(f"Ação inválida. Ações legais: {legal_descriptions}")


def run_game(
    problem: PacmanAdversarialProblem,
    pacman_mode: str,
    ghost_mode: str,
    algorithm: str,
    depth: int,
    quiescent: bool,
    seed: int = 0,
    auto_mode: bool = False,
) -> None:
    """
    Executa o loop principal do jogo até um estado terminal ser atingido.

    A cada turno:
    1. Exibe o estado atual via print_state.
    2. Verifica se o estado é terminal; se sim, anuncia o resultado e encerra.
    3. Obtém a ação do jogador ativo (humano, IA ou aleatório).
    4. Aplica a ação e avança para o próximo estado.

    Modos de controle por jogador:
    - "human"  : solicita entrada via teclado (parse_human_action).
    - "ai"     : executa o algoritmo de busca configurado (choose_action_ai).
    - "random" : escolhe uma ação legal aleatoriamente (útil como linha de base).

    Parâmetros
    ----------
    problem : PacmanAdversarialProblem
        Definição do problema.
    pacman_mode : str
        Modo de controle do Pacman ("human", "ai" ou "random").
    ghost_mode : str
        Modo de controle do fantasma ("human", "ai" ou "random").
    algorithm : str
        Algoritmo de busca usado pelos jogadores IA.
    depth : int
        Profundidade para algoritmos heurísticos.
    quiescent : bool
        Ativa extensão de quiescência (quando algorithm == "quiescent").
    seed : int
        Semente do gerador aleatório (para reprodutibilidade no modo random).
    auto_mode : bool
        Se True, não pausa entre turnos — útil para partidas IA vs IA rápidas.
        É ativado automaticamente quando ambos os jogadores são IA.
    """
    random.seed(seed)
    state = problem.s0

    while True:
        clear_screen()
        print_state(problem, state)

        if problem.terminal_test(state):
            print()
            if problem.collision(state):
                print("Resultado: Pacman foi capturado. Vitória de MIN.")
            elif problem.win(state):
                print("Resultado: Pacman comeu toda a comida. Vitória de MAX.")
            else:
                print("Resultado: limite de jogadas atingido.")
            print(f"Utility final: {problem.utility(state):.2f}")
            break

        if state.turn == "MAX" and pacman_mode == "human":
            action = parse_human_action(problem, state)
            state = problem.result(state, action)
            continue

        if state.turn == "MIN" and ghost_mode == "human":
            action = parse_human_action(problem, state)
            state = problem.result(state, action)
            continue

        if state.turn == "MAX" and pacman_mode == "random":
            action = random.choice(problem.legal_actions(state))
            print(f"\nPacman aleatório escolheu: {action}")
            input("Enter para continuar...")
            state = problem.result(state, action)
            continue

        if state.turn == "MIN" and ghost_mode == "random":
            action = random.choice(problem.legal_actions(state))
            print(f"\nFantasma aleatório escolheu: {action}")
            input("Enter para continuar...")
            state = problem.result(state, action)
            continue

        result = choose_action_ai(problem, state, algorithm, depth, quiescent)
        actor = "Pacman" if state.turn == "MAX" else "Fantasma"
        print()
        print(f"{actor} ({algorithm}) escolheu: {result.action}")
        print(f"Valor estimado: {result.value:.2f}")
        print(f"Nós examinados: {result.nodes}")
        if not auto_mode:
            input("Enter para continuar...")
        state = problem.result(state, result.action)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """
    Define e processa os argumentos de linha de comando.

    Retorna um namespace com os valores de todos os parâmetros configuráveis:
    grid, depth, max_ply, algorithm, pacman_mode, ghost_mode, seed e auto_mode.
    Use --help para ver a descrição completa de cada parâmetro.
    """
    parser = argparse.ArgumentParser(
        description="Pacman simplificado para estudo de busca adversarial."
    )
    parser.add_argument(
        "--grid",
        default="lab",
        choices=sorted(GRID_CATALOG.keys()),
        help="Nome do grid do catálogo.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Profundidade da busca heurística (H-Minimax / quiescent).",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=40,
        help="Limite máximo de plies do jogo.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["minimax", "alphabeta", "hminimax", "quiescent"],
        default="quiescent",
        help="Algoritmo usado pelos jogadores controlados por IA.",
    )
    parser.add_argument(
        "--pacman-mode",
        choices=["human", "ai", "random"],
        default="ai",
        help="Como Pacman será controlado.",
    )
    parser.add_argument(
        "--ghost-mode",
        choices=["human", "ai", "random"],
        default="ai",
        help="Como o fantasma será controlado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semente para modo aleatório.",
    )
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        help="Executa sem parar para input entre turns (útil para IA vs IA).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Ponto de entrada do script. Lê os argumentos da linha de comando,
    instancia o problema com o grid escolhido e inicia a partida.

    ─────────────────────────────────────────────────────────────────────────
    EXEMPLOS DE USO
    ─────────────────────────────────────────────────────────────────────────

    1. Execução padrão — IA vs IA no grid "lab" com busca quiescente:

        python3 busca_adversarial_minimax.py

        Ambos os jogadores são controlados por IA usando o algoritmo
        quiescent (H-Minimax com extensão de quiescência, profundidade 4).
        Como ambos são IA, o modo automático é ativado e a partida roda
        sem pausas. Use para observar como os dois agentes interagem.

    ─────────────────────────────────────────────────────────────────────────

    2. Grid "beco" com H-Minimax de profundidade 3:

        python3 busca_adversarial_minimax.py --grid beco --algorithm hminimax --depth 3

        O grid "beco" tem um corredor estreito que dificulta a fuga do
        Pacman. Com H-Minimax puro (sem quiescência) e profundidade 3,
        o agente pode sofrer do *efeito de horizonte*: ele avalia o estado
        como seguro logo antes de uma sequência forçada desfavorável que
        está além do limite de profundidade. Compare com o exemplo seguinte
        para visualizar a diferença.

    ─────────────────────────────────────────────────────────────────────────

    3. Grid "beco" com busca quiescente de profundidade 2:

        python3 busca_adversarial_minimax.py --grid beco --algorithm quiescent --depth 2

        Mesma configuração do exemplo anterior, mas com profundidade 2 e
        extensão de quiescência ativada. Quando o agente detecta que o estado
        na fronteira da busca é taticamente instável (fantasma próximo em
        corredor ou beco), ele estende a busca por até 4 plies extras antes
        de aplicar eval(). Observe se o número de nós examinados aumenta em
        situações de perigo — esse é o custo da quiescência em ação.

    ─────────────────────────────────────────────────────────────────────────

    4. Você controla o Pacman; fantasma usa IA adversarial:

        python3 busca_adversarial_minimax.py --pacman-mode human --ghost-mode ai --algorithm quiescent

        Use as teclas c/b/e/d (ou Enter para parar) para mover o Pacman.
        O fantasma escolhe automaticamente a melhor jogada segundo a busca
        quiescente. Experimente diferentes rotas e tente coletar todas as
        comidas sem ser capturado. Bom para sentir na prática o que a IA
        "vê" que você não vê.

    ─────────────────────────────────────────────────────────────────────────

    5. Pacman usa IA; fantasma move aleatoriamente:

        python3 busca_adversarial_minimax.py --pacman-mode ai --ghost-mode random --algorithm hminimax

        O Pacman usa H-Minimax enquanto o fantasma é aleatório. Serve como
        linha de base para comparar a qualidade da busca: um agente bem
        calibrado deve vencer consistentemente contra um oponente aleatório.
        Use --seed N para repetir a mesma sequência de jogadas aleatórias e
        isolar o comportamento do algoritmo. Execute várias vezes com seeds
        diferentes para medir a taxa de vitória.

    ─────────────────────────────────────────────────────────────────────────
    """
    args = parse_args()
    problem = PacmanAdversarialProblem(GRID_CATALOG[args.grid], max_ply=args.max_ply)
    
    # Auto-ativa modo automático se ambos os jogadores são IA
    auto_mode = args.auto_mode or (args.pacman_mode == "ai" and args.ghost_mode == "ai")
    
    run_game(
        problem,
        pacman_mode=args.pacman_mode,
        ghost_mode=args.ghost_mode,
        algorithm=args.algorithm,
        depth=args.depth,
        quiescent=(args.algorithm == "quiescent"),
        seed=args.seed,
        auto_mode=auto_mode,
    )


if __name__ == "__main__":
    main()
