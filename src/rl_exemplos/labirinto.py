from __future__ import annotations
import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from collections import deque
import heapq

State = Tuple[int, int]  # (row, col)


DEFAULT_GRID = [
    "S..........#",
    "######.###.#",
    "#....#.###.#",
    "#.##.#.....#",
    "#.##.#####.#",
    "#.##.....#.#",
    "#.######.#.#",
    "#........#.G",
    "############",
]


# ----------------------------
# Problema: Labirinto em grade
# ----------------------------

class MazeProblem:
    """
    Labirinto em grade.
    - '#' = parede
    - 'S' = início
    - 'G' = objetivo
    - '.' ou espaço = livre
    Movimentos: N, S, E, W. Custo por passo: 1.
    """

    ACTIONS: Dict[str, Tuple[int, int]] = {
        "N": (-1, 0),
        "S": (1, 0),
        "W": (0, -1),
        "E": (0, 1),
    }

    def __init__(self, grid: List[str]):
        self._validate_grid(grid)
        self.grid = grid
        self.R = len(grid)
        self.C = len(grid[0]) if self.R > 0 else 0
        self.s0 = self._find("S")
        self.goal = self._find("G")

    def _validate_grid(self, grid: List[str]) -> None:
        if not grid:
            raise ValueError("O labirinto não pode ser vazio.")

        width = len(grid[0])
        if width == 0:
            raise ValueError("As linhas do labirinto não podem ser vazias.")

        if any(len(row) != width for row in grid):
            raise ValueError("Todas as linhas do labirinto devem ter o mesmo tamanho.")

        chars = "".join(grid)
        if chars.count("S") != 1:
            raise ValueError("O labirinto deve conter exatamente um estado inicial 'S'.")
        if chars.count("G") != 1:
            raise ValueError("O labirinto deve conter exatamente um objetivo 'G'.")

        allowed = {"#", ".", " ", "S", "G"}
        invalid = sorted(set(chars) - allowed)
        if invalid:
            raise ValueError(f"Caracteres inválidos no labirinto: {invalid}")

    def _find(self, ch: str) -> State:
        for r in range(self.R):
            for c in range(self.C):
                if self.grid[r][c] == ch:
                    return (r, c)
        raise ValueError(f"Caractere {ch} não encontrado no grid.")

    def in_bounds(self, s: State) -> bool:
        r, c = s
        return 0 <= r < self.R and 0 <= c < self.C

    def passable(self, s: State) -> bool:
        r, c = s
        return self.grid[r][c] != "#"

    def GoalTest(self, s: State) -> bool:
        return s == self.goal

    def ACTIONS_fn(self, s: State) -> Iterable[str]:
        for a, (dr, dc) in self.ACTIONS.items():
            s2 = (s[0] + dr, s[1] + dc)
            if self.in_bounds(s2) and self.passable(s2):
                yield a

    def T(self, s: State, a: str) -> State:
        dr, dc = self.ACTIONS[a]
        return (s[0] + dr, s[1] + dc)

    def c(self, s: State, a: str) -> int:
        return 1

    def render_with_path(self, path_states: List[State]) -> str:
        path_set = set(path_states)
        return self.render(path=path_set)

    def render(
        self,
        path: Optional[Set[State]] = None,
        agent: Optional[State] = None,
        explored: Optional[Set[State]] = None,
    ) -> str:
        path = path or set()
        explored = explored or set()
        out = []
        for r in range(self.R):
            row = []
            for c in range(self.C):
                s = (r, c)
                ch = self.grid[r][c]
                if agent == s:
                    row.append("@")
                elif s in path and ch not in ("S", "G"):
                    row.append("*")
                elif s in explored and ch not in ("S", "G"):
                    row.append("+")
                else:
                    row.append(ch)
            out.append("".join(row))
        return "\n".join(out)


# ----------------------------
# Nó de busca + reconstrução
# ----------------------------

@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    action: Optional[str]
    g: int  # custo acumulado

def reconstruct_path(n: Node) -> Tuple[List[State], List[str]]:
    states: List[State] = []
    actions: List[str] = []
    cur: Optional[Node] = n
    while cur is not None:
        states.append(cur.state)
        if cur.action is not None:
            actions.append(cur.action)
        cur = cur.parent
    states.reverse()
    actions.reverse()
    return states, actions


# -----------------------------------------
# Esquema genérico: busca em grafo (explored)
# -----------------------------------------

FrontierPop = Callable[[], Node]
FrontierPush = Callable[[Node], None]
FrontierEmpty = Callable[[], bool]

@dataclass
class SearchResult:
    goal_node: Optional[Node]
    explored_order: List[State]

def generic_graph_search(
    problem: MazeProblem,
    frontier_push: FrontierPush,
    frontier_pop: FrontierPop,
    frontier_empty: FrontierEmpty,
) -> SearchResult:
    start = Node(state=problem.s0, parent=None, action=None, g=0)
    frontier_push(start)

    explored: Set[State] = set()
    explored_order: List[State] = []
    in_frontier: Set[State] = {start.state}  # pra evitar duplicar estado na fronteira

    while not frontier_empty():
        n = frontier_pop()
        in_frontier.discard(n.state)

        if problem.GoalTest(n.state):
            return SearchResult(goal_node=n, explored_order=explored_order)

        explored.add(n.state)
        explored_order.append(n.state)

        for a in problem.ACTIONS_fn(n.state):
            s2 = problem.T(n.state, a)
            if s2 in explored or s2 in in_frontier:
                continue
            n2 = Node(state=s2, parent=n, action=a, g=n.g + problem.c(n.state, a))
            frontier_push(n2)
            in_frontier.add(s2)

    return SearchResult(goal_node=None, explored_order=explored_order)


# ----------------------------
# Instâncias: BFS, DFS, UCS
# ----------------------------

def bfs(problem: MazeProblem) -> SearchResult:
    q = deque()

    def push(n: Node) -> None:
        q.append(n)

    def pop() -> Node:
        return q.popleft()

    def empty() -> bool:
        return len(q) == 0

    return generic_graph_search(problem, push, pop, empty)

def dfs(problem: MazeProblem) -> SearchResult:
    st: List[Node] = []

    def push(n: Node) -> None:
        st.append(n)

    def pop() -> Node:
        return st.pop()

    def empty() -> bool:
        return len(st) == 0

    return generic_graph_search(problem, push, pop, empty)

def ucs(problem: MazeProblem) -> SearchResult:
    heap: List[Tuple[int, int, Node]] = []
    counter = 0  # desempate estável

    def push(n: Node) -> None:
        nonlocal counter
        heapq.heappush(heap, (n.g, counter, n))
        counter += 1

    def pop() -> Node:
        return heapq.heappop(heap)[2]

    def empty() -> bool:
        return len(heap) == 0

    return generic_graph_search(problem, push, pop, empty)


SOLVERS = {
    "bfs": bfs,
    "dfs": dfs,
    "ucs": ucs,
}


def load_maze(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f]
    return rows


def clear_screen() -> None:
    print("\033[H\033[J", end="")


def animate_solution(
    problem: MazeProblem,
    path_states: List[State],
    explored_order: List[State],
    delay: float,
    show_explored: bool,
) -> None:
    explored = set(explored_order) if show_explored else set()
    path_so_far: Set[State] = set()

    for state in path_states:
        path_so_far.add(state)
        clear_screen()
        print(problem.render(path=path_so_far, agent=state, explored=explored))
        time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstra busca em um labirinto configurável."
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=sorted(SOLVERS),
        default="bfs",
        help="algoritmo de busca usado na solução",
    )
    parser.add_argument(
        "-m",
        "--maze-file",
        help="arquivo texto com o labirinto; use # para parede, . ou espaço para livre, S para início e G para objetivo",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.25,
        help="atraso, em segundos, entre os quadros da animação",
    )
    parser.add_argument(
        "--show-explored",
        action="store_true",
        help="marca com + os estados expandidos antes de animar a solução",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="mostra apenas o resultado final, sem animação",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid = load_maze(args.maze_file) if args.maze_file else DEFAULT_GRID
    problem = MazeProblem(grid)
    result = SOLVERS[args.algorithm](problem)

    print(f"Algoritmo: {args.algorithm.upper()}")
    print(f"Estados expandidos: {len(result.explored_order)}")

    if result.goal_node is None:
        print("Falha: sem solução.")
        print(problem.render(explored=set(result.explored_order)))
        return 1

    states, actions = reconstruct_path(result.goal_node)
    print("Ações:", actions)
    print("Custo g:", result.goal_node.g)
    print("Caminho (estados):", states)

    if args.no_animation:
        print(problem.render_with_path(states))
        return 0

    print("\nIniciando animação...")
    time.sleep(1.0)
    animate_solution(
        problem=problem,
        path_states=states,
        explored_order=result.explored_order,
        delay=args.delay,
        show_explored=args.show_explored,
    )
    print("\nLegenda: # parede, . livre, S início, G objetivo, @ agente, * caminho, + expandido")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAnimação interrompida.", file=sys.stderr)
        raise SystemExit(130)
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(2)
