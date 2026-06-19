from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from collections import deque
import heapq

State = Tuple[int, int]  # (row, col)


# ----------------------------
# Problema: Labirinto em grade
# ----------------------------

class MazeProblem:
    """
    Labirinto em grade.
    - '#' = parede
    - 'S' = início
    - 'G' = objetivo
    - '.' = livre
    Movimentos: N, S, E, W. Custo por passo: 1.
    """

    ACTIONS: Dict[str, Tuple[int, int]] = {
        "N": (-1, 0),
        "S": (1, 0),
        "W": (0, -1),
        "E": (0, 1),
    }

    def __init__(self, grid: List[str]):
        self.grid = grid
        self.R = len(grid)
        self.C = len(grid[0]) if self.R > 0 else 0
        self.s0 = self._find("S")
        self.goal = self._find("G")

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
        out = []
        for r in range(self.R):
            row = []
            for c in range(self.C):
                ch = self.grid[r][c]
                if (r, c) in path_set and ch not in ("S", "G"):
                    row.append("*")
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

def generic_graph_search(
    problem: MazeProblem,
    frontier_push: FrontierPush,
    frontier_pop: FrontierPop,
    frontier_empty: FrontierEmpty,
) -> Optional[Node]:
    start = Node(state=problem.s0, parent=None, action=None, g=0)
    frontier_push(start)

    explored: Set[State] = set()
    in_frontier: Set[State] = {start.state}  # pra evitar duplicar estado na fronteira

    while not frontier_empty():
        n = frontier_pop()
        in_frontier.discard(n.state)

        if problem.GoalTest(n.state):
            return n

        explored.add(n.state)

        for a in problem.ACTIONS_fn(n.state):
            s2 = problem.T(n.state, a)
            if s2 in explored or s2 in in_frontier:
                continue
            n2 = Node(state=s2, parent=n, action=a, g=n.g + problem.c(n.state, a))
            frontier_push(n2)
            in_frontier.add(s2)

    return None


# ----------------------------
# Instâncias: BFS, DFS, UCS
# ----------------------------

def bfs(problem: MazeProblem) -> Optional[Node]:
    q = deque()

    def push(n: Node) -> None:
        q.append(n)

    def pop() -> Node:
        return q.popleft()

    def empty() -> bool:
        return len(q) == 0

    return generic_graph_search(problem, push, pop, empty)

def dfs(problem: MazeProblem) -> Optional[Node]:
    st: List[Node] = []

    def push(n: Node) -> None:
        st.append(n)

    def pop() -> Node:
        return st.pop()

    def empty() -> bool:
        return len(st) == 0

    return generic_graph_search(problem, push, pop, empty)

def ucs(problem: MazeProblem) -> Optional[Node]:
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


# ----------------------------
# Demo com o labirinto 3x3
# ----------------------------

if __name__ == "__main__":
    # Seu labirinto 3x3:
    # S . .
    # # # .
    # . . G
    grid = [
        "S..",
        "##.",
        "..G",
    ]

    prob = MazeProblem(grid)

    for name, solver in [("BFS", bfs), ("DFS", dfs), ("UCS", ucs)]:
        goal_node = solver(prob)
        print(f"\n=== {name} ===")
        if goal_node is None:
            print("Falha: sem solução.")
            continue

        states, actions = reconstruct_path(goal_node)
        print("Ações:", actions)
        print("Custo g:", goal_node.g)
        print("Caminho (estados):", states)
        print(prob.render_with_path(states))