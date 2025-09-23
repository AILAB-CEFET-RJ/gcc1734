"""
Este programa implementa uma versão unificada do jogo Conecta 4 (Connect Four),
onde o jogador humano pode disputar contra uma Inteligência Artificial (IA)
utilizando um dos dois algoritmos disponíveis: Monte Carlo Tree Search (MCTS)
ou Minimax com função de avaliação heurística (HEF). O algoritmo a ser usado
pela IA é definido pelo usuário via linha de comando.

Principais funcionalidades:
- Inicializa um tabuleiro vazio de Connect Four.
- Permite ao usuário escolher seu símbolo (X ou O).
- Alterna turnos entre o humano e a IA.
- O humano escolhe interativamente a coluna em que deseja jogar.
- A IA decide sua jogada de acordo com o algoritmo selecionado:
  * **MCTS**: simula múltiplas partidas para estimar o valor dos estados.
  * **Minimax com HEF**: busca em profundidade limitada, avaliando estados
    com base em padrões vantajosos ou ameaçadores.
- Exibe o tabuleiro atualizado a cada jogada.
- Detecta o término do jogo (vitória, derrota ou empate) e mostra o resultado
  colorido no terminal com auxílio da biblioteca `colorama`.

Funções principais:
- `evaluate_connect_four(board, player)`: calcula a pontuação heurística de um
  estado do tabuleiro com base em combinações de quatro posições.
- `best_move_minimax(game, depth)`: aplica o algoritmo Minimax com a função
  heurística para escolher a jogada mais promissora para a IA.
- `play(algorithm)`: executa o loop principal do jogo, recebendo o algoritmo
  ("mcts" ou "minimax") e coordenando as interações entre humano e IA.

Módulos utilizados:
- `board_connect_four.ConnectFour`: implementação do tabuleiro e regras do jogo.
- `mcts.mcts`: implementação do algoritmo de Monte Carlo Tree Search.
- `minimax.minimax_with_hef`: implementação do algoritmo Minimax com HEF.
- `helper_functions.print_board`: exibição do tabuleiro no console.
- `colorama`: realce visual no terminal.

Execução:
O programa é executado via linha de comando, especificando o algoritmo desejado:

$ python play_connect_four.py --algo mcts
$ python play_connect_four.py --algo minimax
"""

import time
import argparse
from colorama import Fore, Style, init
init(autoreset=True)

from board_connect_four import ConnectFour, COLS
from helper_functions import print_board

# Importações específicas
from mcts import mcts
from minimax import minimax_with_hef


# ========================
# Função heurística Minimax
# ========================
def evaluate_connect_four(board, player):
    opponent = 'O' if player == 'X' else 'X'
    score = 0

    def evaluate_window(window):
        if window.count(player) == 4:
            return 1000
        elif window.count(player) == 3 and window.count(' ') == 1:
            return 50
        elif window.count(player) == 2 and window.count(' ') == 2:
            return 10
        elif window.count(opponent) == 3 and window.count(' ') == 1:
            return -80
        return 0

    rows = len(board)
    cols = len(board[0])

    # horizontais
    for r in range(rows):
        for c in range(cols - 3):
            score += evaluate_window(board[r][c:c+4])
    # verticais
    for r in range(rows - 3):
        for c in range(cols):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window(window)
    # diagonais \
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window)
    # diagonais /
    for r in range(3, rows):
        for c in range(cols - 3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window)

    return score


def best_move_minimax(game, depth=4):
    player = game.current
    best_score = float('-inf')
    move_choice = None
    for move in game.available_moves():
        new_game = game.copy()
        new_game.make_move(move)
        score = minimax_with_hef(
            game=new_game,
            depth=depth - 1,
            maximizing=False,
            player=player,
            evaluate_fn=evaluate_connect_four
        )
        if score > best_score:
            best_score = score
            move_choice = move
    return move_choice


# ========================
# Função principal
# ========================
def play(algorithm: str):
    game = ConnectFour()
    human = input("Escolha seu lado (X ou O): ").strip().upper()
    assert human in ['X', 'O']
    ai = 'O' if human == 'X' else 'X'

    while not game.game_over():
        print_board(game.board, COLS)
        if game.current == human:
            while True:
                try:
                    col = int(input(f"Sua jogada ({human}), escolha coluna (0-{COLS-1}): "))
                    if col in game.available_moves():
                        game.make_move(col)
                        time.sleep(0.5)
                        break
                    else:
                        print("Coluna inválida ou cheia.")
                except ValueError:
                    print("Entrada inválida.")
        else:
            print("IA pensando...")
            if algorithm == "mcts":
                move = mcts(game, iterations=200)
            elif algorithm == "minimax":
                move = best_move_minimax(game, depth=4)
            else:
                raise ValueError(f"Algoritmo desconhecido: {algorithm}")
            print(f"IA joga na coluna {move}")
            game.make_move(move)
            time.sleep(0.8)

    print_board(game.board, COLS)
    winner = game.winner()
    if winner == human:
        print(Fore.GREEN + "Você venceu!")
    elif winner == ai:
        print(Fore.RED + "A IA venceu.")
    else:
        print(Fore.CYAN + "Empate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jogo Connect Four com IA usando MCTS ou Minimax."
    )
    parser.add_argument(
        "--algo", "-a",
        choices=["mcts", "minimax"],
        required=True,
        help="Algoritmo da IA a ser usado (mcts ou minimax)."
    )
    args = parser.parse_args()
    play(args.algo)
