"""
Módulo de funções auxiliares para exibição e interação com o jogo Connect Four.

Funções disponíveis:
- `clear_screen()`: limpa a tela do console, de forma compatível com Windows
  e sistemas Unix (Linux/Mac). Usada para atualizar o tabuleiro a cada jogada.
- `colorize(cell)`: aplica cores aos símbolos do jogo usando a biblioteca
  `colorama`. A letra 'X' é exibida em vermelho, 'O' em amarelo e células
  vazias permanecem em branco.
- `print_board(board, cols)`: imprime o tabuleiro atual no console,
  com células coloridas e índices das colunas na parte inferior. A função
  chama `clear_screen()` antes de exibir, garantindo que apenas o estado
  mais recente do jogo apareça na tela.

Uso principal:
Esse módulo serve como suporte visual para os programas de Connect Four
(baseados em Minimax ou MCTS), tornando a experiência interativa mais clara
e intuitiva para o usuário que joga via terminal.
"""

import os
from colorama import Fore, Style, init
init(autoreset=True)

# Helper: clean screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Helper: color cell content
def colorize(cell):
    if cell == 'X':
        return Fore.RED + 'X' + Style.RESET_ALL
    elif cell == 'O':
        return Fore.YELLOW + 'O' + Style.RESET_ALL
    else:
        return ' '

def print_board(board, cols):
    clear_screen()
    for row in board:
        print('|' + '|'.join(colorize(cell) for cell in row) + '|')
    print(' ' + ' '.join(map(str, range(cols))))
