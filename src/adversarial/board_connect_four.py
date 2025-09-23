from board_game import BoardGame
import numpy as np

ROWS, COLS = 6, 7

class ConnectFour(BoardGame):
    def __init__(self):
        super().__init__(6, 7)

    def available_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == ' ']

    def make_move(self, col):
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == ' ':
                self.board[r][col] = self.current
                self.current = 'O' if self.current == 'X' else 'X'
                return True
        return False

    def winner(self):
        b = self.board
        for r in range(self.rows):
            for c in range(self.cols-3):
                if b[r][c] != ' ' and all(b[r][c+i] == b[r][c] for i in range(4)):
                    return b[r][c]
        for r in range(self.rows-3):
            for c in range(self.cols):
                if b[r][c] != ' ' and all(b[r+i][c] == b[r][c] for i in range(4)):
                    return b[r][c]
        for r in range(self.rows-3):
            for c in range(self.cols-3):
                if b[r][c] != ' ' and all(b[r+i][c+i] == b[r][c] for i in range(4)):
                    return b[r][c]
        for r in range(3, self.rows):
            for c in range(self.cols-3):
                if b[r][c] != ' ' and all(b[r-i][c+i] == b[r][c] for i in range(4)):
                    return b[r][c]
        return None
