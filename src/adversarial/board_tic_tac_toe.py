from board_game import BoardGame

class TicTacToe(BoardGame):
    def __init__(self):
        super().__init__(3, 3)

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == ' ']

    def make_move(self, move):
        r, c = move
        if self.board[r][c] == ' ':
            self.board[r][c] = self.current
            self.current = 'O' if self.current == 'X' else 'X'
            return True
        return False

    def winner(self):
        lines = []
        # Rows, columns, diagonals
        lines.extend(self.board)
        lines.extend([[self.board[r][c] for r in range(3)] for c in range(3)])
        lines.append([self.board[i][i] for i in range(3)])
        lines.append([self.board[i][2-i] for i in range(3)])
        for line in lines:
            if line[0] != ' ' and all(cell == line[0] for cell in line):
                return line[0]
        return None
