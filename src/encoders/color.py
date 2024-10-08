import numpy as np

PIECES = 'PNBRQK'
CASTLING_SIDES = 'KQ'
WHITE = 'w'


class ColorEncoder:
    @staticmethod
    def encode(color: str) -> np.ndarray:
        return np.ones((1, 8, 8), dtype=int) * (color == WHITE)


class ColorPerspective:
    def __init__(self, color: str):
        if color == WHITE:
            self.pieces = PIECES
            self.castling_sides = CASTLING_SIDES
            self.opponent_pieces = PIECES.lower()
            self.opponent_castling_sides = CASTLING_SIDES.lower()
            self.should_flip_board = False
        else:
            self.pieces = PIECES.lower()
            self.castling_sides = CASTLING_SIDES.lower()
            self.opponent_pieces = PIECES
            self.opponent_castling_sides = CASTLING_SIDES
            self.should_flip_board = True
