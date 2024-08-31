import numpy as np

from src.encoders.color import ColorPerspective
from src.processors import FenProcessor


class PiecesEncoder:
    @staticmethod
    def encode(fen: str, perspective: ColorPerspective) -> np.ndarray:
        board_string = FenProcessor().get_board_string(fen)
        encoded_pieces = PiecesEncoder()._encode_pieces(board_string, perspective.pieces)
        encoded_opponent_pieces = PiecesEncoder()._encode_pieces(board_string, perspective.opponent_pieces)
        return np.concatenate((encoded_pieces, encoded_opponent_pieces), axis=0)

    def _encode_pieces(self, board: str, pieces: str) -> np.ndarray:
        return np.array([self._encode_piece(board, piece) for piece in pieces])

    @staticmethod
    def _encode_piece(board: str, piece: str) -> np.ndarray:
        rows = board.split('/')
        return np.array([[int(piece == square) for square in row] for row in rows], dtype=int)
