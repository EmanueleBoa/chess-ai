import numpy as np

from src.encoders.castling import CastlingRightsEncoder
from src.encoders.color import ColorPerspective, ColorEncoder
from src.encoders.pieces import PiecesEncoder
from src.processors import FenProcessor


class BoardEncoder:
    def encode(self, fen: str, color: str = None) -> np.ndarray:
        color_ = FenProcessor().get_color(fen) if color is None else color
        perspective = ColorPerspective(color_)
        encoded_pieces = PiecesEncoder().encode(fen, perspective)
        encoded_castling_rights = CastlingRightsEncoder().encode(fen, perspective.castling_sides)
        encoded_opponent_castling_rights = CastlingRightsEncoder().encode(fen, perspective.opponent_castling_sides)
        encoded_color = ColorEncoder().encode(color_)
        encoded_board = np.concatenate(
            (encoded_pieces, encoded_castling_rights, encoded_opponent_castling_rights, encoded_color), axis=0)
        if perspective.should_flip_board:
            return self._flip_encoded_board(encoded_board)
        return encoded_board

    @staticmethod
    def _flip_encoded_board(encoded_board: np.ndarray) -> np.ndarray:
        return encoded_board[:, ::-1, ::-1]
