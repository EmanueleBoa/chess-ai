import numpy as np

from src.processors import FenProcessor


class CastlingRightsEncoder:
    @staticmethod
    def encode(fen: str, castling_sides: str) -> np.ndarray:
        final_fen_part = FenProcessor().remove_board_from_fen(fen)
        return np.array([np.ones((8, 8), dtype=int) * (side in final_fen_part) for side in castling_sides])
