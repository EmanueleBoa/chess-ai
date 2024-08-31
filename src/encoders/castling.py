import numpy as np


class CastlingRightsEncoder:
    @staticmethod
    def encode(fen: str, castling_sides: str) -> np.ndarray:
        final_fen_part = ''.join(fen.split(' ')[2:])
        return np.array([np.ones((8, 8), dtype=int) * (side in final_fen_part) for side in castling_sides])
