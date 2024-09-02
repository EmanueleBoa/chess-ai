import chess
import numpy as np


class MatesEncoder:
    def encode(self, fen: str) -> np.ndarray:
        is_checkmate = self._encode_is_chekmate(fen)
        is_stalemate = self._encode_is_stalemate(fen)
        return np.concatenate((is_checkmate, is_stalemate), axis=0)

    @staticmethod
    def _encode_is_chekmate(fen: str) -> np.ndarray:
        is_chekmate = chess.Board(fen=fen).is_checkmate()
        return np.array([np.ones((8, 8), dtype=int) * is_chekmate])

    @staticmethod
    def _encode_is_stalemate(fen: str) -> np.ndarray:
        is_stalemate = chess.Board(fen=fen).is_stalemate()
        return np.array([np.ones((8, 8), dtype=int) * is_stalemate])
