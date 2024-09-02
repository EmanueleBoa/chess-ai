from typing import List

import chess
import numpy as np


class TargetMoveEncoder:
    def encode(self, fen: str, played_move: str) -> np.ndarray:
        legal_moves = self._get_legal_moves_uci(fen)
        return np.array([int(move == played_move) for move in legal_moves])

    @staticmethod
    def _get_legal_moves_uci(fen: str) -> List[str]:
        board = chess.Board(fen=fen)
        return [move.uci() for move in board.legal_moves]
