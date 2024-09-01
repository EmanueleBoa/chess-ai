from typing import List

import chess
import numpy as np

from src.encoders import BoardEncoder
from src.encoders.color import ColorPerspective
from src.encoders.history import HistoryEncoder
from src.encoders.mate import MateEncoder
from src.processors import FenProcessor


class LegalMovesEncoder:
    def __init__(self, history_size: int = 0):
        self.history_encoder = HistoryEncoder(history_size=history_size)

    def encode(self, fen: str, history: List[str]) -> np.ndarray:
        color = FenProcessor().get_color(fen)
        encoded_starting_board = BoardEncoder().encode(fen)
        if self.history_encoder.is_history_needed():
            encoded_history = self.history_encoder.encode(history, color)
            encoded_starting_board = np.concatenate((encoded_history, encoded_starting_board), axis=0)
        final_boards_fen = self._get_next_possible_board_fens(fen)
        encoded_moves = []
        for final_fen in final_boards_fen:
            encoded_final_board = BoardEncoder().encode(final_fen, color=color)
            encoded_mate_terminations = MateEncoder().encode(final_fen)
            encoded_move = np.concatenate(
                (encoded_starting_board, encoded_final_board, encoded_mate_terminations), axis=0)
            encoded_moves.append(encoded_move)
        return np.array(encoded_moves)

    @staticmethod
    def _get_next_possible_board_fens(starting_fen: str) -> List[str]:
        starting_board = chess.Board(fen=starting_fen)
        final_boards_fen = []
        for move in starting_board.legal_moves:
            final_board = starting_board.copy()
            final_board.push(move)
            final_boards_fen.append(final_board.fen())
        return final_boards_fen

    @staticmethod
    def _get_legal_moves_uci(fen: str) -> List[str]:
        board = chess.Board(fen=fen)
        return [move.uci() for move in board.legal_moves]
