from typing import List

import numpy as np

from src.encoders.color import ColorPerspective
from src.encoders.pieces import PiecesEncoder


class HistoryEncoder:
    def __init__(self, history_size: int):
        self.history_size = history_size

    def encode(self, past_fens: List[str], color: str) -> np.ndarray:
        perspective = ColorPerspective(color)
        encoded_past_boards = []
        for fen in past_fens:
            encoded_past_boards += list(PiecesEncoder().encode(fen, perspective))
        encoded_past_boards = np.array(encoded_past_boards)
        if len(past_fens) < self.history_size:
            missing_boards = self.history_size - len(past_fens)
            empty_history = np.zeros((missing_boards * 12, 8, 8), dtype=int)
            if len(past_fens) == 0:
                return empty_history
            return np.concatenate((empty_history, encoded_past_boards), axis=0)
        return encoded_past_boards

    def is_history_needed(self) -> bool:
        return self.history_size > 0
