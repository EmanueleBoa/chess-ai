from typing import List

import numpy as np

from src.encoders import LegalMovesEncoder, TargetMoveEncoder
from src.models import ResNet


class MovePredictor:
    def __init__(self, model: ResNet, encoder: LegalMovesEncoder):
        self.encoder = encoder
        self.model = model
        self.model.eval()

    def search(self, fen: str, history: List[str]) -> str:
        probabilities = self._moves_probabilities(fen, history)
        move_names = TargetMoveEncoder().get_legal_moves_uci(fen)
        move_index = np.argmax(probabilities)
        return move_names[move_index]

    def evaluate_legal_moves(self, fen: str, history: List[str]) -> dict:
        probabilities = self._moves_probabilities(fen, history)
        move_names = TargetMoveEncoder().get_legal_moves_uci(fen)
        return dict(zip(move_names, probabilities))

    def _moves_probabilities(self, fen: str, history: List[str]) -> np.ndarray:
        encoded_moves = self.encoder.encode(fen, history)
        probabilities = self.model.predict(encoded_moves)
        probabilities /= probabilities.sum()
        return probabilities
