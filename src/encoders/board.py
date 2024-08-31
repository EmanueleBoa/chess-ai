import re
from typing import List

import chess
import numpy as np

from src.encoders.common import EMPTY_SQUARE, WHITE, ColorPerspective


class BoardEncoder:
    def encode_board(self, fen: str, color_perspective: str = None) -> np.ndarray:
        color = self.get_color(fen) if color_perspective is None else color_perspective
        perspective = ColorPerspective(color)
        encoded_board_pieces = self.encode_board_pieces(fen, perspective)
        encoded_castling_rights = self._encode_castling_rights(fen, perspective.castling_sides)
        encoded_opponent_castling_rights = self._encode_castling_rights(fen, perspective.opponent_castling_sides)
        encoded_color = [self._encode_color(color)]
        encoded_state = np.array(
            encoded_board_pieces + encoded_castling_rights +
            encoded_opponent_castling_rights + encoded_color)
        if perspective.should_flip_board:
            return self._flip_encoded_board(encoded_state)
        return encoded_state

    def encode_board_pieces(self, fen: str, perspective: ColorPerspective) -> List[np.ndarray]:
        board_string = self._get_board_string(fen)
        encoded_pieces = self._encode_pieces(board_string, perspective.pieces)
        encoded_opponent_pieces = self._encode_pieces(board_string, perspective.opponent_pieces)
        return encoded_pieces + encoded_opponent_pieces

    @staticmethod
    def encode_is_chekmate(fen: str) -> np.ndarray:
        is_chekmate = chess.Board(fen=fen).is_checkmate()
        return np.ones((8, 8), dtype=int) * is_chekmate

    @staticmethod
    def encode_is_stalemate(fen: str) -> np.ndarray:
        is_stalemate = chess.Board(fen=fen).is_stalemate()
        return np.ones((8, 8), dtype=int) * is_stalemate

    @staticmethod
    def get_color(fen: str) -> str:
        return fen.split(' ')[1]

    @staticmethod
    def _get_board_string(fen: str) -> str:
        board_fen = fen.split(' ')[0]
        numbers = list(set(re.findall(r'\d+', board_fen)))
        replacements = [EMPTY_SQUARE * int(n) for n in numbers]
        processed_board = board_fen
        for number, replacement in zip(numbers, replacements):
            processed_board = processed_board.replace(number, replacement)
        return processed_board

    @staticmethod
    def _encode_piece(board: str, piece: str) -> np.ndarray:
        rows = board.split('/')
        return np.array([[int(piece == square) for square in row] for row in rows], dtype=int)

    def _encode_pieces(self, board: str, pieces: str) -> List[np.ndarray]:
        return [self._encode_piece(board, piece) for piece in pieces]

    @staticmethod
    def _encode_castling_rights(fen: str, castling_sides: str) -> List[np.ndarray]:
        final_fen_part = ''.join(fen.split(' ')[2:])
        return [np.ones((8, 8), dtype=int) * (side in final_fen_part) for side in castling_sides]

    @staticmethod
    def _encode_color(color: str) -> np.ndarray:
        return np.ones((8, 8), dtype=int) * (color == WHITE)

    @staticmethod
    def _flip_encoded_board(encoded_board: np.ndarray) -> np.ndarray:
        return encoded_board[:, ::-1, ::-1]
