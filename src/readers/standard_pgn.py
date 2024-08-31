from typing import List, Optional

import chess
from chess.pgn import Game

from src.readers.base import BaseReader
from src.readers.game_selector import GameSelector


class StandardPGNReader(BaseReader):
    def __init__(self, file_name: str, game_selector: GameSelector = GameSelector()):
        self.file = open(file_name, 'r')
        self.selector = game_selector

    def read_games(self, max_games: Optional[int]) -> List[Game]:
        games = []
        while self._should_keep_reading(len(games), max_games):
            game = chess.pgn.read_game(self.file)
            if game is None:
                break
            if self.selector.select_game(game):
                games.append(game)
        return games
