import io
from typing import List, Optional

import chess
from chess.pgn import Game
import zstandard

from src.readers.base import BaseReader
from src.readers.game_selector import GameSelector


class CompressedPGNReader(BaseReader):
    def __init__(self, file_name: str, game_selector: GameSelector, chunk_size: Optional[int] = 16384):
        self.file = open(file_name, 'rb')
        self.selector = game_selector
        self.decompressor = zstandard.ZstdDecompressor()
        self.reader = self.decompressor.stream_reader(self.file)
        self.chunk_size = chunk_size
        self.string_leftovers = ''
        self.chunk = None

    def read_games(self, max_games: Optional[int]) -> List[Game]:
        games = []
        while self._should_keep_reading(len(games), max_games):
            self.chunk = self.reader.read(self.chunk_size)
            if self.chunk is None:
                break
            string = self.string_leftovers + self.chunk.decode('UTF-8')
            pgn = io.StringIO(string)
            offsets = []
            while True:
                offset = pgn.tell()
                headers = chess.pgn.read_headers(pgn)
                if headers is None:
                    break
                offsets.append(offset)
            for offset in offsets[:-1]:
                pgn.seek(offset)
                game = chess.pgn.read_game(pgn)
                if self.selector.select_game(game):
                    games.append(game)
            self.string_leftovers = string[offsets[-1]:]
        return games
