from typing import Optional


class BaseReader:
    def read_games(self, max_games: Optional[int] = None):
        raise NotImplementedError('Subclasses should implement this method')

    @staticmethod
    def _should_keep_reading(n_games: int, max_games: Optional[int]) -> bool:
        if max_games is None:
            return True
        return n_games < max_games
