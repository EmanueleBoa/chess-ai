from typing import Optional

from chess.pgn import Headers, Game


class GameSelector:
    def __init__(self, time_format: Optional[str], target_elo: Optional[int],
                 elo_range: Optional[int] = 100, minimum_moves: Optional[int] = 0):
        self.time_format = time_format
        self.target_elo = target_elo
        self.elo_range = elo_range
        self.minimum_moves = minimum_moves

    def select_game(self, game: Game) -> bool:
        headers = game.headers
        return self.is_right_time_format(headers) and self.are_both_players_right_elo_range(
            headers) and self.is_long_enough(game)

    def are_both_players_right_elo_range(self, headers: Headers) -> bool:
        if self.target_elo is None:
            return True
        white_elo = int(headers.get("WhiteElo", -1))
        black_elo = int(headers.get("BlackElo", -1))
        return self.is_right_elo_range(white_elo) and self.is_right_elo_range(black_elo)

    def is_right_time_format(self, headers: Headers) -> bool:
        if self.time_format is None:
            return True
        return self.time_format in headers.get("Event")

    def is_right_elo_range(self, elo: int) -> bool:
        return self.target_elo <= elo < self.target_elo + self.elo_range

    def is_long_enough(self, game: Game):
        return len(list(game.mainline())) >= self.minimum_moves
