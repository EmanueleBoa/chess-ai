from typing import List

from chess.pgn import Game

from src.processors import FenProcessor


class GameProcessor:
    def process_games(self, games: List[Game]) -> List[dict]:
        states = []
        for game in games:
            states += self.process_game(game)
        return states

    def process_game(self, game: Game) -> List[dict]:
        game_id = self._get_game_id(game)
        result = self._get_game_result(game)
        white_player = self._get_white_user_name(game)
        black_player = self._get_black_user_name(game)
        mainline = game.mainline()
        states = []
        for node in mainline:
            fen = node.parent.board().fen()
            ply = node.ply()
            clock = node.clock()
            player = self._get_player_to_move(fen, white_player, black_player)
            move_played = node.move.uci()
            state = self._build_state_dict(game_id, fen, ply, clock, player, move_played, result)
            states.append(state)
        return states

    @staticmethod
    def _get_game_id(game: Game) -> str:
        return game.headers.get("Site", "").split('/')[-1]

    @staticmethod
    def _get_game_result(game: Game) -> str:
        return game.headers.get("Result")

    @staticmethod
    def _get_white_user_name(game: Game) -> str:
        return game.headers.get("White")

    @staticmethod
    def _get_black_user_name(game: Game) -> str:
        return game.headers.get("Black")

    @staticmethod
    def _get_player_to_move(fen: str, white_player: str, black_player: str) -> str:
        color = FenProcessor().get_color(fen)
        if color == 'w':
            return white_player
        return black_player

    @staticmethod
    def _build_state_dict(game_id: str, fen: str, ply: int, clock: float, player: str, move_played: str,
                          result: str) -> dict:
        return {
            'game_id': game_id,
            'fen': fen,
            'ply': ply,
            'clock': clock,
            'player': player,
            'move_played': move_played,
            'result': result
        }
