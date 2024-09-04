from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.encoders import LegalMovesEncoder, TargetMoveEncoder


class DataHandler:
    def __init__(self, df_states: pd.DataFrame, df_games: pd.DataFrame, history_size: int):
        self.df_states = df_states
        self.df_games = df_games
        self.history_size = history_size

    def get_encoded_games_batch(self, n_games: int, target_player: Optional[str] = None,
                                min_clock_seconds: Optional[float] = None,
                                random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        df_batch = self._get_games_batch(n_games, target_player, min_clock_seconds, random_state)
        input_encoder = LegalMovesEncoder(history_size=self.history_size)
        output_encoder = TargetMoveEncoder()
        X = []
        y = []
        for _, row in df_batch.iterrows():
            x = input_encoder.encode(row.fen, row.history)
            target = output_encoder.encode(row.fen, row.move_played)
            X += list(x)
            y += list(target)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def _get_games_batch(self, n_games: int, target_player: Optional[str] = None,
                         min_clock_seconds: Optional[float] = None,
                         random_state: Optional[int] = None) -> pd.DataFrame:
        games = self.df_games.game_id.sample(n=n_games, random_state=random_state)
        df_batch = self.df_states[self.df_states.game_id.isin(games)].copy()
        if target_player is not None:
            df_batch = df_batch[df_batch.player == target_player].copy()
        if min_clock_seconds is not None:
            df_batch = df_batch[df_batch.clock >= min_clock_seconds].copy()
        df_batch['history'] = df_batch.apply(
            lambda row: self._get_state_history(row.game_id, row.ply), axis=1)
        return df_batch

    def _get_state_history(self, game_id: str, ply: int) -> List:
        history = self.df_games.loc[self.df_games.game_id == game_id]['fen'].to_list()[0]
        return history[max(ply - 1 - self.history_size, 0):ply - 1]
