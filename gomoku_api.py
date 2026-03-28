"""
Gomoku API — simple interface for getting AI move recommendations.

Usage:
    from gomoku_api import GomokuAPI

    api = GomokuAPI()
    move = api.get_move(board, color_to_play=1)
    top5 = api.get_top_moves(board, color_to_play=2, n=5)

    # Or use the module-level singleton
    from gomoku_api import get_move, get_top_moves

    move = get_move(board, color_to_play=1)
    top5 = get_top_moves(board, color_to_play=2, n=5)

Board format:
    19x19 numpy array or nested list.
    0 = empty, 1 = black, 2 = white.

Move format:
    (row, col) tuple, 0-indexed.
"""

import numpy as np
from gomoku_ai import GomokuAI, MODEL_PATH

BOARD_SIZE = 19


class GomokuAPI:
    """Simple API for getting AI move recommendations.

    Loads a pre-trained model and scores candidate moves using
    K-Means clustering + logistic regression on rich move features.
    """

    def __init__(self, model_path: str = str(MODEL_PATH)):
        self.ai = GomokuAI()
        self.ai.load(model_path)

    def get_move(self, board, color_to_play: int) -> tuple[int, int] | None:
        """Get the single best move.

        Args:
            board: 19x19 list or numpy array (0=empty, 1=black, 2=white).
            color_to_play: 1=black, 2=white.

        Returns:
            (row, col) or None if board is full.
        """
        board = _to_array(board)
        return self.ai.choose_move(board, color_to_play)

    def get_top_moves(self, board, color_to_play: int,
                      n: int = 5) -> list[tuple[tuple[int, int], float]]:
        """Get the top N moves ranked by win probability.

        Args:
            board: 19x19 list or numpy array (0=empty, 1=black, 2=white).
            color_to_play: 1=black, 2=white.
            n: Number of top moves to return.

        Returns:
            List of ((row, col), win_probability), sorted descending.
        """
        board = _to_array(board)
        return self.ai.score_moves(board, color_to_play, top_n=n)

    def score_position(self, board, color_to_play: int) -> dict:
        """Get full analysis of the position.

        Returns:
            dict with keys:
                - best_move: (row, col) or None
                - best_prob: float
                - top_moves: list of ((row, col), win_probability)
                - cluster: int (which position class the board belongs to)
                - empty_cells: int
        """
        board = _to_array(board)
        top = self.ai.score_moves(board, color_to_play, top_n=5)
        empty = int(np.sum(board == 0))

        flat = board.flatten().astype(float).reshape(1, -1)
        cluster_features = np.column_stack([flat, [[float(color_to_play)]]])
        cluster_id = int(self.ai.kmeans.predict(cluster_features)[0])

        return {
            "best_move": top[0][0] if top else None,
            "best_prob": top[0][1] if top else 0.0,
            "top_moves": top,
            "cluster": cluster_id,
            "empty_cells": empty,
        }


def _to_array(board) -> np.ndarray:
    """Convert board to numpy array if it isn't one already."""
    if not isinstance(board, np.ndarray):
        board = np.array(board, dtype=int)
    if board.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Board must be {BOARD_SIZE}x{BOARD_SIZE}, got {board.shape}")
    return board


# ─── Module-level singleton for quick one-liners ────────────────────────────

_instance: GomokuAPI | None = None


def _get_instance() -> GomokuAPI:
    global _instance
    if _instance is None:
        _instance = GomokuAPI()
    return _instance


def get_move(board, color_to_play: int) -> tuple[int, int] | None:
    """Get the best move. Loads model on first call."""
    return _get_instance().get_move(board, color_to_play)


def get_top_moves(board, color_to_play: int, n: int = 5) -> list[tuple[tuple[int, int], float]]:
    """Get top N moves. Loads model on first call."""
    return _get_instance().get_top_moves(board, color_to_play, n)


def score_position(board, color_to_play: int) -> dict:
    """Get full position analysis. Loads model on first call."""
    return _get_instance().score_position(board, color_to_play)
