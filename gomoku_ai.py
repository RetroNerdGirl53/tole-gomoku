#!/usr/bin/env python3
"""
Gomoku AI — K-Means clustering + Logistic Regression

Pipeline (per ai-sketch.txt):
  1. K-Means clustering on board states → clustered classes of positions
  2. Logistic regression per cluster → classifies move quality (win/loss)
  3. At play time: read board → find cluster → score empty cells → pick best move

Usage:
    python gomoku_ai.py                          # train on data/move_sequences.csv
    python gomoku_ai.py --train                  # same as above
    python gomoku_ai.py --play                   # interactive play against the AI
    python gomoku_ai.py --play --color white     # play as white
    python gomoku_ai.py --data path/to/move_sequences.csv
    python gomoku_ai.py --clusters 8 --evaluate  # evaluate with different cluster count
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BOARD_SIZE = 19
BOARD_COLS = [f"b_{r}_{c}" for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
DEFAULT_DATA_PATH = "data/move_sequences.csv"
MODEL_PATH = "data/gomoku_ai.pkl"


class GomokuAI:
    """K-Means + Logistic Regression Gomoku AI.

    Clusters board positions into classes, then trains a logistic regression
    per cluster on move quality (win/loss) to select the best move.
    """

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.models: dict[int, LogisticRegression] = {}
        self.cluster_stats: dict[int, dict] = {}

    # ─── Training ────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """Train on a move_sequences DataFrame.

        Args:
            df: DataFrame with columns b_0_0..b_18_18, color_to_play, label_row,
                label_col, move_won.

        Returns:
            dict with training metrics.
        """
        # Extract board features
        X = df[BOARD_COLS].values.astype(float)

        # Normalize: always view from current player's perspective
        # Current player's stones → 1, opponent → 2 (keep as-is since color_to_play
        # is already encoded in the board state via the game flow)
        color = df["color_to_play"].values.astype(float)

        # Augment features: append color_to_play
        X_aug = np.column_stack([X, color])

        # Labels
        y = df["move_won"].values
        label_row = df["label_row"].values.astype(int)
        label_col = df["label_col"].values.astype(int)

        # Drop rows with missing labels
        valid = ~pd.isna(y)
        X_aug = X_aug[valid]
        y = y[valid].astype(int)
        label_row = label_row[valid]
        label_col = label_col[valid]

        print(f"Training on {len(X_aug)} samples, {X_aug.shape[1]} features")
        print(f"  Win rate: {y.mean():.3f}")

        # Step 1: K-Means clustering
        print(f"\nStep 1: K-Means clustering (k={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(X_aug)

        unique, counts = np.unique(clusters, return_counts=True)
        for c, n in zip(unique, counts):
            print(f"  Cluster {c}: {n} samples ({n / len(clusters) * 100:.1f}%)")

        # Step 2: Logistic regression per cluster
        print(f"\nStep 2: Training logistic regression per cluster...")
        self.models = {}
        self.cluster_stats = {}

        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            X_c = X_aug[mask]
            y_c = y[mask]

            win_rate = y_c.mean()
            self.cluster_stats[cluster_id] = {
                "size": int(mask.sum()),
                "win_rate": float(win_rate),
            }

            if len(np.unique(y_c)) < 2:
                # Only one class — can't train logistic regression
                print(f"  Cluster {cluster_id}: skipped (only one class, n={mask.sum()})")
                continue

            # Split for evaluation
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_c, y_c, test_size=0.2, random_state=self.random_state, stratify=y_c
            )

            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
            model.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, model.predict(X_te))
            self.models[cluster_id] = model

            print(f"  Cluster {cluster_id}: acc={acc:.3f}, win_rate={win_rate:.3f}, n={mask.sum()}")

        # Overall train accuracy
        preds = self._predict_clusters(X_aug, clusters)
        overall_acc = (preds == y).mean()

        metrics = {
            "n_samples": len(X_aug),
            "n_clusters": self.n_clusters,
            "clusters_trained": len(self.models),
            "overall_train_acc": float(overall_acc),
        }
        print(f"\nOverall train accuracy: {overall_acc:.3f}")
        return metrics

    def _predict_clusters(self, X_aug: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Predict win/loss for each sample using its cluster model."""
        preds = np.zeros(len(X_aug), dtype=int)
        for cluster_id, model in self.models.items():
            mask = clusters == cluster_id
            if mask.any():
                preds[mask] = model.predict(X_aug[mask])
        return preds

    # ─── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate on a test DataFrame. Returns metrics dict."""
        X = df[BOARD_COLS].values.astype(float)
        color = df["color_to_play"].values.astype(float)
        X_aug = np.column_stack([X, color])
        y = df["move_won"].values
        label_row = df["label_row"].values.astype(int)
        label_col = df["label_col"].values.astype(int)

        valid = ~pd.isna(y)
        X_aug = X_aug[valid]
        y = y[valid].astype(int)
        label_row = label_row[valid]
        label_col = label_col[valid]

        clusters = self.kmeans.predict(X_aug)
        preds = self._predict_clusters(X_aug, clusters)
        acc = (preds == y).mean()

        # Top-1 move accuracy: does the model's best move match the actual move?
        top1_hits = 0
        for i in range(len(X_aug)):
            board = X[i].reshape(BOARD_SIZE, BOARD_SIZE).astype(int)
            c = int(color[i])
            actual_r, actual_l = int(label_row[i]), int(label_col[i])
            best = self.choose_move(board, c)
            if best == (actual_r, actual_l):
                top1_hits += 1
        top1_acc = top1_hits / len(X_aug)

        metrics = {
            "win_loss_acc": float(acc),
            "top1_move_acc": float(top1_acc),
            "n_eval": len(X_aug),
        }
        print(f"Eval — Win/loss acc: {acc:.3f}, Top-1 move acc: {top1_acc:.3f} (n={len(X_aug)})")
        return metrics

    # ─── Inference ───────────────────────────────────────────────────────

    def choose_move(self, board: np.ndarray, color_to_play: int) -> tuple[int, int] | None:
        """Choose the best move for the given board state.

        Args:
            board: 19x19 numpy array (0=empty, 1=black, 2=white).
            color_to_play: 1=black, 2=white.

        Returns:
            (row, col) tuple or None if no valid move.
        """
        if self.kmeans is None or not self.models:
            raise RuntimeError("Model not trained. Call train() first.")

        empty = np.argwhere(board == 0)
        if len(empty) == 0:
            return None

        # Flatten board + color into feature vector
        flat = board.flatten().astype(float).reshape(1, -1)
        features = np.column_stack([flat, [[float(color_to_play)]]])

        # Find cluster
        cluster_id = int(self.kmeans.predict(features)[0])

        # If cluster has no model, fall back to first available
        if cluster_id not in self.models:
            available = sorted(self.models.keys())
            if not available:
                # No model at all — random
                idx = np.random.randint(len(empty))
                return (int(empty[idx][0]), int(empty[idx][1]))
            cluster_id = available[0]

        model = self.models[cluster_id]

        # Score each empty cell
        best_score = -np.inf
        best_pos = None

        for r, c in empty:
            trial = board.copy()
            trial[r, c] = color_to_play
            trial_flat = trial.flatten().astype(float).reshape(1, -1)
            trial_features = np.column_stack([trial_flat, [[float(color_to_play)]]])

            # Probability of winning
            prob = model.predict_proba(trial_features)[0]
            win_prob = prob[1] if len(prob) > 1 else prob[0]

            if win_prob > best_score:
                best_score = win_prob
                best_pos = (int(r), int(c))

        return best_pos

    # ─── Save / Load ─────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        state = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "kmeans": self.kmeans,
            "models": self.models,
            "cluster_stats": self.cluster_stats,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model saved to {path}")

    def load(self, path: str = MODEL_PATH):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.n_clusters = state["n_clusters"]
        self.random_state = state["random_state"]
        self.kmeans = state["kmeans"]
        self.models = state["models"]
        self.cluster_stats = state["cluster_stats"]
        print(f"Model loaded from {path} ({len(self.models)} cluster models)")


# ─── Interactive play ───────────────────────────────────────────────────────

def print_board(board: np.ndarray):
    symbols = {0: ".", 1: "X", 2: "O"}
    header = "   " + " ".join(f"{c:>2}" for c in range(BOARD_SIZE))
    print(header)
    for r in range(BOARD_SIZE):
        row_str = f"{r:>2} " + " ".join(f"{symbols[board[r, c]]:>2}" for c in range(BOARD_SIZE))
        print(row_str)
    print()


def play_interactive(ai: GomokuAI, human_color: int = 1):
    """Play against the AI in the terminal."""
    ai_color = 2 if human_color == 1 else 1
    color_names = {1: "black (X)", 2: "white (O)"}

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    current = 1  # black goes first

    print(f"You are {color_names[human_color]}")
    print(f"AI is {color_names[ai_color]}")
    print("Enter moves as: row col (e.g. '9 9' for center)")
    print()

    while True:
        print_board(board)

        if current == human_color:
            while True:
                try:
                    raw = input(f"Your move ({color_names[current]}): ").strip()
                    if raw.lower() in ("quit", "exit", "q"):
                        return
                    parts = raw.split()
                    r, c = int(parts[0]), int(parts[1])
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0:
                        break
                    print("Invalid move. Try again.")
                except (ValueError, IndexError):
                    print("Enter as: row col (e.g. '9 9')")
            board[r, c] = current
        else:
            print("AI thinking...")
            move = ai.choose_move(board, current)
            if move is None:
                print("AI has no moves!")
                return
            r, c = move
            board[r, c] = current
            print(f"AI plays: {r} {c}")

        # Check for win
        if _check_win(board, r, c, current):
            print_board(board)
            winner = "You" if current == human_color else "AI"
            print(f"{winner} win!")
            return

        # Check for draw
        if np.all(board != 0):
            print_board(board)
            print("Draw!")
            return

        current = 2 if current == 1 else 1


def _check_win(board: np.ndarray, row: int, col: int, color: int) -> bool:
    """Check if placing at (row, col) made 5 in a row."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
                count += 1
                r += dr * sign
                c += dc * sign
        if count >= 5:
            return True
    return False


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gomoku AI — K-Means + Logistic Regression")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to move_sequences.csv")
    parser.add_argument("--model", default=MODEL_PATH, help="Model save/load path")
    parser.add_argument("--clusters", type=int, default=6, help="Number of K-Means clusters")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--play", action="store_true", help="Play against the AI")
    parser.add_argument("--color", choices=["black", "white"], default="black",
                        help="Your color when playing (default: black)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on training data")
    parser.add_argument("--no-save", action="store_true", help="Don't save model after training")

    args = parser.parse_args()

    # Default: train if no mode specified
    if not args.play and not args.evaluate:
        args.train = True

    ai = GomokuAI(n_clusters=args.clusters)

    if args.train:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: {data_path} not found. Run gomoku_scraper.py first.")
            sys.exit(1)

        print(f"Loading {data_path}...")
        df = pd.read_csv(data_path)
        print(f"  {len(df)} rows, {df['game_id'].nunique()} games")

        ai.train(df)

        if not args.no_save:
            ai.save(args.model)

    if args.evaluate:
        if ai.kmeans is None:
            ai.load(args.model)
        data_path = Path(args.data)
        df = pd.read_csv(data_path)
        ai.evaluate(df)

    if args.play:
        if ai.kmeans is None:
            ai.load(args.model)
        human_color = 1 if args.color == "black" else 2
        play_interactive(ai, human_color)


if __name__ == "__main__":
    main()
