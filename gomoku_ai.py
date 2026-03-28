#!/usr/bin/env python3
"""
Gomoku AI — K-Means clustering + Logistic Regression

Pipeline (per ai-sketch.txt):
  1. K-Means clustering on board states → clustered classes of positions
  2. Logistic regression per cluster → classifies move types (win/loss)
     using rich move-specific features (line counts, neighbor density, etc.)
  3. At play time: read board → find cluster → score empty cells → return ranked moves

Usage:
    python gomoku_ai.py                          # train on auto-detected scraper data
    python gomoku_ai.py --train                  # same as above
    python gomoku_ai.py --kfolds 5               # train with 5-fold cross-validation
    python gomoku_ai.py --update                 # retrain on new data only
    python gomoku_ai.py --play                   # interactive play against the AI
    python gomoku_ai.py --play --color white     # play as white
    python gomoku_ai.py --data path/to/move_sequences.csv
    python gomoku_ai.py --clusters 8 --evaluate  # evaluate with different cluster count
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

BOARD_SIZE = 19
BOARD_COLS = [f"b_{r}_{c}" for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
CENTER = BOARD_SIZE // 2
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "move_sequences.csv"
MODEL_PATH = DEFAULT_DATA_DIR / "gomoku_ai.pkl"
SEEN_IDS_PATH = DEFAULT_DATA_DIR / "seen_game_ids.json"

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def find_scraper_data(data_dir: Path = DEFAULT_DATA_DIR) -> Path | None:
    """Auto-detect scraper move_sequences.csv in the data directory."""
    for name in ("move_sequences.csv", "move_sequences.parquet"):
        p = data_dir / name
        if p.exists():
            return p
    return None


def load_seen_ids(path: Path = SEEN_IDS_PATH) -> set:
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(ids: set, path: Path = SEEN_IDS_PATH):
    with open(path, "w") as f:
        json.dump(sorted(ids), f)


# ─── Feature engineering ────────────────────────────────────────────────────

def compute_move_features(board: np.ndarray, row: int, col: int, color: int) -> np.ndarray:
    """Compute rich feature vector for placing `color` at (row, col).

    Returns a fixed-length vector describing the move's strategic context:
    line potential, neighbor density, center proximity, blocking value.

    This is what the logistic regression trains on — features that directly
    encode "what does this move do?" rather than raw board cells.
    """
    opp = 2 if color == 1 else 1

    # Directional line analysis: how many in a row + is the end open?
    dir_counts = np.zeros(4, dtype=float)
    dir_open = np.zeros(4, dtype=float)

    for i, (dr, dc) in enumerate(DIRECTIONS):
        # Count same-color stones continuing from (row, col) in +direction
        count_plus = 0
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            count_plus += 1
            r += dr
            c += dc
        open_plus = 1.0 if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0) else 0.0

        # Count same-color stones continuing in -direction
        count_minus = 0
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            count_minus += 1
            r -= dr
            c -= dc
        open_minus = 1.0 if (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0) else 0.0

        dir_counts[i] = count_plus + count_minus
        dir_open[i] = open_plus + open_minus

    # Neighbor density (3x3 and 5x5 windows around the move)
    friendly_near = 0
    opponent_near = 0
    friendly_far = 0
    opponent_far = 0

    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = row + dr, col + dc
            if (r == row and c == col) or r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                continue
            if board[r, c] == color:
                if abs(dr) <= 1 and abs(dc) <= 1:
                    friendly_near += 1
                friendly_far += 1
            elif board[r, c] == opp:
                if abs(dr) <= 1 and abs(dc) <= 1:
                    opponent_near += 1
                opponent_far += 1

    # Distance from center (normalized)
    dist_center = np.sqrt((row - CENTER) ** 2 + (col - CENTER) ** 2) / (CENTER * np.sqrt(2))

    # Board sparsity (fraction of empty cells)
    empty_frac = np.sum(board == 0) / (BOARD_SIZE * BOARD_SIZE)

    # Is this cell near an existing stone at all?
    has_any_neighbor = 1.0 if (friendly_near + opponent_near) > 0 else 0.0

    return np.concatenate([
        dir_counts,                     # 4: total stones in each direction
        dir_open,                       # 4: open ends in each direction (0, 1, or 2)
        [friendly_near, opponent_near], # 2: 3x3 neighborhood
        [friendly_far, opponent_far],   # 2: 5x5 neighborhood
        [dist_center],                  # 1: center proximity
        [empty_frac],                   # 1: game phase
        [has_any_neighbor],             # 1: adjacency to existing play
    ])  # total: 15 features


MOVE_FEATURE_NAMES = [
    "dir_h_count", "dir_v_count", "dir_d1_count", "dir_d2_count",
    "dir_h_open", "dir_v_open", "dir_d1_open", "dir_d2_open",
    "friendly_near", "opponent_near",
    "friendly_far", "opponent_far",
    "dist_center", "empty_frac", "has_neighbor",
]
N_MOVE_FEATURES = len(MOVE_FEATURE_NAMES)


class GomokuAI:
    """K-Means + Logistic Regression Gomoku AI.

    K-Means clusters board positions into classes of similar game states.
    Logistic regression per cluster scores candidate moves by their
    likelihood of leading to a win, using rich move-specific features.
    """

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.models: dict[int, LogisticRegression] = {}
        self.cluster_stats: dict[int, dict] = {}
        self.seen_game_ids: set = set()

    # ─── Data preparation ────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame):
        """Extract features from DataFrame.

        Returns:
            board_features: (N, 362) raw board + color for clustering
            move_features:  (N, 15) rich move features for LR
            y:              (N,) win/loss labels
            game_ids:       (N,) game_id per sample
        """
        boards = df[BOARD_COLS].values.astype(float)
        color = df["color_to_play"].values.astype(float)
        label_row = df["label_row"].values.astype(int)
        label_col = df["label_col"].values.astype(int)
        y = df["move_won"].values
        game_ids = df["game_id"].values

        valid = ~pd.isna(y)
        boards = boards[valid]
        color = color[valid]
        label_row = label_row[valid]
        label_col = label_col[valid]
        y = y[valid].astype(int)
        game_ids = game_ids[valid]

        # Board features for clustering
        board_features = np.column_stack([boards, color])

        # Rich move features for logistic regression
        move_features = np.empty((len(boards), N_MOVE_FEATURES), dtype=float)
        for i in range(len(boards)):
            board = boards[i].reshape(BOARD_SIZE, BOARD_SIZE).astype(int)
            r, c = int(label_row[i]), int(label_col[i])
            move_features[i] = compute_move_features(board, r, c, int(color[i]))

        return board_features, move_features, y, game_ids, boards, label_row, label_col

    # ─── Training ────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, kfolds: int = 0) -> dict:
        """Train on a move_sequences DataFrame.

        K-Means clusters on raw board states. Logistic regression per cluster
        trains on rich move features to predict win/loss.

        Args:
            df: DataFrame with board columns, color_to_play, label_row/col, move_won, game_id.
            kfolds: If > 0, run k-fold cross-validation first.

        Returns:
            dict with training metrics.
        """
        board_feat, move_feat, y, game_ids, raw_boards, label_row, label_col = self._prepare(df)

        print(f"Training on {len(board_feat)} samples")
        print(f"  Games: {len(np.unique(game_ids))}")
        print(f"  Win rate: {y.mean():.3f}")
        print(f"  Move features: {N_MOVE_FEATURES} (line counts, neighbor density, center dist, etc.)")

        self.seen_game_ids = set(np.unique(game_ids).tolist())

        # K-fold cross-validation
        if kfolds > 1:
            self._kfold_eval(board_feat, move_feat, y, kfolds)

        # Step 1: K-Means clustering on raw board states
        print(f"\nStep 1: K-Means clustering (k={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(board_feat)

        unique, counts = np.unique(clusters, return_counts=True)
        for c, n in zip(unique, counts):
            print(f"  Cluster {c}: {n} samples ({n / len(clusters) * 100:.1f}%)")

        # Step 2: Logistic regression per cluster on move features
        print(f"\nStep 2: Training logistic regression per cluster (on {N_MOVE_FEATURES} move features)...")
        self.models = {}
        self.cluster_stats = {}

        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            X_c = move_feat[mask]
            y_c = y[mask]

            win_rate = y_c.mean()
            self.cluster_stats[cluster_id] = {
                "size": int(mask.sum()),
                "win_rate": float(win_rate),
            }

            if len(np.unique(y_c)) < 2:
                print(f"  Cluster {cluster_id}: skipped (only one class, n={mask.sum()})")
                continue

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

            # Show which features matter most
            coefs = model.coef_[0]
            top_idx = np.argsort(np.abs(coefs))[-3:]
            top_feats = [(MOVE_FEATURE_NAMES[j], coefs[j]) for j in top_idx]
            top_str = ", ".join(f"{n}={v:+.2f}" for n, v in top_feats)
            print(f"  Cluster {cluster_id}: acc={acc:.3f}, win_rate={win_rate:.3f}, n={mask.sum()}")
            print(f"    top features: {top_str}")

        # Overall train accuracy (using move features)
        preds = np.zeros(len(y), dtype=int)
        for cid, model in self.models.items():
            cmask = clusters == cid
            if cmask.any():
                preds[cmask] = model.predict(move_feat[cmask])
        overall_acc = (preds == y).mean()

        metrics = {
            "n_samples": len(board_feat),
            "n_games": len(np.unique(game_ids)),
            "n_clusters": self.n_clusters,
            "clusters_trained": len(self.models),
            "overall_train_acc": float(overall_acc),
        }
        print(f"\nOverall train accuracy: {overall_acc:.3f}")
        return metrics

    def _kfold_eval(self, board_feat, move_feat, y, kfolds: int):
        """Run k-fold cross-validation on move features."""
        print(f"\nK-Fold Cross-Validation (k={kfolds})...")
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=self.random_state)

        fold_accs = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(board_feat)):
            b_tr, b_te = board_feat[train_idx], board_feat[test_idx]
            m_tr, m_te = move_feat[train_idx], move_feat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            tr_clusters = km.fit_predict(b_tr)
            te_clusters = km.predict(b_te)

            preds = np.full(len(y_te), -1, dtype=int)
            n_trained = 0

            for cid in range(self.n_clusters):
                tr_mask = tr_clusters == cid
                te_mask = te_clusters == cid
                if tr_mask.sum() < 10 or len(np.unique(y_tr[tr_mask])) < 2:
                    continue
                model = LogisticRegression(
                    max_iter=1000, random_state=self.random_state, class_weight="balanced"
                )
                model.fit(m_tr[tr_mask], y_tr[tr_mask])
                if te_mask.any():
                    preds[te_mask] = model.predict(m_te[te_mask])
                n_trained += 1

            eval_mask = preds >= 0
            acc = (preds[eval_mask] == y_te[eval_mask]).mean() if eval_mask.any() else 0.0
            fold_accs.append(acc)
            print(f"  Fold {fold_idx + 1}: acc={acc:.3f} ({n_trained} clusters)")

        print(f"  Mean: {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}\n")

    # ─── Incremental update ──────────────────────────────────────────────

    def update(self, df: pd.DataFrame, kfolds: int = 0) -> dict:
        """Retrain on new data only (games not yet seen), then retrain full model."""
        all_ids = set(df["game_id"].unique())
        new_ids = all_ids - self.seen_game_ids

        if not new_ids:
            print("No new games found. Model is up to date.")
            return {"new_games": 0}

        print(f"Found {len(new_ids)} new games ({len(all_ids)} total, {len(self.seen_game_ids)} previously seen)")
        print("Retraining on full dataset...")
        return self.train(df, kfolds=kfolds)

    # ─── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate on a DataFrame."""
        board_feat, move_feat, y, game_ids, raw_boards, label_row, label_col = self._prepare(df)

        clusters = self.kmeans.predict(board_feat)
        preds = np.zeros(len(y), dtype=int)
        for cid, model in self.models.items():
            cmask = clusters == cid
            if cmask.any():
                preds[cmask] = model.predict(move_feat[cmask])
        acc = (preds == y).mean()

        # Top-1 move accuracy
        print("Computing top-1 move accuracy...")
        top1_hits = 0
        for i in range(len(raw_boards)):
            board = raw_boards[i].reshape(BOARD_SIZE, BOARD_SIZE).astype(int)
            c = int(board_feat[i, -1])
            actual_r, actual_l = int(label_row[i]), int(label_col[i])
            scored = self.score_moves(board, c, top_n=1)
            if scored and scored[0][0] == (actual_r, actual_l):
                top1_hits += 1
        top1_acc = top1_hits / len(raw_boards)

        metrics = {"win_loss_acc": float(acc), "top1_move_acc": float(top1_acc), "n_eval": len(y)}
        print(f"Eval — Win/loss acc: {acc:.3f}, Top-1 move acc: {top1_acc:.3f} (n={len(y)})")
        return metrics

    # ─── Inference ───────────────────────────────────────────────────────

    def score_moves(self, board: np.ndarray, color_to_play: int,
                    top_n: int | None = None) -> list[tuple[tuple[int, int], float]]:
        """Score all empty cells and return ranked list of (position, win_probability).

        This is the core of the AI: for each candidate move, compute rich features
        and ask the logistic regression "how likely is this to lead to a win?"

        Args:
            board: 19x19 numpy array (0=empty, 1=black, 2=white).
            color_to_play: 1=black, 2=white.
            top_n: If set, return only the top N moves.

        Returns:
            List of ((row, col), win_probability) sorted descending by probability.
        """
        if self.kmeans is None or not self.models:
            raise RuntimeError("Model not trained. Call train() first.")

        empty = np.argwhere(board == 0)
        if len(empty) == 0:
            return []

        # Find which cluster this board belongs to
        flat = board.flatten().astype(float).reshape(1, -1)
        cluster_features = np.column_stack([flat, [[float(color_to_play)]]])
        cluster_id = int(self.kmeans.predict(cluster_features)[0])

        if cluster_id not in self.models:
            available = sorted(self.models.keys())
            if not available:
                return [((int(r), int(c)), 0.5) for r, c in empty]
            cluster_id = available[0]

        model = self.models[cluster_id]

        # Score every empty cell using rich move features
        scores = []
        for r, c in empty:
            feats = compute_move_features(board, int(r), int(c), color_to_play).reshape(1, -1)
            prob = model.predict_proba(feats)[0]
            win_prob = prob[1] if len(prob) > 1 else prob[0]
            scores.append(((int(r), int(c)), float(win_prob)))

        scores.sort(key=lambda x: x[1], reverse=True)
        if top_n:
            scores = scores[:top_n]
        return scores

    def choose_move(self, board: np.ndarray, color_to_play: int) -> tuple[int, int] | None:
        """Choose the single best move."""
        scored = self.score_moves(board, color_to_play, top_n=1)
        return scored[0][0] if scored else None

    # ─── Save / Load ─────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        state = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "kmeans": self.kmeans,
            "models": self.models,
            "cluster_stats": self.cluster_stats,
            "seen_game_ids": sorted(self.seen_game_ids),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        save_seen_ids(self.seen_game_ids)
        print(f"Model saved to {path} ({len(self.seen_game_ids)} games tracked)")

    def load(self, path: str = MODEL_PATH):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.n_clusters = state["n_clusters"]
        self.random_state = state["random_state"]
        self.kmeans = state["kmeans"]
        self.models = state["models"]
        self.cluster_stats = state["cluster_stats"]
        self.seen_game_ids = set(state.get("seen_game_ids", []))
        print(f"Model loaded from {path} ({len(self.models)} cluster models, {len(self.seen_game_ids)} games)")


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
    current = 1

    print(f"You are {color_names[human_color]}")
    print(f"AI is {color_names[ai_color]}")
    print("Enter moves as: row col (e.g. '9 9' for center)")
    print("Type 'top3' to see AI's top 3 moves for the current position")
    print()

    while True:
        print_board(board)

        if current == human_color:
            while True:
                try:
                    raw = input(f"Your move ({color_names[current]}): ").strip()
                    if raw.lower() in ("quit", "exit", "q"):
                        return
                    if raw.lower() == "top3":
                        scored = ai.score_moves(board, current, top_n=3)
                        for (r, c), prob in scored:
                            print(f"  ({r:>2}, {c:>2})  win_prob={prob:.3f}")
                        continue
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
            scored = ai.score_moves(board, current, top_n=5)
            if not scored:
                print("AI has no moves!")
                return
            (r, c), prob = scored[0]
            board[r, c] = current
            print(f"AI plays: {r} {c}  (win_prob={prob:.3f})")
            if len(scored) > 1:
                alts = ", ".join(f"({r2},{c2})={p2:.3f}" for (r2, c2), p2 in scored[1:3])
                print(f"  Runner-up: {alts}")

        if _check_win(board, r, c, current):
            print_board(board)
            winner = "You" if current == human_color else "AI"
            print(f"{winner} win!")
            return

        if np.all(board != 0):
            print_board(board)
            print("Draw!")
            return

        current = 2 if current == 1 else 1


def _check_win(board: np.ndarray, row: int, col: int, color: int) -> bool:
    """Check if placing at (row, col) made 5 in a row."""
    for dr, dc in DIRECTIONS:
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
    parser.add_argument("--data", default=None, help="Path to move_sequences.csv (auto-detected if omitted)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Data directory to scan")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Model save/load path")
    parser.add_argument("--clusters", type=int, default=6, help="Number of K-Means clusters")
    parser.add_argument("--kfolds", type=int, default=0, help="K-fold cross-validation folds (0=disabled)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--update", action="store_true", help="Retrain on new scraper data only")
    parser.add_argument("--play", action="store_true", help="Play against the AI")
    parser.add_argument("--color", choices=["black", "white"], default="black",
                        help="Your color when playing (default: black)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on training data")
    parser.add_argument("--no-save", action="store_true", help="Don't save model after training")

    args = parser.parse_args()

    if not args.play and not args.evaluate and not args.update:
        args.train = True

    # Auto-detect data path
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = find_scraper_data(Path(args.data_dir))
        if data_path is None:
            print(f"Error: No move_sequences.csv found in {args.data_dir}/. Run gomoku_scraper.py first.")
            sys.exit(1)
        print(f"Auto-detected data: {data_path}")

    if args.train or args.update:
        if not data_path.exists():
            print(f"Error: {data_path} not found.")
            sys.exit(1)
        print(f"Loading {data_path}...")
        df = pd.read_csv(data_path)
        print(f"  {len(df)} rows, {df['game_id'].nunique()} games")

    ai = GomokuAI(n_clusters=args.clusters)

    if args.update:
        model_path = Path(args.model)
        if model_path.exists():
            ai.load(str(model_path))
        else:
            print("No existing model found. Training from scratch.")
        ai.update(df, kfolds=args.kfolds)
        if not args.no_save:
            ai.save(args.model)

    elif args.train:
        ai.train(df, kfolds=args.kfolds)
        if not args.no_save:
            ai.save(args.model)

    if args.evaluate:
        if ai.kmeans is None:
            ai.load(args.model)
        if not data_path.exists():
            print(f"Error: {data_path} not found.")
            sys.exit(1)
        df = pd.read_csv(data_path)
        ai.evaluate(df)

    if args.play:
        if ai.kmeans is None:
            ai.load(args.model)
        human_color = 1 if args.color == "black" else 2
        play_interactive(ai, human_color)


if __name__ == "__main__":
    main()
