#!/usr/bin/env python3
"""
Gomoku Game Scraper — collects full game history and live games from the
Gomoku server at http://108.81.9.145:10101/

Outputs clean pandas DataFrames (CSV + Parquet) for ML training:
  - games.csv       : one row per game (metadata, result, ELO)
  - moves.csv       : one row per move (game_id, move_num, row, col, color, is_opening)
  - board_states.csv: one row per game snapshot (sparse board as features)

Usage:
    python gomoku_scraper.py                 # scrape history + watch live
    python gomoku_scraper.py --history-only  # scrape all completed games then exit
    python gomoku_scraper.py --output-dir data/
"""

import argparse
import json
import os
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import websocket

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_WS_URL = "ws://108.81.9.145:10101/ws"
DEFAULT_TOKEN = "1d4c0e03-6ccb-4fc1-b22d-d6728d131c7e"
BOARD_SIZE = 19


# ─── Scraper ──────────────────────────────────────────────────────────────────

class GomokuScraper:
    """Connects to the Gomoku WebSocket server, collects game data, and
    exports clean pandas DataFrames for ML training."""

    def __init__(self, ws_url: str, token: str, output_dir: str = "data",
                 history_only: bool = False, spectate_batch_size: int = 5):
        self.ws_url = ws_url
        self.token = token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_only = history_only
        self.spectate_batch_size = spectate_batch_size

        # Internal state
        self.ws: websocket.WebSocketApp | None = None
        self.authenticated = False
        self.username: str | None = None

        # Game tracking
        self.games_meta: dict = {}          # game_id -> metadata dict
        self.games_moves: dict = {}         # game_id -> list of move dicts
        self.games_board: dict = {}         # game_id -> 19x19 board (latest)
        self.games_game_over: dict = {}     # game_id -> bool
        self.games_winner: dict = {}        # game_id -> str or None
        self.games_outcome: dict = {}       # game_id -> str or None
        self.games_elo_changes: dict = {}   # game_id -> dict

        # Track which games we've already spectated (completed)
        self.spectated_completed: set = set()
        self.spectating_active: set = set()

        # Spectate queue — rate-limited spectating
        self._spectate_queue: list = []       # game_ids waiting to be spectated
        self._spectate_in_flight: int = 0     # currently pending spectate responses
        self._spectate_max_concurrent = 3     # max simultaneous spectate requests
        self._spectate_delay = 0.8            # seconds between spectate sends

        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._save_interval = 30  # seconds between auto-saves
        self._last_save = time.time()

        # Stats
        self.stats = {
            "games_seen": 0,
            "games_completed": 0,
            "games_spectated": 0,
            "moves_collected": 0,
            "start_time": datetime.now().isoformat(),
        }

        # Load existing data to skip re-spectating
        self._load_existing_data()

    def _load_existing_data(self):
        """Load previously saved data to avoid re-spectating games."""
        games_path = self.output_dir / "games.csv"
        moves_path = self.output_dir / "moves.csv"

        if games_path.exists():
            try:
                df = pd.read_csv(games_path)
                for _, row in df.iterrows():
                    gid = row["game_id"]
                    self.games_meta[gid] = row.dropna().to_dict()
                    if row.get("status") == "completed":
                        self.spectated_completed.add(gid)
                print(f"[{self._ts()}] Loaded {len(self.games_meta)} games from existing data")
            except Exception as e:
                print(f"[{self._ts()}] Warning: could not load existing games: {e}")

        if moves_path.exists():
            try:
                df = pd.read_csv(moves_path)
                for gid, group in df.groupby("game_id"):
                    moves = []
                    for _, row in group.iterrows():
                        moves.append({
                            "row": int(row["row"]),
                            "col": int(row["col"]),
                            "color": int(row["color"]),
                            "is_opening": bool(row["is_opening"]),
                        })
                    self.games_moves[gid] = moves
                    self.stats["moves_collected"] += len(moves)
                print(f"[{self._ts()}] Loaded moves for {len(self.games_moves)} games "
                      f"({self.stats['moves_collected']} total moves)")
            except Exception as e:
                print(f"[{self._ts()}] Warning: could not load existing moves: {e}")

    # ─── WebSocket handlers ───────────────────────────────────────────────

    def _on_open(self, ws):
        print(f"[{self._ts()}] WebSocket connected. Authenticating...")
        ws.send(json.dumps({"type": "authenticate", "token": self.token}))

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "authenticated":
            self.authenticated = True
            self.username = msg.get("username")
            print(f"[{self._ts()}] Authenticated as '{self.username}'")

        elif msg_type == "lobby_update":
            self._handle_lobby_update(msg)

        elif msg_type == "active_games_update":
            self._handle_active_games_update(msg)

        elif msg_type == "spectate_game":
            self._handle_spectate_game(msg)

        elif msg_type == "game_update":
            self._handle_game_update(msg)

        elif msg_type == "ping":
            ws.send(json.dumps({"type": "pong"}))

        elif msg_type == "error":
            print(f"[{self._ts()}] Server error: {msg.get('message')}")

        # Auto-save periodically
        if time.time() - self._last_save > self._save_interval:
            self.save_data()
            self._last_save = time.time()

    def _on_error(self, ws, error):
        print(f"[{self._ts()}] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[{self._ts()}] WebSocket closed: {close_status_code} {close_msg}")
        if not self._stop_event.is_set():
            print(f"[{self._ts()}] Reconnecting in 3s...")
            time.sleep(3)
            self._connect()

    # ─── Message handlers ─────────────────────────────────────────────────

    def _handle_lobby_update(self, msg: dict):
        """Lobby games waiting for opponents — just track metadata."""
        for game in msg.get("games", []):
            gid = game["game_id"]
            if gid not in self.games_meta:
                self.games_meta[gid] = {
                    "game_id": gid,
                    "player1": game.get("creator"),
                    "player1_elo": game.get("creator_elo"),
                    "is_bot": game.get("is_bot"),
                    "created_at": game.get("created_at"),
                    "status": "waiting",
                }

    def _handle_active_games_update(self, msg: dict):
        """Main data source: server pushes ALL games (active + completed)."""
        completed_ids = []

        for game in msg.get("games", []):
            gid = game["game_id"]
            status = game.get("status", "active")

            with self._lock:
                is_new = gid not in self.games_meta
                self.games_meta[gid] = {
                    "game_id": gid,
                    "player1": game.get("player1"),
                    "player2": game.get("player2"),
                    "player1_elo": game.get("player1_elo"),
                    "player2_elo": game.get("player2_elo"),
                    "player1_color": game.get("player1_color"),
                    "combined_elo": game.get("combined_elo"),
                    "move_count": game.get("move_count", 0),
                    "spectator_count": game.get("spectator_count", 0),
                    "status": status,
                    "current_turn": game.get("current_turn"),
                    "winner": game.get("winner"),
                    "end_time": game.get("end_time"),
                }

                if is_new:
                    self.stats["games_seen"] += 1

                if status == "completed" and gid not in self.spectated_completed:
                    completed_ids.append(gid)
                    self.stats["games_completed"] += 1

        # Queue completed games for spectating (rate-limited)
        if completed_ids:
            print(f"[{self._ts()}] Found {len(completed_ids)} new completed game(s). "
                  f"Total completed: {self.stats['games_completed']}")
            with self._lock:
                self._spectate_queue.extend(completed_ids)

        # Also queue some active games for live tracking
        active_ids = [
            g["game_id"] for g in msg.get("games", [])
            if g.get("status") == "active"
            and g["game_id"] not in self.spectating_active
            and g["game_id"] not in self.spectated_completed
        ]
        if active_ids and not self.history_only:
            to_watch = active_ids[:self.spectate_batch_size]
            with self._lock:
                for gid in to_watch:
                    self.spectating_active.add(gid)
                self._spectate_queue.extend(to_watch)

    def _handle_spectate_game(self, msg: dict):
        """Initial spectate response: full board + move history."""
        gid = msg.get("game_id")
        if not gid:
            return

        with self._lock:
            self._ingest_game_state(msg)
            self.stats["games_spectated"] += 1
            self._spectate_in_flight = max(0, self._spectate_in_flight - 1)

        move_count = len(msg.get("move_history", []))
        p1 = msg.get("player1", "?")
        p2 = msg.get("player2", "?")
        q_remaining = len(self._spectate_queue)
        print(f"[{self._ts()}] Spectating {gid}: {p1} vs {p2} — {move_count} moves "
              f"(queue: {q_remaining})")

    def _handle_game_update(self, msg: dict):
        """Live game update: board + move history changes."""
        gid = msg.get("game_id")
        if not gid:
            return

        with self._lock:
            self._ingest_game_state(msg)

            if msg.get("game_over"):
                self.spectated_completed.add(gid)
                self.spectating_active.discard(gid)
                winner = msg.get("winner", "?")
                outcome = msg.get("outcome", "")
                print(f"[{self._ts()}] Game {gid} finished — Winner: {winner} ({outcome})")

    def _ingest_game_state(self, msg: dict):
        """Extract and store game state from a spectate/update message."""
        gid = msg["game_id"]

        # Board state
        if "board" in msg:
            self.games_board[gid] = msg["board"]

        # Move history
        if "move_history" in msg:
            moves = msg["move_history"]
            prev = self.games_moves.get(gid, [])
            if len(moves) > len(prev):
                self.games_moves[gid] = moves
                self.stats["moves_collected"] += len(moves) - len(prev)

        # Game state
        self.games_game_over[gid] = msg.get("game_over", False)
        self.games_winner[gid] = msg.get("winner")
        self.games_outcome[gid] = msg.get("outcome")
        self.games_elo_changes[gid] = msg.get("elo_changes", {})

        # Update metadata
        if gid in self.games_meta:
            meta = self.games_meta[gid]
            for key in ("player1", "player2", "player1_color", "player1_elo", "player2_elo"):
                if key in msg and msg[key] is not None:
                    meta[key] = msg[key]
            if msg.get("game_over"):
                meta["status"] = "completed"
                meta["winner"] = msg.get("winner")
                meta["outcome"] = msg.get("outcome")
                meta["elo_changes"] = msg.get("elo_changes", {})
            meta["move_count"] = len(msg.get("move_history", meta.get("move_count", 0)))

    # ─── Spectating (queue-based, rate-limited) ────────────────────────────

    def _process_spectate_queue(self):
        """Drain the spectate queue at a controlled rate."""
        while not self._stop_event.is_set():
            gid = None
            with self._lock:
                if (self._spectate_queue
                        and self._spectate_in_flight < self._spectate_max_concurrent):
                    gid = self._spectate_queue.pop(0)
                    self._spectate_in_flight += 1

            if gid:
                if self.ws and self.ws.sock and self.ws.sock.connected:
                    self.ws.send(json.dumps({"type": "spectate", "game_id": gid}))
                else:
                    with self._lock:
                        self._spectate_in_flight = max(0, self._spectate_in_flight - 1)
            time.sleep(self._spectate_delay)

    # ─── Data export ──────────────────────────────────────────────────────

    def build_games_df(self) -> pd.DataFrame:
        """One row per game with metadata and results."""
        rows = []
        for gid, meta in self.games_meta.items():
            moves = self.games_moves.get(gid, [])
            elo_changes = self.games_elo_changes.get(gid, {})

            # Determine player colors from moves or metadata
            p1_color = meta.get("player1_color")
            if p1_color is None and moves:
                # Infer from first non-opening move or opening moves
                for m in moves:
                    if not m.get("is_opening"):
                        # The player who moves first after opening is player1_color
                        p1_color = m["color"]
                        break

            row = {
                "game_id": gid,
                "player1": meta.get("player1"),
                "player2": meta.get("player2"),
                "player1_elo": meta.get("player1_elo"),
                "player2_elo": meta.get("player2_elo"),
                "player1_color": p1_color,
                "status": meta.get("status"),
                "winner": meta.get("winner"),
                "outcome": meta.get("outcome"),
                "total_moves": len(moves) if moves else meta.get("move_count", 0),
                "created_at": meta.get("created_at"),
                "end_time": meta.get("end_time"),
                "elo_changes_player1": elo_changes.get(meta.get("player1")),
                "elo_changes_player2": elo_changes.get(meta.get("player2")),
                "is_bot_game": meta.get("is_bot"),
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Normalize types
        for col in ["player1_elo", "player2_elo", "total_moves", "player1_color"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df

    def build_moves_df(self) -> pd.DataFrame:
        """One row per move. Core training data for ML."""
        rows = []
        for gid, moves in self.games_moves.items():
            meta = self.games_meta.get(gid, {})
            p1 = meta.get("player1")
            p2 = meta.get("player2")
            p1_color = meta.get("player1_color")
            winner = meta.get("winner")

            for i, move in enumerate(moves):
                color = move["color"]
                # Map color to player name
                if p1_color == color:
                    player = p1
                elif p1_color is not None:
                    player = p2 if color != p1_color else p1
                else:
                    player = None

                # Determine if this move's player won
                if winner and player:
                    move_result = 1 if player == winner else -1 if winner != "draw" else 0
                else:
                    move_result = None

                rows.append({
                    "game_id": gid,
                    "move_number": i + 1,
                    "row": move["row"],
                    "col": move["col"],
                    "color": color,  # 1=black, 2=white
                    "is_opening": move.get("is_opening", False),
                    "player": player,
                    "player1": p1,
                    "player2": p2,
                    "winner": winner,
                    "move_result": move_result,  # 1=winning side, -1=losing side
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["move_number"] = df["move_number"].astype(int)
        df["row"] = df["row"].astype(int)
        df["col"] = df["col"].astype(int)
        df["color"] = df["color"].astype(int)
        return df

    def build_board_states_df(self) -> pd.DataFrame:
        """One row per game with the final board state as flat features.
        Useful for position-based ML (e.g., CNN input)."""
        rows = []
        for gid, board in self.games_board.items():
            if not board:
                continue
            meta = self.games_meta.get(gid, {})
            flat = {}
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    flat[f"b_{r}_{c}"] = board[r][c] if r < len(board) and c < len(board[r]) else 0

            flat["game_id"] = gid
            flat["winner"] = meta.get("winner")
            flat["total_moves"] = len(self.games_moves.get(gid, []))
            rows.append(flat)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def build_moves_sequence_df(self) -> pd.DataFrame:
        """Board state AFTER each move as flat features + move label.
        Best for sequence models: predicts next move from board state."""
        rows = []
        for gid, moves in self.games_moves.items():
            if not moves:
                continue

            meta = self.games_meta.get(gid, {})
            p1_color = meta.get("player1_color", 1)
            winner = meta.get("winner")

            # Reconstruct board step by step
            board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]

            for i, move in enumerate(moves):
                r, c = move["row"], move["col"]
                color = move["color"]

                # Record state BEFORE this move (input features)
                state = {}
                for br in range(BOARD_SIZE):
                    for bc in range(BOARD_SIZE):
                        state[f"b_{br}_{bc}"] = board[br][bc]

                state["game_id"] = gid
                state["move_number"] = i + 1
                state["color_to_play"] = color
                state["is_opening"] = move.get("is_opening", False)

                # Label: the move position
                state["label_row"] = r
                state["label_col"] = c
                state["label_flat"] = r * BOARD_SIZE + c  # 0-360

                # Outcome label
                player = meta.get("player1") if color == p1_color else meta.get("player2")
                if winner and player:
                    state["move_won"] = 1 if player == winner else 0
                else:
                    state["move_won"] = None

                rows.append(state)

                # Apply move to board
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    board[r][c] = color

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    # ─── Save / Load ──────────────────────────────────────────────────────

    def save_data(self):
        """Save all collected data to CSV and Parquet files."""
        games_df = self.build_games_df()
        moves_df = self.build_moves_df()

        if not games_df.empty:
            games_df.to_csv(self.output_dir / "games.csv", index=False)
            try:
                games_df.to_parquet(self.output_dir / "games.parquet", index=False)
            except Exception:
                pass  # parquet optional

        if not moves_df.empty:
            moves_df.to_csv(self.output_dir / "moves.csv", index=False)
            try:
                moves_df.to_parquet(self.output_dir / "moves.parquet", index=False)
            except Exception:
                pass

        # Save stats
        self.stats["last_save"] = datetime.now().isoformat()
        with open(self.output_dir / "scraper_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        print(f"[{self._ts()}] Saved: {len(games_df)} games, {len(moves_df)} moves "
              f"({self.stats['moves_collected']} total move records)")

    def save_ml_datasets(self):
        """Save ML-ready datasets (board states, move sequences)."""
        board_df = self.build_board_states_df()
        sequence_df = self.build_moves_sequence_df()

        if not board_df.empty:
            board_df.to_csv(self.output_dir / "board_states.csv", index=False)
            try:
                board_df.to_parquet(self.output_dir / "board_states.parquet", index=False)
            except Exception:
                pass

        if not sequence_df.empty:
            sequence_df.to_csv(self.output_dir / "move_sequences.csv", index=False)
            try:
                sequence_df.to_parquet(self.output_dir / "move_sequences.parquet", index=False)
            except Exception:
                pass

        print(f"[{self._ts()}] ML datasets: {len(board_df)} board states, "
              f"{len(sequence_df)} move-sequence rows")

    # ─── Main loop ────────────────────────────────────────────────────────

    def _connect(self):
        """Establish WebSocket connection."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

    def run(self, timeout: int | None = None):
        """Run the scraper. Blocks until stopped or timeout (seconds)."""
        print(f"[{self._ts()}] Starting Gomoku Scraper")
        print(f"  Server: {self.ws_url}")
        print(f"  Output: {self.output_dir}")
        print(f"  Mode: {'history-only' if self.history_only else 'history + live'}")
        print()

        self._connect()

        # Start spectate queue processor
        queue_thread = threading.Thread(target=self._process_spectate_queue, daemon=True)
        queue_thread.start()

        # Run WebSocket in a thread so we can handle timeout + signal
        ws_thread = threading.Thread(
            target=lambda: self.ws.run_forever(ping_interval=30, ping_timeout=10),
            daemon=True,
        )
        ws_thread.start()

        try:
            if timeout:
                time.sleep(timeout)
            else:
                # Run until interrupted
                while not self._stop_event.is_set():
                    time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[{self._ts()}] Interrupted by user")

        self.shutdown()

    def shutdown(self):
        """Graceful shutdown: save data and close connection."""
        self._stop_event.set()
        print(f"\n[{self._ts()}] Shutting down...")

        # Final save
        self.save_data()
        self.save_ml_datasets()

        # Print summary
        self._print_summary()

        if self.ws:
            self.ws.close()

    def _print_summary(self):
        """Print final collection summary."""
        print("\n" + "=" * 60)
        print("SCRAPER SUMMARY")
        print("=" * 60)
        print(f"  Games seen:          {self.stats['games_seen']}")
        print(f"  Games completed:     {self.stats['games_completed']}")
        print(f"  Games spectated:     {self.stats['games_spectated']}")
        print(f"  Move records:        {self.stats['moves_collected']}")
        print(f"  Unique games w/moves: {len(self.games_moves)}")
        print(f"  Output directory:    {self.output_dir}")
        print()

        games_df = self.build_games_df()
        moves_df = self.build_moves_df()

        if not games_df.empty:
            print("Games DataFrame:")
            print(f"  Shape: {games_df.shape}")
            print(f"  Columns: {list(games_df.columns)}")
            print()

        if not moves_df.empty:
            print("Moves DataFrame:")
            print(f"  Shape: {moves_df.shape}")
            print(f"  Columns: {list(moves_df.columns)}")
            print(f"  Unique games: {moves_df['game_id'].nunique()}")
            print(f"  Avg moves/game: {moves_df.groupby('game_id').size().mean():.1f}")
            print()

        print("Files saved:")
        for f in sorted(self.output_dir.iterdir()):
            size = f.stat().st_size
            print(f"  {f.name:30s} {size:>10,} bytes")
        print("=" * 60)

    @staticmethod
    def _ts():
        return datetime.now().strftime("%H:%M:%S")


# ─── Data loading helpers ─────────────────────────────────────────────────────

def load_scraper_data(output_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Load all saved scraper data into pandas DataFrames.

    Returns:
        dict with keys: 'games', 'moves', 'board_states', 'move_sequences'
    """
    d = Path(output_dir)
    result = {}

    for name, filename in [
        ("games", "games.csv"),
        ("moves", "moves.csv"),
        ("board_states", "board_states.csv"),
        ("move_sequences", "move_sequences.csv"),
    ]:
        path = d / filename
        if path.exists():
            result[name] = pd.read_csv(path)
        else:
            result[name] = pd.DataFrame()

    return result


def print_data_summary(output_dir: str = "data"):
    """Quick summary of collected data."""
    data = load_scraper_data(output_dir)

    print("\nGomoku Scraper Data Summary")
    print("-" * 40)

    for name, df in data.items():
        if df.empty:
            print(f"  {name:20s}: (empty)")
        else:
            print(f"  {name:20s}: {df.shape[0]:>6,} rows x {df.shape[1]} cols")

    if not data["games"].empty:
        games = data["games"]
        print(f"\n  Completed games: {(games['status'] == 'completed').sum()}")
        print(f"  Active games:    {(games['status'] == 'active').sum()}")

    if not data["moves"].empty:
        moves = data["moves"]
        print(f"  Unique games:    {moves['game_id'].nunique()}")
        print(f"  Avg moves/game:  {moves.groupby('game_id').size().mean():.1f}")
        print(f"  Opening moves:   {moves['is_opening'].sum()}")

    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gomoku game scraper for ML training data")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="WebSocket URL")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Auth token")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--history-only", action="store_true",
                        help="Scrape all completed games then exit")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Run for N seconds then exit")
    parser.add_argument("--spectate-batch", type=int, default=5,
                        help="Max active games to spectate simultaneously")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing data and exit")

    args = parser.parse_args()

    if args.summary:
        print_data_summary(args.output_dir)
        return

    scraper = GomokuScraper(
        ws_url=args.ws_url,
        token=args.token,
        output_dir=args.output_dir,
        history_only=args.history_only,
        spectate_batch_size=args.spectate_batch,
    )

    # Handle signals for graceful shutdown
    def handle_signal(signum, frame):
        scraper.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    scraper.run(timeout=args.timeout)


if __name__ == "__main__":
    main()
