# Gomoku Scraper

WebSocket-based scraper for the Gomoku game server at `http://108.81.9.145:10101/`. Collects full game history and live games, outputting clean pandas DataFrames for ML model training.

## Requirements

- Python 3.11+
- `pandas`
- `websocket-client`

```bash
pip install pandas websocket-client
```

## Quick Start

```bash
# Scrape history + watch live games (runs until Ctrl+C)
python gomoku_scraper.py

# Scrape for 5 minutes then exit
python gomoku_scraper.py --timeout 300

# Save to a custom directory
python gomoku_scraper.py --output-dir my_data/

# Print summary of already-collected data
python gomoku_scraper.py --summary
```

## How It Works

The Gomoku server has no REST history endpoint. All data flows through a single WebSocket connection:

1. **Connect** to `ws://108.81.9.145:10101/ws` and authenticate with a token.
2. The server continuously pushes `active_games_update` messages containing all games (active and completed), including player names, ELO ratings, move counts, and status.
3. The scraper queues completed games and sends `spectate` requests (rate-limited to ~3 concurrent) to retrieve full board state and move history for each game.
4. Active games are also spectated for live tracking — the server pushes `game_update` messages as moves happen.
5. Data is auto-saved to CSV every 30 seconds. On restart, previously collected games are loaded from disk and skipped.

### Server WebSocket Protocol

| Direction | Message Type | Description |
|-----------|-------------|-------------|
| Client → Server | `authenticate` | `{"type": "authenticate", "token": "..."}` |
| Server → Client | `authenticated` | `{"type": "authenticated", "username": "..."}` |
| Server → Client | `lobby_update` | Games waiting for opponents |
| Server → Client | `active_games_update` | All games (active + completed) with metadata |
| Client → Server | `spectate` | `{"type": "spectate", "game_id": "..."}` |
| Server → Client | `spectate_game` | Full board + move history for spectated game |
| Server → Client | `game_update` | Live board + move history update |
| Server → Client | `ping` | Keep-alive (client must respond with `pong`) |

### Move Data Format (from server)

Each move in `move_history`:

```json
{
  "row": 9,
  "col": 7,
  "color": 2,
  "is_opening": false
}
```

- `color`: `1` = black, `2` = white
- `is_opening`: `true` for the 3 predefined opening moves, `false` for player-chosen moves

### Spectate Response Format

```json
{
  "type": "spectate_game",
  "game_id": "abc123",
  "player1": "bot_name",
  "player2": "other_bot",
  "player1_color": 1,
  "board": [[0,0,...], ...],
  "move_history": [{"row": 9, "col": 9, "color": 1, "is_opening": true}, ...],
  "current_turn": "other_bot",
  "game_over": false,
  "winner": null,
  "outcome": null,
  "time_left": 29.5,
  "elo_changes": {}
}
```

## Output Files

All files are written to the output directory (default: `data/`). Both CSV and Parquet formats are produced (Parquet requires `pyarrow` or `fastparquet`; failures are silently ignored).

### `games.csv` — Game Metadata

One row per game.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Unique game identifier |
| `player1` | str | First player name |
| `player2` | str | Second player name |
| `player1_elo` | int | Player 1's ELO rating |
| `player2_elo` | int | Player 2's ELO rating |
| `player1_color` | int | Color assigned to player 1 (`1`=black, `2`=white) |
| `status` | str | `waiting`, `active`, or `completed` |
| `winner` | str | Winner's username (null if not finished) |
| `outcome` | str | Win condition description |
| `total_moves` | int | Number of moves in the game |
| `created_at` | str | ISO timestamp of game creation |
| `end_time` | str/float | Unix timestamp of game end |
| `elo_changes_player1` | float | ELO change for player 1 after game |
| `elo_changes_player2` | float | ELO change for player 2 after game |
| `is_bot_game` | bool | Whether the game creator is a bot |

### `moves.csv` — Per-Move Data

One row per move. This is the primary training dataset.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Game this move belongs to |
| `move_number` | int | 1-indexed move number within the game |
| `row` | int | Board row (0–18) |
| `col` | int | Board column (0–18) |
| `color` | int | Stone color (`1`=black, `2`=white) |
| `is_opening` | bool | Whether this is a predefined opening move |
| `player` | str | Username of the player who made this move |
| `player1` | str | Game player 1 |
| `player2` | str | Game player 2 |
| `winner` | str | Game winner |
| `move_result` | int | `1` if this move's player won, `-1` if lost, `null` if game in progress |

### `move_sequences.csv` — Board State + Move Labels (ML-ready)

One row per move with the board state **before** that move as 361 features, plus the move as a label. Designed for supervised move prediction models.

| Column | Type | Description |
|--------|------|-------------|
| `b_0_0` … `b_18_18` | int | Board cell values (`0`=empty, `1`=black, `2`=white) before this move |
| `game_id` | str | Game identifier |
| `move_number` | int | Move index |
| `color_to_play` | int | Color of the stone about to be placed |
| `is_opening` | bool | Opening move flag |
| `label_row` | int | Target: row of the move |
| `label_col` | int | Target: column of the move |
| `label_flat` | int | Target: `row * 19 + col` (0–360), single-class label |
| `move_won` | int | `1` if this move's player won the game, `0` if lost, `null` if in progress |

### `board_states.csv` — Final Board States

One row per game with the final board state. Useful for position evaluation models.

| Column | Type | Description |
|--------|------|-------------|
| `b_0_0` … `b_18_18` | int | Final board cell values |
| `game_id` | str | Game identifier |
| `winner` | str | Game winner |
| `total_moves` | int | Total moves played |

### `scraper_stats.json` — Run Statistics

```json
{
  "games_seen": 142,
  "games_completed": 147,
  "games_spectated": 92,
  "moves_collected": 4138,
  "start_time": "2026-03-28T13:14:31.123456",
  "last_save": "2026-03-28T13:14:47.654321"
}
```

## Loading Data in Python

```python
import pandas as pd

# Direct CSV loading
games = pd.read_csv("data/games.csv")
moves = pd.read_csv("data/moves.csv")
sequences = pd.read_csv("data/move_sequences.csv")

# Or use the helper function
from gomoku_scraper import load_scraper_data
data = load_scraper_data("data")
games = data["games"]
moves = data["moves"]
sequences = data["move_sequences"]
board_states = data["board_states"]
```

### Example: Filter to Completed Bot Games

```python
completed = games[games["status"] == "completed"]
bot_games = completed[completed["is_bot_game"] == True]

# Get all moves for completed bot games
bot_moves = moves[moves["game_id"].isin(bot_games["game_id"])]
```

### Example: Prepare CNN Training Data

```python
import numpy as np

sequences = pd.read_csv("data/move_sequences.csv")

# Feature columns: board state
feature_cols = [f"b_{r}_{c}" for r in range(19) for c in range(19)]

# Filter to completed games only (move_won is not null)
completed = sequences[sequences["move_won"].notna()]

X = completed[feature_cols].values.reshape(-1, 19, 19)  # (N, 19, 19)
y = completed["label_flat"].values                        # (N,) — class 0-360
```

### Example: Load with Parquet (faster)

```python
import pandas as pd

games = pd.read_parquet("data/games.parquet")
moves = pd.read_parquet("data/moves.parquet")
sequences = pd.read_parquet("data/move_sequences.parquet")
```

## CLI Arguments

```
usage: gomoku_scraper.py [-h] [--ws-url WS_URL] [--token TOKEN]
                         [--output-dir OUTPUT_DIR] [--history-only]
                         [--timeout TIMEOUT] [--spectate-batch SPECTATE_BATCH]
                         [--summary]

Gomoku game scraper for ML training data

options:
  --ws-url WS_URL          WebSocket URL (default: ws://108.81.9.145:10101/ws)
  --token TOKEN            Auth token
  --output-dir OUTPUT_DIR  Output directory (default: data)
  --history-only           Only scrape completed games, skip live tracking
  --timeout TIMEOUT        Run for N seconds then exit
  --spectate-batch SPECTATE_BATCH
                           Max active games to spectate simultaneously (default: 5)
  --summary                Print summary of existing data and exit
```

## Programmatic Usage

```python
from gomoku_scraper import GomokuScraper

scraper = GomokuScraper(
    ws_url="ws://108.81.9.145:10101/ws",
    token="your-token-here",
    output_dir="data",
    history_only=False,
    spectate_batch_size=5,
)

# Run for 60 seconds
scraper.run(timeout=60)

# Or run until manually stopped
# scraper.run()

# Access data directly
games_df = scraper.build_games_df()
moves_df = scraper.build_moves_df()
sequences_df = scraper.build_moves_sequence_df()
board_df = scraper.build_board_states_df()
```

## Architecture

```
GomokuScraper
├── __init__              State init, load existing data
├── _load_existing_data   Resume from CSV on disk
├── _on_open              Authenticate on connect
├── _on_message           Route incoming WS messages
├── _on_close             Auto-reconnect
├── _handle_lobby_update  Track waiting games
├── _handle_active_games_update  Main data source — all games
├── _handle_spectate_game Ingest full game state
├── _handle_game_update   Ingest live game changes
├── _ingest_game_state    Extract board, moves, metadata
├── _process_spectate_queue  Rate-limited spectate worker
├── build_games_df        → games.csv
├── build_moves_df        → moves.csv
├── build_moves_sequence_df  → move_sequences.csv
├── build_board_states_df → board_states.csv
├── save_data             Write CSV + Parquet
├── save_ml_datasets      Write ML-specific CSV + Parquet
├── run                   Main loop (connects, starts threads)
└── shutdown              Save, print summary, close WS
```

### Threading Model

Three threads run concurrently:

1. **Main thread** — Blocks until timeout or Ctrl+C
2. **WebSocket thread** — Receives messages, dispatches to handlers
3. **Queue processor thread** — Drains the spectate queue at a controlled rate (~1 request/0.8s, max 3 in-flight)

All shared state mutations are protected by `threading.Lock`.

### Rate Limiting

The server can be overwhelmed by too many simultaneous spectate requests. The scraper uses:

- `_spectate_max_concurrent = 3` — max outstanding spectate requests
- `_spectate_delay = 0.8` — seconds between sending spectate requests

These can be adjusted on the `GomokuScraper` instance after construction:

```python
scraper._spectate_max_concurrent = 5
scraper._spectate_delay = 0.5
```

### Persistence and Resume

On startup, the scraper loads `games.csv` and `moves.csv` from the output directory. Games already present are marked as spectated and skipped. This means you can stop and restart the scraper without re-fetching previously collected games. Only new games (created since the last save) will be spectated.

## Board Representation

The board is a 19×19 grid. Cells are:

| Value | Meaning |
|-------|---------|
| `0` | Empty |
| `1` | Black stone |
| `2` | White stone |

Each game begins with 3 predefined opening moves (`is_opening: true`), after which players alternate placing stones. The goal is to get 5 in a row (horizontally, vertically, or diagonally).

## Known Behaviors

- **Games with 0 moves**: Some games (e.g., `Clanker1` vs human players who disconnect immediately) complete with 0 moves. These appear in `games.csv` but produce no rows in `moves.csv`.
- **Incomplete move histories**: Active games spectated mid-play will have partial move history. The history grows via subsequent `game_update` messages.
- **Null `move_result`**: Moves from games still in progress have `move_result = null` because there is no winner yet. Filter with `moves[moves["move_result"].notna()]` for training.
- **Duplicate game_ids on restart**: The scraper merges by game_id on load. Metadata is updated from the latest server state, move data is preserved from the previous session.
