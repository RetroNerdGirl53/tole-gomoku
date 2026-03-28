# Gomoku API

Python API for getting AI move recommendations on a 19x19 Gomoku board. Uses a pre-trained K-Means clustering + logistic regression model to score candidate moves by win probability.

## Requirements

- Python 3.11+
- `numpy`
- `scikit-learn`
- A trained model at `data/gomoku_ai.pkl` (run `python gomoku_ai.py --train` first)

## Quick Start

```python
from gomoku_api import GomokuAPI

api = GomokuAPI()
move = api.get_move(board, color_to_play=1)
top5 = api.get_top_moves(board, color_to_play=2, n=5)
analysis = api.score_position(board, color_to_play=1)
```

Or use module-level functions (lazy-loads the model on first call):

```python
from gomoku_api import get_move, get_top_moves, score_position

move = get_move(board, color_to_play=1)
top5 = get_top_moves(board, color_to_play=2, n=5)
analysis = score_position(board, color_to_play=1)
```

## Board Format

The board is a 19x19 grid. Accepts `numpy.ndarray` or nested lists.

| Value | Meaning |
|-------|---------|
| `0` | Empty |
| `1` | Black stone |
| `2` | White stone |

```python
import numpy as np

# Empty board
board = np.zeros((19, 19), dtype=int)

# Or as a nested list
board = [[0] * 19 for _ in range(19)]
```

## API Reference

### `GomokuAPI` Class

#### `GomokuAPI(model_path: str = "data/gomoku_ai.pkl")`

Create an API instance. Loads the pre-trained model from disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | `"data/gomoku_ai.pkl"` | Path to the trained model file |

```python
api = GomokuAPI()
api = GomokuAPI(model_path="my_model.pkl")
```

---

#### `get_move(board, color_to_play: int) -> tuple[int, int] | None`

Get the single best move for the current position.

| Parameter | Type | Description |
|-----------|------|-------------|
| `board` | `list` or `np.ndarray` | 19x19 board (0=empty, 1=black, 2=white) |
| `color_to_play` | `int` | `1`=black, `2`=white |

**Returns:** `(row, col)` tuple (0-indexed), or `None` if the board is full.

```python
move = api.get_move(board, color_to_play=1)
if move:
    print(f"Best move: row={move[0]}, col={move[1]}")
```

---

#### `get_top_moves(board, color_to_play: int, n: int = 5) -> list[tuple[tuple[int, int], float]]`

Get the top N moves ranked by win probability.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `board` | `list` or `np.ndarray` | — | 19x19 board (0=empty, 1=black, 2=white) |
| `color_to_play` | `int` | — | `1`=black, `2`=white |
| `n` | `int` | `5` | Number of top moves to return |

**Returns:** List of `((row, col), win_probability)` sorted descending by probability.

```python
moves = api.get_top_moves(board, color_to_play=2, n=3)
for (row, col), prob in moves:
    print(f"  ({row}, {col})  win_prob={prob:.3f}")
```

---

#### `score_position(board, color_to_play: int) -> dict`

Get a full analysis of the position, including best move, top candidates, cluster assignment, and board occupancy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `board` | `list` or `np.ndarray` | 19x19 board (0=empty, 1=black, 2=white) |
| `color_to_play` | `int` | `1`=black, `2`=white |

**Returns:** `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `best_move` | `tuple[int, int]` or `None` | `(row, col)` of the top-ranked move |
| `best_prob` | `float` | Win probability of the best move (0.0 if no moves) |
| `top_moves` | `list` | Top 5 moves as `((row, col), win_probability)` |
| `cluster` | `int` | K-Means cluster ID this position belongs to (0-based) |
| `empty_cells` | `int` | Number of empty cells on the board |

```python
analysis = api.score_position(board, color_to_play=1)
print(f"Best move: {analysis['best_move']} (prob={analysis['best_prob']:.3f})")
print(f"Position cluster: {analysis['cluster']}")
print(f"Empty cells: {analysis['empty_cells']}")
print(f"Top 5 moves: {analysis['top_moves']}")
```

---

### Module-Level Functions

These are convenience wrappers around a lazily-initialized singleton `GomokuAPI` instance. The model is loaded on the first call.

#### `get_move(board, color_to_play: int) -> tuple[int, int] | None`

Equivalent to `GomokuAPI().get_move(...)`.

#### `get_top_moves(board, color_to_play: int, n: int = 5) -> list[tuple[tuple[int, int], float]]`

Equivalent to `GomokuAPI().get_top_moves(...)`.

#### `score_position(board, color_to_play: int) -> dict`

Equivalent to `GomokuAPI().score_position(...)`.

---

### Board Validation

The internal `_to_array(board)` helper converts lists to `numpy.ndarray` and validates the shape. Raises `ValueError` if the board is not 19x19.

## Error Handling

| Error | Cause |
|-------|-------|
| `ValueError` | Board shape is not 19x19 |
| `RuntimeError` | Model not trained (from `gomoku_ai.py`) |
| `FileNotFoundError` | Model file does not exist at `model_path` |

## Examples

### Get best move for black

```python
from gomoku_api import get_move

board = [[0] * 19 for _ in range(19)]
board[9][9] = 1   # black center
board[9][10] = 2  # white response

move = get_move(board, color_to_play=1)
print(f"Black should play: {move}")
```

### Evaluate a mid-game position

```python
from gomoku_api import score_position
import numpy as np

board = np.zeros((19, 19), dtype=int)
# ... set up board state ...

analysis = score_position(board, color_to_play=2)
print(f"White's best move: {analysis['best_move']}")
print(f"Win probability: {analysis['best_prob']:.1%}")
print(f"Game phase: {analysis['empty_cells']} empty cells")
```

### Compare multiple candidate moves

```python
from gomoku_api import GomokuAPI

api = GomokuAPI()
candidates = api.get_top_moves(board, color_to_play=1, n=10)

print("Top 10 moves for black:")
for i, ((r, c), prob) in enumerate(candidates, 1):
    print(f"  {i:>2}. ({r:>2}, {c:>2})  {prob:.1%}")
```

## How It Works

1. **K-Means clustering** groups board positions into classes of similar game states (default: 6 clusters).
2. **Logistic regression per cluster** scores candidate moves using 15 rich features:
   - Line potential: stone counts and open ends in 4 directions (horizontal, vertical, 2 diagonals)
   - Neighbor density: friendly/opponent stones in 3x3 and 5x5 windows
   - Center proximity: normalized distance from board center
   - Game phase: fraction of empty cells
   - Adjacency: whether the move is near existing stones
3. At inference time: the board is assigned to a cluster, each empty cell is scored, and moves are ranked by predicted win probability.

## CLI (Training)

The API relies on a trained model. Use `gomoku_ai.py` to train:

```bash
# Train on scraped data
python gomoku_ai.py --train

# Train with 5-fold cross-validation
python gomoku_ai.py --kfolds 5

# Incremental update with new data
python gomoku_ai.py --update

# Play against the AI
python gomoku_ai.py --play

# Evaluate model accuracy
python gomoku_ai.py --evaluate
```

See `python gomoku_ai.py --help` for all options.
