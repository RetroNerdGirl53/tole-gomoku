import asyncio
import websockets
import json
import random
import sys
import os

# Load configuration from config.json
def load_config():
    """Load configuration from config.json"""
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {
        "server_url": "ws://localhost:8000/ws",
        "username": "",
        "token": ""
    }

# Load config
config = load_config()
SERVER_URL = config.get("server_url", "ws://localhost:8000/ws")
TOKEN = config.get("token", "")
USERNAME = config.get("username", "")

# Board representation
BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

WALL = 3  # sentinel value for out-of-bounds in pattern windows


class GomokuBot:
    def __init__(self, token, username):
        self.token = token
        self.username = username
        self.ws = None
        self.game_id = None
        self.my_color = None
        self.board = None
        self.game_over = False

    async def connect(self):
        """Connect to the game server"""
        print(f"[BOT] Connecting to {SERVER_URL}...")
        self.ws = await websockets.connect(SERVER_URL)

        await self.send_message({
            "type": "authenticate",
            "token": self.token
        })

        print(f"[BOT] Connected as {self.username}")

    async def send_message(self, message):
        """Send a message to the server"""
        await self.ws.send(json.dumps(message))

    async def handle_message(self, message):
        """Handle incoming messages from server"""
        msg_type = message.get("type")

        if msg_type == "authenticated":
            print(f"[BOT] Authenticated successfully")
            await self.find_game()

        elif msg_type == "ping":
            await self.send_message({"type": "pong"})

        elif msg_type == "lobby_update":
            games = message.get("games", [])
            if games:
                # Filter out games created by self to avoid self-match
                other_games = [g for g in games if g.get("creator") != self.username]
                if other_games:
                    target = other_games[0]
                    print(f"[BOT] Joining game {target['game_id']} created by {target['creator']}")
                    await self.send_message({
                        "type": "join_game",
                        "game_id": target["game_id"]
                    })
            # If no joinable games, we already created one via find_game()

        elif msg_type == "game_created":
            self.game_id = message.get("game_id")
            self.my_color = message.get("color")
            color_name = "Black" if self.my_color == BLACK else "White"
            print(f"[BOT] Created game {self.game_id}, playing as {color_name}")
            print(f"[BOT] Waiting for opponent...")

        elif msg_type == "game_started":
            self.game_id = message.get("game_id")
            self.my_color = message.get("your_color")
            player1 = message.get("player1")
            player2 = message.get("player2")
            opponent = player2 if player1 == self.username else player1
            color_name = "Black" if self.my_color == BLACK else "White"
            print(f"[BOT] Game started! Playing as {color_name} against {opponent}")

        elif msg_type == "game_update":
            self.board = message.get("board")
            current_turn = message.get("current_turn")
            game_over = message.get("game_over")
            winner = message.get("winner")

            if game_over:
                self.game_over = True
                if winner == self.username:
                    print(f"[BOT] Victory! I won the game!")
                elif winner:
                    print(f"[BOT] Defeat. {winner} won the game.")
                else:
                    print(f"[BOT] Game ended in a draw.")

                await asyncio.sleep(2)
                self.reset_game_state()
                await self.find_game()

            elif current_turn == self.my_color:
                move = self.choose_move(self.board, self.my_color, self.game_id)
                if move:
                    row, col = move
                    print(f"[BOT] Making move: ({row},{col})")
                    await self.send_message({
                        "type": "make_move",
                        "row": row,
                        "col": col
                    })

        elif msg_type == "error":
            code = message.get("code")
            error_message = message.get("message")
            print(f"[ERROR] {code}: {error_message}")

            if code == "ALREADY_IN_GAME":
                pass
            elif code == "GAME_NOT_FOUND":
                await self.find_game()

    async def find_game(self):
        """Create a new game and wait for an opponent"""
        print(f"[BOT] Creating new game")
        await self.send_message({"type": "create_game"})

    def reset_game_state(self):
        """Reset game state for next game"""
        self.game_id = None
        self.my_color = None
        self.board = None
        self.game_over = False

    def choose_move(self, board, my_color, game_id):
        """
        Score candidate cells near existing stones by scanning all 8 directions
        with a sliding window and accumulating scores. Scores from multiple
        directions add up, so double-threat cells naturally rank highest.
        """
        opponent = 3 - my_color

        # Count total stones placed to detect opening phase
        stone_count = sum(1 for r in range(BOARD_SIZE)
                          for c in range(BOARD_SIZE) if board[r][c] != EMPTY)

        # Opening: if we are about to play move 1 or 2, prefer center area
        if stone_count == 0:
            center = BOARD_SIZE // 2
            return (center, center)
        if stone_count == 1:
            center = BOARD_SIZE // 2
            # Play adjacent to the lone stone, biased toward center
            options = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = center + dr, center + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                        options.append((nr, nc))
            if options:
                return random.choice(options)

        candidates = self._get_candidates(board)

        if not candidates:
            center = BOARD_SIZE // 2
            return (center, center)

        # Build score grid only over candidates for speed
        scores = {}
        for row, col in candidates:
            scores[(row, col)] = score_cell(board, row, col, my_color, opponent)

        # If the top score is very high (threat or win), return immediately without
        # the random tiebreaker noise to avoid accidentally choosing a weaker move.
        best_score = max(scores.values())
        if best_score >= 8000:
            # High-stakes situation: pick the highest-scoring cell deterministically
            best = max(scores, key=lambda k: scores[k])
        else:
            best = max(scores, key=lambda k: scores[k] + random.random() * 0.5)
        return best

    def _get_candidates(self, board):
        """
        Return empty cells within radius of any placed stone.
        Uses radius 2 normally, but expands to radius 3 to ensure long-range
        threats (e.g. opponent building a row far from current action) are included.
        """
        occupied = [(r, c) for r in range(BOARD_SIZE)
                    for c in range(BOARD_SIZE) if board[r][c] != EMPTY]

        if not occupied:
            center = BOARD_SIZE // 2
            return [(center, center)]

        candidates = set()
        for r, c in occupied:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    # Radius 2: always include
                    # Radius 3: include only axis-aligned and diagonal extensions
                    # (chessboard distance ≤ 3 but skip pure off-axis cells at dist 3
                    #  to keep candidate set manageable)
                    if abs(dr) <= 2 and abs(dc) <= 2:
                        pass  # always include radius-2 neighborhood
                    elif abs(dr) == 3 or abs(dc) == 3:
                        # Only extend to radius 3 along the 8 main directions
                        if not (abs(dr) <= 1 or abs(dc) <= 1 or abs(dr) == abs(dc)):
                            continue
                    else:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == EMPTY:
                        candidates.add((nr, nc))
        return list(candidates)

    async def run(self):
        """Main bot loop"""
        self.reset_game_state()
        await self.connect()

        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self.handle_message(data)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse message: {e}")
                except Exception as e:
                    print(f"[ERROR] Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print(f"[BOT] Connection closed")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            if self.ws:
                await self.ws.close()


def score_cell(board, row, col, my_color, opponent):
    """
    Accumulate scores from all 8 directions around (row, col).
    Each direction contributes a 9-cell window where index 4 is the candidate cell.
    Scores from all directions are summed, naturally rewarding double-threat moves.
    """
    # 8 directions: right, down-right, down, down-left (+ opposites via window)
    wheel = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    total = 0

    for v, h in wheel:
        window = [WALL]*9
        window[4] = EMPTY  # the cell being evaluated
        for i in range(1, 5):
            r, c = row - v*i, col - h*i
            window[4-i] = board[r][c] if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE else WALL
        for i in range(1, 5):
            r, c = row + v*i, col + h*i
            window[4+i] = board[r][c] if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE else WALL
        total += score_window(window, my_color, opponent)

    # Center proximity tiebreaker
    center = BOARD_SIZE // 2
    total += max(0, 10 - (abs(row - center) + abs(col - center)))
    return total


def score_window(w, my, opp):
    """
    Score a 9-cell directional window where w[4] is the candidate empty cell.
    Scores accumulate — returning a value for each matching pattern so that
    a cell active in multiple directions gets points from all of them.

    Window layout: [w0, w1, w2, w3, w4=candidate, w5, w6, w7, w8]
    Directions checked for each 5-cell run containing position 4:
      positions [0-4], [1-5], [2-6], [3-7], [4-8]
    """
    score = 0

    # ── Immediate win: place candidate to make 5-in-a-row ───────────────────
    # All four adjacent positions in same direction as candidate are mine.
    if w[0]==my and w[1]==my and w[2]==my and w[3]==my: score += 20000  # ....X
    if w[1]==my and w[2]==my and w[3]==my and w[5]==my: score += 20000  # .XXX.X
    if w[3]==my and w[5]==my and w[6]==my and w[7]==my: score += 20000  # X.XXX.
    if w[5]==my and w[6]==my and w[7]==my and w[8]==my: score += 20000  # X....
    # Broken-four (skip-gap) wins: four stones in a 5-span with one internal gap
    # Placing candidate fills the gap to make 5-in-a-row.
    if w[1]==my and w[2]==my and w[3]==my and w[6]==my: score += 20000  # .XXX_X
    if w[0]==my and w[2]==my and w[3]==my and w[5]==my: score += 20000  # X_XX_X (spans [0-5])
    if w[0]==my and w[1]==my and w[3]==my and w[5]==my: score += 20000  # XX_X_X
    if w[3]==my and w[5]==my and w[6]==my and w[8]==my: score += 20000  # X_XX_X
    if w[2]==my and w[3]==my and w[5]==my and w[7]==my: score += 20000  # XX_X_X

    # ── Block opponent's immediate win (raised to equal attack priority) ─────
    if w[0]==opp and w[1]==opp and w[2]==opp and w[3]==opp: score += 19000
    if w[1]==opp and w[2]==opp and w[3]==opp and w[5]==opp: score += 19000
    if w[3]==opp and w[5]==opp and w[6]==opp and w[7]==opp: score += 19000
    if w[5]==opp and w[6]==opp and w[7]==opp and w[8]==opp: score += 19000
    # Broken-four blocks
    if w[1]==opp and w[2]==opp and w[3]==opp and w[6]==opp: score += 19000
    if w[0]==opp and w[2]==opp and w[3]==opp and w[5]==opp: score += 19000
    if w[0]==opp and w[1]==opp and w[3]==opp and w[5]==opp: score += 19000
    if w[3]==opp and w[5]==opp and w[6]==opp and w[8]==opp: score += 19000
    if w[2]==opp and w[3]==opp and w[5]==opp and w[7]==opp: score += 19000

    # ── Open four: candidate creates unstoppable double-ended four ───────────
    # Both sides of candidate are occupied by 2 of mine, with open ends beyond.
    if w[2]==my and w[3]==my and w[5]==my and w[6]==my:   score += 9000
    # Half-open four that still threatens immediate win next turn:
    # Three mine on one side, one on the other — opponent can only block one end.
    if w[1]==my and w[2]==my and w[3]==my and w[6]==EMPTY: score += 7000  # .XXX_._
    if w[2]==EMPTY and w[3]==my and w[5]==my and w[6]==my and w[7]==my: score += 7000  # _._XXX.
    if w[0]==EMPTY and w[1]==my and w[2]==my and w[3]==my and w[6]==EMPTY: score += 7000
    if w[2]==EMPTY and w[3]==my and w[5]==my and w[6]==my and w[8]==EMPTY: score += 7000

    # ── Block opponent open four (raised: must always exceed our open-three) ──
    if w[2]==opp and w[3]==opp and w[5]==opp and w[6]==opp: score += 8500
    # Block half-open fours for opponent
    if w[1]==opp and w[2]==opp and w[3]==opp and w[6]==EMPTY: score += 6500
    if w[2]==EMPTY and w[3]==opp and w[5]==opp and w[6]==opp and w[7]==opp: score += 6500
    if w[0]==EMPTY and w[1]==opp and w[2]==opp and w[3]==opp and w[6]==EMPTY: score += 6500
    if w[2]==EMPTY and w[3]==opp and w[5]==opp and w[6]==opp and w[8]==EMPTY: score += 6500

    # ── Open three (both ends open) — creates unstoppable fork threat ────────
    if w[1]==EMPTY and w[2]==my and w[3]==my and w[5]==my and w[6]==EMPTY:  score += 2000
    if w[0]==EMPTY and w[1]==my and w[2]==my and w[3]==my and w[5]==EMPTY:  score += 2000
    if w[1]==EMPTY and w[3]==my and w[5]==my and w[6]==my and w[7]==EMPTY:  score += 2000
    # Broken open three: my._.my.my or my.my._.my with open ends
    if w[0]==EMPTY and w[1]==my and w[2]==EMPTY and w[3]==my and w[5]==my and w[6]==EMPTY: score += 1800
    if w[0]==EMPTY and w[1]==my and w[2]==my and w[3]==EMPTY and w[5]==my and w[6]==EMPTY: score += 1800
    if w[1]==EMPTY and w[2]==my and w[3]==EMPTY and w[5]==my and w[6]==my and w[7]==EMPTY: score += 1800
    if w[1]==EMPTY and w[2]==EMPTY and w[3]==my and w[5]==my and w[6]==my and w[7]==EMPTY: score += 1800

    # ── Block opponent open three ────────────────────────────────────────────
    if w[1]==EMPTY and w[2]==opp and w[3]==opp and w[5]==opp and w[6]==EMPTY: score += 1700
    if w[0]==EMPTY and w[1]==opp and w[2]==opp and w[3]==opp and w[5]==EMPTY: score += 1700
    if w[1]==EMPTY and w[3]==opp and w[5]==opp and w[6]==opp and w[7]==EMPTY: score += 1700
    # Broken open three blocks
    if w[0]==EMPTY and w[1]==opp and w[2]==EMPTY and w[3]==opp and w[5]==opp and w[6]==EMPTY: score += 1500
    if w[0]==EMPTY and w[1]==opp and w[2]==opp and w[3]==EMPTY and w[5]==opp and w[6]==EMPTY: score += 1500
    if w[1]==EMPTY and w[2]==opp and w[3]==EMPTY and w[5]==opp and w[6]==opp and w[7]==EMPTY: score += 1500
    if w[1]==EMPTY and w[2]==EMPTY and w[3]==opp and w[5]==opp and w[6]==opp and w[7]==EMPTY: score += 1500

    # ── Half-open three (one end blocked, other open) ────────────────────────
    if w[1]==my and w[2]==my and w[3]==my and w[5]==EMPTY: score += 600
    if w[1]==EMPTY and w[2]==my and w[3]==my and w[5]==my: score += 600
    if w[3]==my and w[5]==my and w[6]==my and w[7]==EMPTY: score += 600
    # Broken half-open threes
    if w[2]==my and w[3]==EMPTY and w[5]==my and w[6]==my and w[7]==EMPTY: score += 500
    if w[1]==EMPTY and w[2]==my and w[3]==EMPTY and w[5]==my and w[6]==my: score += 500

    if w[1]==opp and w[2]==opp and w[3]==opp and w[5]==EMPTY: score += 500
    if w[1]==EMPTY and w[2]==opp and w[3]==opp and w[5]==opp: score += 500
    if w[3]==opp and w[5]==opp and w[6]==opp and w[7]==EMPTY: score += 500
    # Broken half-open three blocks
    if w[2]==opp and w[3]==EMPTY and w[5]==opp and w[6]==opp and w[7]==EMPTY: score += 420
    if w[1]==EMPTY and w[2]==opp and w[3]==EMPTY and w[5]==opp and w[6]==opp: score += 420

    # ── Open two ─────────────────────────────────────────────────────────────
    if w[2]==EMPTY and w[3]==my and w[5]==my and w[6]==EMPTY: score += 200
    if w[1]==EMPTY and w[2]==my and w[3]==EMPTY and w[5]==my and w[6]==EMPTY: score += 150
    if w[1]==EMPTY and w[2]==EMPTY and w[3]==my and w[5]==my and w[6]==EMPTY: score += 120
    if w[1]==EMPTY and w[2]==my and w[3]==my and w[5]==EMPTY and w[6]==EMPTY: score += 120

    if w[2]==EMPTY and w[3]==opp and w[5]==opp and w[6]==EMPTY: score += 170
    if w[1]==EMPTY and w[2]==opp and w[3]==EMPTY and w[5]==opp and w[6]==EMPTY: score += 130
    if w[1]==EMPTY and w[2]==EMPTY and w[3]==opp and w[5]==opp and w[6]==EMPTY: score += 110
    if w[1]==EMPTY and w[2]==opp and w[3]==opp and w[5]==EMPTY and w[6]==EMPTY: score += 110

    return score


async def main():
    if not TOKEN or not USERNAME:
        print("ERROR: Please set your TOKEN and USERNAME in config.json")
        print("Get your token by running:")
        print("  curl -X POST 'http://localhost:8000/register?username=YOUR_USERNAME&is_bot=true'")
        sys.exit(1)

    bot = GomokuBot(TOKEN, USERNAME)

    while True:
        try:
            await bot.run()
        except Exception as e:
            print(f"[ERROR] Bot crashed: {e}")

        print("[BOT] Reconnecting in 5 seconds...")
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
