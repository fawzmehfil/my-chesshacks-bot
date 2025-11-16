'''

from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.polyglot as polyglot
import random
import time
import os
from .utils.evaluation import CNNEvaluator
import torch

# ============================================================
# GLOBALS / CONFIG
# ============================================================

TT = {}  # zobrist_key -> (depth, flag, score, best_move)

INFINITY = 10_000_000
MAX_DEPTH = 4  # bump to 4+ if it’s fast enough

# TT flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# Basic piece values (used in eval AND MVV-LVA)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,   # king material value irrelevant in eval
}


model_path = os.path.join(os.path.dirname(__file__), "utils", "chess_cnn_final.pth")
cnn_eval = CNNEvaluator(model_path)
# model_path = "src/utils/chess_cnn_final.pth"

cnn_eval = CNNEvaluator(model_path)


def evaluate(board: chess.Board) -> float:
    """
    Uses CNN to evaluate the board.
    Positive = good for side to move.
    """
    score = cnn_eval.evaluate_board(board)

    return score if board.turn == chess.WHITE else -score

# ============================================================
# EVALUATION FUNCTION (placeholder for a "neural net")
# ============================================================
# def evaluate_basic(board: chess.Board) -> int:
#     """
#     placeholder eval function
#     """

#     # Material from White's perspective
#     score = 0
#     for ptype, value in PIECE_VALUES.items():
#         score += len(board.pieces(ptype, chess.WHITE)) * value
#         score -= len(board.pieces(ptype, chess.BLACK)) * value

#     # Mobility
#     mobility = board.legal_moves.count()
#     if board.turn == chess.WHITE:
#         score += 5 * mobility
#     else:
#         score -= 5 * mobility

#     # Convert to "side to move" perspective for negamax
#     return score if board.turn == chess.WHITE else -score


# ============================================================
# TRANSPOSITION TABLE HELPERS
# ============================================================

def tt_hash(board: chess.Board) -> int:
    """Compute a zobrist hash for the position."""
    return polyglot.zobrist_hash(board)


def tt_probe(board: chess.Board, depth: int, alpha: int, beta: int):
    """
    Try to retrieve a useful TT entry.
    Returns (hit, score, stored_move).
    """
    key = tt_hash(board)
    entry = TT.get(key)
    if entry is None:
        return False, None, None

    stored_depth, flag, stored_score, stored_move = entry

    if stored_depth < depth:
        return False, None, None

    if flag == EXACT:
        return True, stored_score, stored_move
    if flag == LOWERBOUND and stored_score >= beta:
        return True, stored_score, stored_move
    if flag == UPPERBOUND and stored_score <= alpha:
        return True, stored_score, stored_move

    return False, None, None


def tt_store(board: chess.Board, depth: int, flag: int, score: int, best_move: Move | None):
    """Store an entry in the TT (replace only if deeper or empty)."""
    key = tt_hash(board)
    existing = TT.get(key)
    if existing is None or existing[0] <= depth:
        TT[key] = (depth, flag, score, best_move)

    # Crude size control
    if len(TT) > 200_000:
        TT.clear()


# ============================================================
# MVV-LVA MOVE ORDERING
# ============================================================

def mvv_lva_score(board: chess.Board, move: Move) -> int:
    """
    MVV-LVA: Most Valuable Victim - Least Valuable Attacker.
    Higher score = better (we sort descending).
    """
    # Victim piece (handle en passant separately)
    victim_piece = None
    if board.is_en_passant(move):
        victim_piece = chess.Piece(chess.PAWN, not board.turn)
    else:
        victim_piece = board.piece_at(move.to_square)

    attacker_piece = board.piece_at(move.from_square)

    if victim_piece is None or attacker_piece is None:
        return 0

    victim_value = PIECE_VALUES[victim_piece.piece_type]
    attacker_value = PIECE_VALUES[attacker_piece.piece_type]

    # Victim high, attacker low is best
    return victim_value * 10 - attacker_value


def order_moves(board: chess.Board, moves, tt_move: Move | None):
    """
    Move ordering:
    - TT move first (if present)
    - Then captures ordered by MVV-LVA
    - Then quiet moves
    """

    def key(m: Move):
        score = 0

        # Transposition table best move gets a huge bonus
        if tt_move is not None and m == tt_move:
            score += 100_000

        # Captures ordered by MVV-LVA
        if board.is_capture(m):
            score += 1_000 + mvv_lva_score(board, m)

        # (You can add history / killer heuristics here if you want)

        return score

    return sorted(moves, key=key, reverse=True)


# ============================================================
# SEARCH (NEGAMAX + ALPHA-BETA)
# ============================================================

def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """
    Negamax with alpha-beta pruning and TT.
    Returns a score from the perspective of the side to move.
    """

    # Terminal nodes
    if board.is_game_over():
        if board.is_checkmate():
            return -INFINITY + 1
        return 0

    if depth == 0:
        return evaluate(board)

    alpha_orig = alpha

    # TT probe
    tt_hit, tt_score, tt_move = tt_probe(board, depth, alpha, beta)
    if tt_hit:
        return tt_score

    best_score = -INFINITY
    best_move: Move | None = None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -INFINITY + 1
        return 0

    ordered_moves = order_moves(board, legal_moves, tt_move)

    for move in ordered_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            if alpha >= beta:
                break  # beta cutoff

    # Store in TT
    if best_score <= alpha_orig:
        flag = UPPERBOUND
    elif best_score >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT

    tt_store(board, depth, flag, best_score, best_move)

    return best_score


def root_search(board: chess.Board, max_depth: int) -> Move | None:
    """
    Root search with iterative deepening.
    Returns the best move found, or None if no legal moves.
    """

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = random.choice(legal_moves)
    best_score = -INFINITY

    for depth in range(1, max_depth + 1):
        current_best_move = None
        current_best_score = -INFINITY

        tt_hit, _, tt_move = tt_probe(board, depth, -INFINITY, INFINITY)
        ordered_moves = order_moves(board, legal_moves, tt_move)

        alpha = -INFINITY
        beta = INFINITY

        for move in ordered_moves:
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if current_best_move is None or score > current_best_score:
                current_best_score = score
                current_best_move = move

            if score > alpha:
                alpha = score

        if current_best_move is not None:
            best_move = current_best_move
            best_score = current_best_score

    return best_move


# ============================================================
# ENTRYPOINTS (INTEGRATION WITH YOUR FRAMEWORK)
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called every time the model needs to make a move.
    Must return a python-chess Move that is legal for the current position.
    """

    print("Cooking move...")
    print("Move stack:", [m.uci() for m in ctx.board.move_stack])
    time.sleep(0.01)

    board = ctx.board

    # Pure search: no book, just minimax+alpha-beta+MVV-LVA
    best_move = root_search(board, MAX_DEPTH)

    if best_move is None or best_move not in board.legal_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (checkmate or stalemate).")
        best_move = random.choice(legal_moves)

    # Log a degenerate distribution: chosen move has probability 1
    ctx.logProbabilities({best_move: 1.0})

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    Clears transposition table and any model state.
    """
    global TT
    TT = {}
    print("New game: transposition table cleared.")


# if __name__ == "__main__":
#     game = chess.Board()

#     print(root_search(game, 2))

'''