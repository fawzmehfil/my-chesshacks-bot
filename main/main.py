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
MAX_DEPTH = 20  # bump to 4+ if it’s fast enough
MAX_TIME = 5 # seconds

# TT flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# Basic piece values (for MVV-LVA, etc.)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,   # king material value irrelevant in eval
}

# ============================================================
# MODEL INITIALIZATION (CNN EVALUATOR)
# ============================================================

model_path = os.path.join(os.path.dirname(__file__), "utils", "chess_cnn_final.pth")
cnn_eval = CNNEvaluator(model_path)


def evaluate(board: chess.Board) -> float:
    """
    Uses CNN to evaluate the board.
    Positive = good for side to move.
    """
    score = cnn_eval.evaluate_board(board)
    # CNN returns a score from White's POV; convert to side-to-move POV.
    return score if board.turn == chess.WHITE else -score


# ============================================================
# MVV-LVA TABLE
# ============================================================
# Indexing: MVV_LVA_TABLE[victim_type][attacker_type]
MVV_LVA_TABLE = [[0] * 7 for _ in range(7)]  # piece types 1..6

MVV_VICTIM_VALUE = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   10,  # arbitrary high; king captures are rare
}

MVV_ATTACKER_VALUE = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   10,
}

for victim_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                    chess.ROOK, chess.QUEEN, chess.KING):
    for attacker_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                          chess.ROOK, chess.QUEEN, chess.KING):
        v = MVV_VICTIM_VALUE[victim_type]
        a = MVV_ATTACKER_VALUE[attacker_type]
        # Most valuable victim (large v), least valuable attacker (small a)
        MVV_LVA_TABLE[victim_type][attacker_type] = v * 10 - a


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


def tt_store(board: chess.Board, depth: int, flag: int, score: float, best_move: Move | None):
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

    # Determine victim piece
    if board.is_en_passant(move):
        # En passant always captures a pawn
        victim_type = chess.PAWN
    else:
        victim_piece = board.piece_at(move.to_square)
        if victim_piece is None:
            return 0  # should not happen for a capture, but be safe
        victim_type = victim_piece.piece_type

    # Determine attacker piece type
    attacker_piece = board.piece_at(move.from_square)
    if attacker_piece is None:
        return 0

    # Promotions: treat the attacker as the promotion piece, not just a pawn
    if move.promotion is not None:
        attacker_type = move.promotion
    else:
        attacker_type = attacker_piece.piece_type

    return MVV_LVA_TABLE[victim_type][attacker_type]


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
# QUIESCENCE SEARCH
# ============================================================

def quiesce(board: chess.Board, alpha: float, beta: float) -> float:
    """
    Quiescence search:
    - Only search "tactical" moves (captures).
    - Use "stand pat" static eval as a lower bound.
    - Avoid horizon effect by not stopping in the middle of a capture sequence.
    """

    # If the game is already over, handle like in normal search
    if board.is_game_over():
        if board.is_checkmate():
            return -INFINITY + 1
        return 0

    # Stand pat: evaluate current (possibly non-quiet) position
    stand_pat = evaluate(board)

    # Fail-soft style
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    # Generate capture moves only
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        # No more captures: this is a "quiet enough" leaf
        return stand_pat

    # Order captures with MVV-LVA (we don't bother with TT move here)
    captures = sorted(
        captures,
        key=lambda m: mvv_lva_score(board, m),
        reverse=True
    )

    # Search all captures
    for move in captures:
        board.push(move)
        score = -quiesce(board, -beta, -alpha)
        board.pop()

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha


# ============================================================
# NULL MOVE PRUNING HELPERS
# ============================================================

NULL_MOVE_REDUCTION = 2  # R (depth reduction), classic values are 2 or 3


def has_non_pawn_material(board: chess.Board, color: bool) -> bool:
    """Return True if the given side has any piece > pawn (N,B,R,Q)."""
    for ptype in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if board.pieces(ptype, color):
            return True
    return False


def can_do_null_move(board: chess.Board) -> bool:
    """
    Basic conditions where we allow a null move:
    - Not in check
    - Side to move has some non-pawn material
    (to reduce zugzwang blowups in pawn-only endgames)
    """
    if board.is_check():
        return False

    color = board.turn
    if not has_non_pawn_material(board, color):
        return False

    return True


# ============================================================
# SEARCH (NEGAMAX + ALPHA-BETA + NULL MOVE)
# ============================================================

def negamax(board: chess.Board, depth: int, alpha: float, beta: float, allow_null: bool = True) -> float:
    """
    Negamax with alpha-beta pruning, TT, quiescence, and null move pruning.
    Returns a score from the perspective of the side to move.
    """

    # Terminal nodes
    if board.is_game_over():
        if board.is_checkmate():
            return -INFINITY + 1
        return 0

    # Leaf: switch to quiescence search instead of plain evaluate()
    if depth == 0:
        return quiesce(board, alpha, beta)

    alpha_orig = alpha

    # Try null move pruning (only if allowed and depth is large enough)
    if allow_null and depth >= 3 and can_do_null_move(board):
        # Make a null move (side passes the turn)
        board.push(chess.Move.null())

        null_depth = depth - 1 - NULL_MOVE_REDUCTION
        if null_depth < 0:
            null_depth = 0

        # Reduced, zero-width search
        score = -negamax(board, null_depth, -beta, -(beta - 1), allow_null=False)

        board.pop()

        if score >= beta:
            # Null move search says "this position is already so good
            # that even doing nothing fails high" => cut
            return score

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
        score = -negamax(board, depth - 1, -beta, -alpha, allow_null=True)
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

def root_search(board: chess.Board, max_depth: int, time_limit: float = 10.0) -> Move | None:
    """
    Root search with iterative deepening and time cutoff.
    Returns the best move found so far.
    """

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = random.choice(legal_moves)
    best_score = -INFINITY

    start_time = time.time()

    for depth in range(1, max_depth + 1):
        current_best_move = None
        current_best_score = -INFINITY

        # TT move if available
        tt_hit, _, tt_move = tt_probe(board, depth, -INFINITY, INFINITY)
        ordered_moves = order_moves(board, legal_moves, tt_move)

        alpha = -INFINITY
        beta = INFINITY

        for move in ordered_moves:
            # Time check at the root level
            if time.time() - start_time > time_limit:
                print(f"Time limit reached at depth {depth}")
                return best_move  # return the last fully completed depth

            board.push(move)
            score = -negamax(board, depth, -beta, -alpha, allow_null=True)
            board.pop()

            if current_best_move is None or score > current_best_score:
                current_best_score = score
                current_best_move = move

            if score > alpha:
                alpha = score

        # Update best move only if the whole depth finished
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

    # Search: minimax + alpha-beta + MVV-LVA + quiescence + null move pruning + CNN eval + iterative deepening (10s limit)
    best_move = root_search(board, MAX_DEPTH, MAX_TIME)

    if best_move is None or best_move not in board.legal_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (checkmate or stalemate).")
        best_move = random.choice(legal_moves)

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