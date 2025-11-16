import pymongo
import os
import sys
from typing import Optional
import numpy as np
import chess
import chess.pgn
import chess.engine
import requests
import json
import io

myclient = pymongo.MongoClient("mongodb+srv://22nwd3_db_user:<passwrod>@lichessgames.c2mdjob.mongodb.net/?appName=lichessGames")
mydb = myclient["lichessGames"]
collection = mydb['mack']

collection.create_index("fen", unique=True)

stockfish_path = "/Users/mackrabeau/Documents/stockfish/stockfish-macos-x86-64-bmi2"

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

def board_to_tensor_coords(board: chess.Board):
    """
    Convert a chess.Board into a sparse list of coordinates
    representing ones in a 12x8x8 tensor.
    Returns a list of (channel, row, col) tuples.
    """
    ones_coords = []
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)  # invert rank so row 0 is bottom
        col = square % 8
        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6
        ones_coords.append((channel, row, col))
    return ones_coords


username = "dinnerbone123" 
url = f"https://lichess.org/api/games/user/{username}"


params = {
    'max': 100,
    'perfType': 'bullet',
    "pgnInJson": "true",
}

response = requests.get(
    url, 
    params=params, 
    headers = {"Accept": "application/x-ndjson"}
)


batch = []
game_count = 0
position_count = 0
batch_size = 100

for line in response.iter_lines():
    if not line:
        continue

    game_json = json.loads(line.decode("utf-8"))
    pgn_text = game_json.get("pgn")

    if pgn_text is None:
        continue
    
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        continue

    headers = game.headers

    # ----- filter elo -----
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        continue

    # if white_elo < 2000 or black_elo < 2000:
    #     continue  # skip low-elo games

    board = game.board()

    for move in game.mainline_moves():
        board.push(move)

        fen = board.fen()

        # Skip positions that already exist
        if collection.count_documents({"fen": fen}, limit=1) > 0:
            continue

        # --- Evaluate position with Stockfish ---
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        score = info["score"].white()  # always relative to White

        if score.is_mate():
            eval_score = None
            mate = True
        else:
            eval_score = score.score()  # centipawns
            mate = False
        depth = info.get("depth", None)

        doc = {
            "fen": fen,
            "tensor": board_to_tensor_coords(board),
            "eval": eval_score,      # placeholder, fill after Stockfish eval
            "depth": depth,
            "mate": mate
        }

        batch.append(doc)
        position_count += 1

        if len(batch) >= batch_size:
            try:
                collection.insert_many(batch, ordered=False)
            except Exception as e:
                print("Insert warning:", e)
            print(f"Saved batch of {len(batch)} positions. Total = {position_count}")
            batch = []

    game_count += 1
    print(f"Processed game #{game_count}")


# Final flush
if batch:
    try:
        collection.insert_many(batch, ordered=False)
    except Exception as e:
        print("Insert warning:", e)
    print(f"Saved final batch of {len(batch)} positions.")

print("Done!")
print(f"Total games processed: {game_count}")
print(f"Total positions inserted: {position_count}")

engine.quit()