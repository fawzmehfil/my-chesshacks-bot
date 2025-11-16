# # evaluation.py
# import torch
# import numpy as np
# from .chessCNN import ChessCNN
# import chess
# import os


# # from .data_ingestion import board_to_tensor_coords

# def board_to_tensor_coords(board: chess.Board):
#     """
#     Convert a chess.Board into a sparse list of coordinates
#     representing ones in a 12x8x8 tensor.
#     Returns a list of (channel, row, col) tuples.
#     """
#     ones_coords = []
#     for square, piece in board.piece_map().items():
#         row = 7 - (square // 8)  # invert rank so row 0 is bottom
#         col = square % 8
#         channel = piece.piece_type - 1
#         if piece.color == chess.BLACK:
#             channel += 6
#         ones_coords.append((channel, row, col))
#     return ones_coords


# class CNNEvaluator:
#     """Wraps your CNN for evaluation of python-chess Board objects."""
#     def __init__(self, model_path="final_cnn.pth"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = ChessCNN().to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#         print(f"Using device: {self.device}")

#     def evaluate_board(self, board) -> float:
#         tensor = board_to_tensor_coords(board).to(self.device)
#         with torch.no_grad():
#             return self.model(tensor).item()
# evaluation.py
import torch
import numpy as np
from .chessCNN import ChessCNN
import chess
import os

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board into a 12x8x8 tensor of 0s and 1s.
    Channels 0-5 = white pieces, 6-11 = black pieces
    """
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)  # rank 0 = bottom
        col = square % 8
        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6
        tensor[channel, row, col] = 1.0

    # Add batch dimension: (1, 12, 8, 8)
    return tensor.unsqueeze(0)


class CNNEvaluator:
    """Wraps ChessCNN for evaluating python-chess Board objects."""
    def __init__(self, model_path="final_cnn.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Using device: {self.device}")

    def evaluate_board(self, board: chess.Board) -> float:
        tensor = board_to_tensor(board).to(self.device)
        with torch.no_grad():
            return self.model(tensor).item()
