import random

import chess
import numpy as np

from chessml.encoding import (
    NUM_MOVES,
    SCALE,
    decode_move,
    encode_move,
    featurize,
    featurize_int8,
    legal_mask,
    mirror_move,
)

EDGE_FENS = [
    "r1b1kbnr/pPpp1ppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 5",
    "rnbqkbnr/p1pppppp/8/8/8/8/PpPPPPP1/R1BQKBNR b KQkq - 0 5",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 12 45",
    "8/8/8/8/8/5k2/6p1/7K b - - 0 60",
]


def random_boards(games: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    for _ in range(games):
        board = chess.Board()
        for _ in range(rng.randrange(10, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def all_test_boards() -> list[chess.Board]:
    return [chess.Board(fen) for fen in EDGE_FENS] + random_boards(games=60, seed=7)


def test_encode_decode_roundtrip_every_legal_move() -> None:
    for board in all_test_boards():
        for move in board.legal_moves:
            idx = encode_move(move)
            assert 0 <= idx < NUM_MOVES, (board.fen(), move.uci())
            assert decode_move(idx, board) == move, (board.fen(), move.uci())


def test_roundtrip_through_the_rotated_frame() -> None:
    for board in all_test_boards():
        if board.turn == chess.WHITE:
            continue
        rotated = board.mirror()
        for move in board.legal_moves:
            rotated_move = mirror_move(move)
            idx = encode_move(rotated_move)
            assert idx >= 0, (board.fen(), move.uci())
            decoded = decode_move(idx, rotated)
            assert mirror_move(decoded) == move, (board.fen(), move.uci())


def test_mirror_featurize_parity() -> None:
    for board in all_test_boards():
        flat = chess.Board(board.fen())
        np.testing.assert_array_equal(featurize_int8(flat), featurize_int8(flat.mirror()))


def test_quantization_grid() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 7 42")
    x8 = featurize_int8(board)
    assert x8.min() >= 0 and x8.max() <= 2 * SCALE
    assert (x8[17] == 7 // 2).all()
    assert (x8[18] == 42 // 2).all()
    assert (x8[19] == SCALE).all()
    xf = featurize(board)
    np.testing.assert_array_equal(xf * SCALE, x8.astype(np.float32))
    assert xf.dtype == np.float32


def test_clock_planes_saturate() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 140 260")
    x8 = featurize_int8(board)
    assert (x8[17] == SCALE).all()
    assert (x8[18] == 2 * SCALE).all()


def test_repetition_plane_counts_history() -> None:
    board = chess.Board()
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    assert (featurize_int8(board)[20] == 0).all()
    for uci in shuffle:
        board.push_uci(uci)
    assert (featurize_int8(board)[20] == SCALE // 2).all()
    for uci in shuffle:
        board.push_uci(uci)
    assert (featurize_int8(board)[20] == SCALE).all()


def test_en_passant_plane_single_and_mirrored() -> None:
    white = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    x = featurize_int8(white)
    assert x[16].sum() == SCALE
    assert x[16, chess.square_rank(chess.F6), chess.square_file(chess.F6)] == SCALE

    black = chess.Board("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3")
    x = featurize_int8(black)
    assert x[16].sum() == SCALE
    assert x[16, chess.square_rank(chess.D6), chess.square_file(chess.D6)] == SCALE


def test_castling_planes_swap_for_black() -> None:
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Qk - 0 1")
    x = featurize_int8(board)
    assert (x[12] == SCALE).all()
    assert (x[13] == 0).all()
    assert (x[14] == 0).all()
    assert (x[15] == SCALE).all()


def test_legal_mask_is_exact_and_injective() -> None:
    for board in all_test_boards():
        legal = list(board.legal_moves)
        indices = {encode_move(m) for m in legal}
        assert -1 not in indices, board.fen()
        assert len(indices) == len(legal), board.fen()
        mask = legal_mask(board)
        assert int(mask.sum()) == len(legal), board.fen()
        assert all(mask[i] for i in indices), board.fen()


def test_queen_promotion_needs_pawn_on_board() -> None:
    board = chess.Board("r1b1kbnr/pPpp1ppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 5")
    promo = chess.Move.from_uci("b7a8q")
    assert decode_move(encode_move(promo), board) == promo
    rook_move = chess.Move.from_uci("a1b1")
    decoded = decode_move(encode_move(rook_move), board)
    assert decoded == rook_move and decoded.promotion is None


def test_underpromotions_roundtrip() -> None:
    board = chess.Board("r1b1kbnr/pPpp1ppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 5")
    for target in ("a8", "b8", "c8"):
        for piece in ("n", "b", "r"):
            move = chess.Move.from_uci(f"b7{target}{piece}")
            if move in board.legal_moves:
                assert decode_move(encode_move(move), board) == move


def test_en_passant_plane_matches_platform_fen() -> None:
    board = chess.Board()
    board.push_san("e4")
    assert board.ep_square is not None
    assert not board.has_legal_en_passant()
    assert featurize_int8(board)[16].sum() == 0
    for b in all_test_boards():
        assert (featurize_int8(chess.Board(b.fen())) == featurize_int8(b)).all(), b.fen()
