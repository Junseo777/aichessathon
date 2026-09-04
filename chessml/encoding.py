import chess
import numpy as np
import numpy.typing as npt

NUM_PLANES = 21
SCALE = 50
NUM_MOVE_TYPES = 73
NUM_MOVES = 64 * NUM_MOVE_TYPES

_ONE = SCALE

_PIECE_PLANE = {
    chess.PAWN: 0,
    chess.ROOK: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def repetition_level(board: chess.Board) -> int:
    """How many times the position already occurred, capped at 2; plane 20 shows this."""
    if board.is_repetition(3):
        return 2
    if board.is_repetition(2):
        return 1
    return 0


def _clock_planes(board: chess.Board, x: npt.NDArray[np.int8]) -> None:
    x[17, :, :] = min(board.halfmove_clock, 100) // 2
    x[18, :, :] = min(board.fullmove_number, 200) // 2
    x[19, :, :] = _ONE
    x[20, :, :] = (_ONE // 2) * repetition_level(board)


def featurize_int8(board: chess.Board) -> npt.NDArray[np.int8]:
    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.int8)
    black = board.turn == chess.BLACK
    mover = chess.BLACK if black else chess.WHITE

    for square, piece in board.piece_map().items():
        h = chess.square_rank(square)
        w = chess.square_file(square)
        if black:
            h = 7 - h
        plane = _PIECE_PLANE[piece.piece_type] + (0 if piece.color == mover else 6)
        x[plane, h, w] = _ONE

    if board.has_kingside_castling_rights(mover):
        x[12, :, :] = _ONE
    if board.has_queenside_castling_rights(mover):
        x[13, :, :] = _ONE
    if board.has_kingside_castling_rights(not mover):
        x[14, :, :] = _ONE
    if board.has_queenside_castling_rights(not mover):
        x[15, :, :] = _ONE

    ep = board.ep_square
    if ep is not None and board.has_legal_en_passant():
        eh = chess.square_rank(ep)
        if black:
            eh = 7 - eh
        x[16, eh, chess.square_file(ep)] = _ONE

    _clock_planes(board, x)
    return x


def featurize(board: chess.Board) -> npt.NDArray[np.float32]:
    return featurize_int8(board).astype(np.float32) / np.float32(SCALE)


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


_SLIDE_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_KNIGHT_DIRS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
_UNDERPROM_FILES = [-1, 0, 1]
_UNDERPROM_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _build_tables() -> tuple[dict[tuple[int, int, int | None], int], npt.NDArray[np.int8]]:
    encode: dict[tuple[int, int, int | None], int] = {}
    decode = np.full((64, NUM_MOVE_TYPES, 2), -1, dtype=np.int8)

    for from_sq in range(64):
        fh, fw = chess.square_rank(from_sq), chess.square_file(from_sq)

        for d_idx, (dh, dw) in enumerate(_SLIDE_DIRS):
            for dist in range(1, 8):
                th, tw = fh + dh * dist, fw + dw * dist
                if not (0 <= th < 8 and 0 <= tw < 8):
                    break
                to_sq = th * 8 + tw
                mt = d_idx * 7 + (dist - 1)
                encode[(from_sq, to_sq, None)] = mt
                encode[(from_sq, to_sq, chess.QUEEN)] = mt
                decode[from_sq, mt] = (to_sq, -1)

        for k_idx, (dh, dw) in enumerate(_KNIGHT_DIRS):
            th, tw = fh + dh, fw + dw
            if 0 <= th < 8 and 0 <= tw < 8:
                to_sq = th * 8 + tw
                mt = 56 + k_idx
                encode[(from_sq, to_sq, None)] = mt
                decode[from_sq, mt] = (to_sq, -1)

        for dh in (1, -1):
            th = fh + dh
            if th not in (0, 7):
                continue
            for f_idx, df in enumerate(_UNDERPROM_FILES):
                tw = fw + df
                if not (0 <= tw < 8):
                    continue
                to_sq = th * 8 + tw
                for p_idx, piece in enumerate(_UNDERPROM_PIECES):
                    mt = 64 + f_idx * 3 + p_idx
                    encode[(from_sq, to_sq, piece)] = mt
                    decode[from_sq, mt] = (to_sq, piece)

    return encode, decode


_ENCODE_TABLE, _DECODE_TABLE = _build_tables()


def encode_move(move: chess.Move) -> int:
    mt = _ENCODE_TABLE.get((move.from_square, move.to_square, move.promotion), -1)
    return -1 if mt < 0 else move.from_square * NUM_MOVE_TYPES + mt


def decode_move(flat_idx: int, board: chess.Board) -> chess.Move:
    from_sq, mt = divmod(flat_idx, NUM_MOVE_TYPES)
    to_sq = int(_DECODE_TABLE[from_sq, mt, 0])
    if to_sq < 0:
        return chess.Move.null()
    prom = int(_DECODE_TABLE[from_sq, mt, 1])
    promotion: int | None = prom if prom > 0 else None
    if (
        promotion is None
        and chess.square_rank(to_sq) in (0, 7)
        and board.piece_type_at(from_sq) == chess.PAWN
    ):
        promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_mask(board: chess.Board) -> npt.NDArray[np.bool_]:
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        idx = encode_move(move)
        if idx >= 0:
            mask[idx] = True
    return mask


def transposition_key(board: chess.Board) -> object:
    return board._transposition_key()
