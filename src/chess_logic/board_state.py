"""Chess board state representation."""

from dataclasses import dataclass, field

import chess


@dataclass
class Move:
    """Represents a chess move with additional metadata."""

    uci: str
    san: str
    from_square: str
    to_square: str
    is_capture: bool
    is_check: bool
    is_checkmate: bool
    is_castling: bool
    promotion: str | None = None

    @classmethod
    def from_chess_move(cls, board: chess.Board, move: chess.Move) -> "Move":
        """Create Move from python-chess Move object."""
        san = board.san(move)

        return cls(
            uci=move.uci(),
            san=san,
            from_square=chess.square_name(move.from_square),
            to_square=chess.square_name(move.to_square),
            is_capture=board.is_capture(move),
            is_check=board.gives_check(move),
            is_checkmate="#" in san,
            is_castling=board.is_castling(move),
            promotion=chess.piece_name(move.promotion) if move.promotion else None,
        )


@dataclass
class BoardState:
    """Represents current state of the chess board."""

    fen: str
    board: chess.Board = field(default_factory=chess.Board)

    def __post_init__(self):
        if self.fen:
            self.board = chess.Board(self.fen)

    @classmethod
    def from_fen(cls, fen: str) -> "BoardState":
        """Create BoardState from FEN string."""
        return cls(fen=fen)

    @classmethod
    def starting_position(cls) -> "BoardState":
        """Create BoardState with starting position."""
        return cls(fen=chess.STARTING_FEN)

    @property
    def turn(self) -> str:
        """Get side to move ('white' or 'black')."""
        return "white" if self.board.turn else "black"

    @property
    def is_check(self) -> bool:
        """Check if current side is in check."""
        return self.board.is_check()

    @property
    def is_checkmate(self) -> bool:
        """Check if position is checkmate."""
        return self.board.is_checkmate()

    @property
    def is_stalemate(self) -> bool:
        """Check if position is stalemate."""
        return self.board.is_stalemate()

    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()

    @property
    def legal_moves(self) -> list[Move]:
        """Get all legal moves."""
        return [
            Move.from_chess_move(self.board, move)
            for move in self.board.legal_moves
        ]

    def get_piece_at(self, square: str) -> str | None:
        """Get piece at given square."""
        sq = chess.parse_square(square)
        piece = self.board.piece_at(sq)
        return piece.symbol() if piece else None

    def make_move(self, move: str) -> "BoardState":
        """Make a move and return new state."""
        board_copy = self.board.copy()

        try:
            chess_move = chess.Move.from_uci(move)
        except ValueError:
            chess_move = board_copy.parse_san(move)

        board_copy.push(chess_move)
        return BoardState(fen=board_copy.fen())

    def is_legal_move(self, move: str) -> bool:
        """Check if a move is legal."""
        try:
            try:
                chess_move = chess.Move.from_uci(move)
            except ValueError:
                chess_move = self.board.parse_san(move)
            return chess_move in self.board.legal_moves
        except Exception:
            return False

    def to_unicode(self) -> str:
        """Get Unicode representation of board."""
        return self.board.unicode()

    def to_ascii(self) -> str:
        """Get ASCII representation of board."""
        return str(self.board)

    def piece_map(self) -> dict[str, str]:
        """Get map of squares to pieces."""
        return {
            chess.square_name(sq): piece.symbol()
            for sq, piece in self.board.piece_map().items()
        }