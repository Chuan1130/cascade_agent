"""
Small source-grounded Cascade demonstration, not a historical match replay.

Run from the repository root:
    python -m examples.cascade_decision_demo

This constructs an illustrative play-phase state, prints the actual
referee board, demonstrates one legal CASCADE action, and asks the
implemented decision policy what action it would choose.
"""

from referee.game import (
    Board,
    CascadeAction,
    CellState,
    Coord,
    Direction,
    PlayerColor,
)
from agent.movegen import legal_actions
from agent.clean_search import choose_search_action


def make_board() -> Board:
    red = PlayerColor.RED
    blue = PlayerColor.BLUE
    cells = {
        Coord(3, 2): CellState(red, 3),
        Coord(3, 4): CellState(blue, 2),
        Coord(1, 1): CellState(red, 2),
        Coord(6, 1): CellState(red, 3),
        Coord(1, 6): CellState(blue, 3),
        Coord(5, 5): CellState(blue, 3),
        Coord(6, 6): CellState(blue, 1),
    }

    board = Board(initial_state=cells, initial_player=red)
    # Board(initial_state=...) counts occupied cells as opening placements.
    # This is a constructed mid-game example, so switch to PLAY explicitly.
    board.set_placement_count(8)
    return board


def main() -> None:
    board = make_board()
    print("EXAMPLE: original board")
    print(board.render())

    moves = legal_actions(board)
    cascade = CascadeAction(Coord(3, 2), Direction.Right)
    assert cascade in moves, "Expected illustrative cascade to be legal"

    print("Illustrated legal action:", cascade)
    board.apply_action(cascade)
    print("EXAMPLE: after CASCADE")
    print(board.render())

    # The blue stack on (3,4) is pushed twice to (3,6).
    assert board[Coord(3, 6)] == CellState(PlayerColor.BLUE, 2)
    assert board[Coord(3, 3)] == CellState(PlayerColor.RED, 1)
    assert board[Coord(3, 4)] == CellState(PlayerColor.RED, 1)
    assert board[Coord(3, 5)] == CellState(PlayerColor.RED, 1)

    board.undo_action()
    print("Undo restores original position:", board[Coord(3, 2)])

    # Actual policy result is computed by the repository's implementation.
    # This may differ from the illustrative cascade; no choice is hardcoded.
    chosen = choose_search_action(
        board, legal_actions(board), PlayerColor.RED, time_remaining=10.0
    )
    print("Agent policy chooses:", chosen)


if __name__ == "__main__":
    main()
