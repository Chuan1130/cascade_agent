from referee.game import (
    BOARD_N,
    Action,
    Board,
    CascadeAction,
    Coord,
    Direction,
    EatAction,
    GamePhase,
    MoveAction,
    PlaceAction,
)


CARDINAL_DIRECTIONS = (
    Direction.Up,
    Direction.Down,
    Direction.Left,
    Direction.Right,
)


# generate all legal actions for the current board.
# this is used by opening, greedy search, and negamax.
# actions are generated directly from the rules, so we do not need to
# apply and undo every possible action just to check legality.
def legal_actions(board: Board) -> list[Action]:
    if board.phase == GamePhase.PLACEMENT:
        return _legal_place_actions(board)

    return _legal_play_actions(board)


# generate legal PLACE actions during the placement phase.
# after the first placement, a new stack cannot be adjacent to opponent stacks.
def _legal_place_actions(board: Board) -> list[Action]:
    actions: list[Action] = []
    avoid_opponent_adjacency = board.turn_count > 0
    opponent = board.turn_color.opponent

    # scan every board cell and keep the empty cells that satisfy placement rules
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)

            # cannot place on an occupied cell
            if not board[coord].is_empty:
                continue

            # after the first placement, avoid cells adjacent to opponent
            if (
                avoid_opponent_adjacency
                and _is_adjacent_to_color(board, coord, opponent)
            ):
                continue

            actions.append(PlaceAction(coord))

    return actions


# generate legal MOVE, EAT, and CASCADE actions during play phase.
# this follows the game rules from the current board state.
def _legal_play_actions(board: Board) -> list[Action]:
    actions: list[Action] = []
    color = board.turn_color

    # scan every cell and only generate actions from current player's stacks
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color != color:
                continue

            # MOVE and EAT only look at adjacent cardinal cells
            for direction in CARDINAL_DIRECTIONS:
                target_coord = _offset(coord, direction)
                if target_coord is None:
                    continue

                target = board[target_coord]

                # MOVE is legal if the target is empty or friendly
                if target.is_empty or target.color == color:
                    actions.append(MoveAction(coord, direction))

                # EAT is legal if the target is enemy and not taller than us
                elif cell.height >= target.height:
                    actions.append(EatAction(coord, direction))

            # CASCADE is legal for any stack with height at least 2
            if cell.height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))

    return actions


# check whether a coordinate is next to a stack of one colour.
# this is mainly used for placement legality.
def _is_adjacent_to_color(board: Board, coord: Coord, color) -> bool:
    for direction in CARDINAL_DIRECTIONS:
        neighbor = _offset(coord, direction)

        if neighbor is not None and board[neighbor].color == color:
            return True

    return False


# return the next coordinate in one direction.
# return None if the target cell is outside the board.
def _offset(coord: Coord, direction: Direction) -> Coord | None:
    r = coord.r + direction.r
    c = coord.c + direction.c

    if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
        return None

    return Coord(r, c)