from math import inf

from referee.game import (
    Action,
    BOARD_N,
    Board,
    Coord,
    Direction,
    GamePhase,
    PlaceAction,
    PlayerColor,
)


CARDINAL_DIRECTIONS = (
    Direction.Up,
    Direction.Down,
    Direction.Left,
    Direction.Right,
)


# opening search parameters.
# placement phase has only eight total moves, so we can use a small controlled search.
# beam width keeps the opening search fast enough.
LOOKAHEAD_PLIES = 2
ROOT_BEAM_WIDTH = 16
SEARCH_BEAM_WIDTH = 10


# small tested coordinate preference for opening shape.
# this is not a hard opening book; it only gives bonus inside placement scoring.
OPENING_BOOK = {
    PlayerColor.RED: (
        Coord(3, 3),
        Coord(4, 3),
        Coord(3, 4),
        Coord(2, 4),
    ),
    PlayerColor.BLUE: (
        Coord(4, 4),
        Coord(3, 4),
        Coord(4, 3),
        Coord(5, 3),
    ),
}


# choose a placement action during the opening phase.
# the goal is to build a connected line / curve while limiting opponent space.
def choose_opening_action(
    board: Board,
    actions: list[Action],
    color: PlayerColor,
) -> Action:
    place_actions = [
        action for action in actions
        if isinstance(action, PlaceAction)
    ]

    # if something unexpected happens, still return a legal action
    if not place_actions:
        return actions[0]

    # first move is symmetric.
    # do not waste time searching empty-board positions.
    if (
        not _collect_coords(board, PlayerColor.RED)
        and not _collect_coords(board, PlayerColor.BLUE)
    ):
        preferred = Coord(3, 3) if color == PlayerColor.RED else Coord(4, 4)
        for action in place_actions:
            if action.coord == preferred:
                return action

    # after the first own placement, prefer cells close to our existing chain.
    # this makes the opening form a line or curve instead of scattered stacks.
    candidate_actions = _chain_filtered_actions(board, place_actions, color)

    scored: list[tuple[float, PlaceAction]] = []
    for action in candidate_actions:
        score = _placement_score(board, action.coord, color)
        scored.append((score, action))

    # keep only the best root candidates before doing lookahead
    scored.sort(key=lambda item: item[0], reverse=True)
    candidates = scored[:_dynamic_root_beam(board)]

    best_action = candidates[0][1]
    best_score = -inf

    # try each candidate and add a limited future placement search score
    for base_score, action in candidates:
        board.apply_action(action)
        try:
            future_score = _placement_search(
                board,
                color,
                _dynamic_opening_depth(board),
            )
        finally:
            board.undo_action()

        total_score = base_score + 0.85 * future_score

        if total_score > best_score:
            best_score = total_score
            best_action = action

    return best_action


# choose how far to look ahead in the placement phase.
# late placement has fewer legal moves, so it can search a bit deeper.
def _dynamic_opening_depth(board: Board) -> int:
    placed = (
        len(_collect_coords(board, PlayerColor.RED))
        + len(_collect_coords(board, PlayerColor.BLUE))
    )

    remaining = max(0, 8 - placed)

    # early / middle placement: depth 2 is usually enough
    if remaining >= 4:
        return 2

    # last placement decisions are cheaper and important
    return 3


# choose how many root placement candidates to keep.
# this controls the branching factor of opening search.
def _dynamic_root_beam(board: Board) -> int:
    placed = (
        len(_collect_coords(board, PlayerColor.RED))
        + len(_collect_coords(board, PlayerColor.BLUE))
    )

    if placed <= 2:
        return 12

    if placed <= 5:
        return 16

    return 20


# limited minimax-style search for placement phase.
# our placement maximises the score, and opponent placement minimises it.
def _placement_search(
    board: Board,
    perspective: PlayerColor,
    depth: int,
) -> float:
    if depth <= 0 or board.phase != GamePhase.PLACEMENT:
        return _opening_static_score(board, perspective)

    actions = _legal_place_actions(board)
    if not actions:
        return _opening_static_score(board, perspective)

    mover = board.turn_color
    actions = _chain_filtered_actions(board, actions, mover)

    scored: list[tuple[float, PlaceAction]] = []
    for action in actions:
        score = _placement_score(board, action.coord, mover)
        scored.append((score, action))

    scored.sort(key=lambda item: item[0], reverse=True)
    candidates = scored[:SEARCH_BEAM_WIDTH]

    # if it is our turn, choose the best future opening score
    if mover == perspective:
        best = -inf
        for _, action in candidates:
            board.apply_action(action)
            try:
                best = max(
                    best,
                    _placement_search(board, perspective, depth - 1),
                )
            finally:
                board.undo_action()
        return best

    # if it is opponent's turn, assume opponent chooses the worst score for us
    best = inf
    for _, action in candidates:
        board.apply_action(action)
        try:
            best = min(
                best,
                _placement_search(board, perspective, depth - 1),
            )
        finally:
            board.undo_action()

    return best


# prefer placement actions that stay close to our existing chain.
# this avoids isolated stacks and makes the opening easier to use later.
def _chain_filtered_actions(
    board: Board,
    actions: list[PlaceAction],
    color: PlayerColor,
) -> list[PlaceAction]:
    own_coords = _collect_coords(board, color)

    if not own_coords:
        return actions

    # main rule: new placement should be directly connected to the chain.
    close = [
        action for action in actions
        if _min_chebyshev(action.coord, own_coords) <= 1
    ]

    if close:
        return close

    # fallback: if opponent restrictions make direct connection impossible,
    # allow a one-cell gap but still avoid isolated placements.
    fallback = [
        action for action in actions
        if _min_chebyshev(action.coord, own_coords) <= 2
    ]

    if fallback:
        return fallback

    return actions


# score one possible placement cell.
# higher score means this cell fits our opening plan better.
def _placement_score(
    board: Board,
    coord: Coord,
    color: PlayerColor,
) -> float:
    own_coords = _collect_coords(board, color)
    opponent_coords = _collect_coords(board, color.opponent)
    own_count = len(own_coords)
    new_own_coords = own_coords + [coord]

    score = 0.0

    # small bonus for tested opening coordinates
    score += _book_bonus(coord, color, own_count)

    # centre control is still the foundation.
    score += 14.0 * _center_weight(coord)

    # the most important part: reduce opponent's legal placement space.
    score += 9.5 * _opponent_space_cut_score(board, coord, color)

    # make a tight line / curve. direct connection is strongly preferred.
    score += _chain_support_score(coord, own_coords)
    score += _line_or_curve_score(new_own_coords)

    # bias opponent future legal cells to one side.
    score += _side_compression_score(board, new_own_coords, color)

    # stand close to opponent without being illegally adjacent.
    score += _opponent_pressure_score(coord, color, opponent_coords)

    # avoid edge placement unless no better connected central move exists.
    score -= _edge_penalty(coord)

    # penalise isolated placement more strongly than old lineopen versions.
    score -= _isolation_penalty(coord, own_coords)

    # avoid making a pure 2x2 block. we want a line / curve, not a ball.
    score -= _over_compact_penalty(new_own_coords)

    return score


# evaluate the whole placement structure from one player's view.
# this is used at the leaf of the opening lookahead.
def _opening_static_score(board: Board, perspective: PlayerColor) -> float:
    own_coords = _collect_coords(board, perspective)
    opponent_coords = _collect_coords(board, perspective.opponent)

    own_value = _formation_value(board, own_coords, perspective)
    opponent_value = _formation_value(board, opponent_coords, perspective.opponent)

    own_space = len(_legal_cells_for_player(board, perspective))
    opponent_space = len(_legal_cells_for_player(board, perspective.opponent))

    space_value = 1.35 * (own_space - opponent_space)

    return own_value - 0.95 * opponent_value + space_value


# evaluate one side's opening formation.
# good formations are central, connected, and create pressure.
def _formation_value(
    board: Board,
    coords: list[Coord],
    color: PlayerColor,
) -> float:
    if not coords:
        return 0.0

    value = 0.0

    # central placement is safer and gives more future options
    for coord in coords:
        value += 11.0 * _center_weight(coord)
        value -= _edge_penalty(coord)

    value += _line_or_curve_score(coords)
    value += _connected_component_score(coords)
    value += 4.5 * _total_blocked_weight(board, coords)

    opponent_coords = _collect_coords(board, color.opponent)
    if opponent_coords:
        value += _formation_pressure_against(coords, opponent_coords, color)

    return value


# small bonus for coordinates from tested opening patterns.
# this is only a bias, not a hard opening-book move.
def _book_bonus(coord: Coord, color: PlayerColor, own_count: int) -> float:
    book = OPENING_BOOK[color]

    if own_count < len(book) and coord == book[own_count]:
        return 12.0

    if coord in book:
        return 5.0

    return 0.0


# estimate how much this placement reduces opponent legal space.
# denying central future cells is more valuable than denying edge cells.
def _opponent_space_cut_score(
    board: Board,
    coord: Coord,
    color: PlayerColor,
) -> float:
    occupied = _occupied_set(board)
    own_coords = _collect_coords(board, color)
    opponent = color.opponent

    before_cells = _legal_cells_for_player_from_sets(
        occupied,
        _collect_coords(board, opponent),
        own_coords,
    )

    occupied_after = set(occupied)
    occupied_after.add((coord.r, coord.c))
    own_after = own_coords + [coord]

    after_cells = _legal_cells_for_player_from_sets(
        occupied_after,
        _collect_coords(board, opponent),
        own_after,
    )

    before_set = {(cell.r, cell.c) for cell in before_cells}
    after_set = {(cell.r, cell.c) for cell in after_cells}
    removed = before_set - after_set

    score = 0.0

    # occupying a legal opponent cell also denies that cell directly
    if (coord.r, coord.c) in before_set:
        score += 0.85 * _center_weight(coord)

    # reward every opponent cell removed by the new adjacency restriction
    for r, c in removed:
        denied = Coord(r, c)
        score += 1.0 + 1.7 * _center_weight(denied)

    return score


# score how well a new placement connects to our existing chain.
# direct or diagonal support is strongly preferred.
def _chain_support_score(coord: Coord, own_coords: list[Coord]) -> float:
    if not own_coords:
        return 0.0

    score = 0.0
    min_cheb = _min_chebyshev(coord, own_coords)
    min_manhattan = min(_manhattan(coord, own) for own in own_coords)

    direct_neighbors = 0
    diagonal_neighbors = 0
    loose_neighbors = 0

    for own in own_coords:
        dr = abs(coord.r - own.r)
        dc = abs(coord.c - own.c)
        cheb = max(dr, dc)
        man = dr + dc

        if cheb == 1:
            if man == 1:
                direct_neighbors += 1
                score += 30.0
            else:
                diagonal_neighbors += 1
                score += 24.0
        elif cheb == 2:
            loose_neighbors += 1
            score += 4.0

    # very strong preference: every new drop should connect to the chain.
    if min_cheb <= 1:
        score += 46.0
    elif min_cheb == 2:
        score -= 32.0
    else:
        score -= 95.0

    # pure straight distance should not be too large either.
    if min_manhattan > 2:
        score -= 20.0 * (min_manhattan - 2)

    # some branching is fine, but too many adjacent contacts make a compact ball.
    if direct_neighbors + diagonal_neighbors >= 3:
        score -= 18.0

    score += 3.0 * loose_neighbors

    return score


# reward a useful line or curve shape.
# avoid turning the opening into a compact block.
def _line_or_curve_score(coords: list[Coord]) -> float:
    if len(coords) <= 1:
        return 0.0

    rows = [coord.r for coord in coords]
    cols = [coord.c for coord in coords]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)

    score = 0.0

    # reward a useful wall span.
    score += 9.0 * min(4, row_span + col_span)

    # reward row / column / diagonal-ish lines.
    if row_span == 0 and col_span >= 2:
        score += 30.0
    elif col_span == 0 and row_span >= 2:
        score += 30.0
    elif abs(row_span - col_span) <= 1 and row_span + col_span >= 2:
        score += 26.0

    score += _connected_component_score(coords)

    # penalise gaps in the chain.
    score -= 18.0 * _gap_count(coords)

    # avoid overly compact square formations.
    if len(coords) >= 4:
        if row_span <= 1 and col_span <= 1:
            score -= 55.0
        elif row_span <= 1 and col_span <= 2:
            score -= 18.0
        elif col_span <= 1 and row_span <= 2:
            score -= 18.0

    return score


# count how many pieces are separated from the chain.
# gaps are bad because the opening becomes harder to support later.
def _gap_count(coords: list[Coord]) -> int:
    if len(coords) <= 1:
        return 0

    gaps = 0

    for coord in coords:
        nearest = min(
            max(abs(coord.r - other.r), abs(coord.c - other.c))
            for other in coords
            if other != coord
        )

        if nearest > 1:
            gaps += 1

    return gaps


# reward if all own placements are connected as one component.
# disconnected formations are less useful for pressure and support.
def _connected_component_score(coords: list[Coord]) -> float:
    if len(coords) <= 1:
        return 0.0

    points = {(coord.r, coord.c) for coord in coords}
    start = next(iter(points))
    stack = [start]
    seen = {start}

    # simple flood-fill over Chebyshev-neighbour connections
    while stack:
        item = stack.pop()

        for other in points:
            if other in seen:
                continue

            if max(abs(item[0] - other[0]), abs(item[1] - other[1])) <= 1:
                seen.add(other)
                stack.append(other)

    if len(seen) == len(points):
        return 34.0 + 5.0 * len(points)

    missing = len(points) - len(seen)
    return -42.0 * missing


# estimate whether our chain pushes opponent options to one side.
# this helps create space pressure before the play phase.
def _side_compression_score(
    board: Board,
    own_after: list[Coord],
    color: PlayerColor,
) -> float:
    if not own_after:
        return 0.0

    occupied = _occupied_set(board)
    opponent_coords = _collect_coords(board, color.opponent)
    legal_for_opp = _legal_cells_for_player_from_sets(
        occupied,
        opponent_coords,
        own_after,
    )

    if len(legal_for_opp) <= 4:
        return 0.0

    center_r = sum(coord.r for coord in own_after) / len(own_after)
    center_c = sum(coord.c for coord in own_after) / len(own_after)

    left = sum(1 for cell in legal_for_opp if cell.c < center_c)
    right = sum(1 for cell in legal_for_opp if cell.c > center_c)
    up = sum(1 for cell in legal_for_opp if cell.r < center_r)
    down = sum(1 for cell in legal_for_opp if cell.r > center_r)

    horizontal_bias = abs(left - right) / max(1, left + right)
    vertical_bias = abs(up - down) / max(1, up + down)

    central_opp_cells = sum(
        1
        for cell in legal_for_opp
        if _center_weight(cell) >= 0.72
    )

    score = 26.0 * max(horizontal_bias, vertical_bias)
    score -= 2.1 * central_opp_cells

    return score


# score how well a placement pressures opponent stacks.
# distance two or three is useful without being illegally adjacent.
def _opponent_pressure_score(
    coord: Coord,
    color: PlayerColor,
    opponent_coords: list[Coord],
) -> float:
    if not opponent_coords:
        return 0.0

    min_distance = min(_manhattan(coord, opp) for opp in opponent_coords)
    score = 0.0

    if min_distance == 2:
        score += 30.0
    elif min_distance == 3:
        score += 20.0
    elif min_distance == 4:
        score += 7.0
    elif min_distance >= 6:
        score -= 22.0

    # row or column pressure is slightly better because cascade / movement lines matter
    for opp in opponent_coords:
        distance = _manhattan(coord, opp)
        if distance < 2 or distance > 4:
            continue

        if coord.r == opp.r or coord.c == opp.c:
            score += 8.0 * (5 - distance)

    if color == PlayerColor.RED:
        score *= 1.12

    return score


# estimate pressure created by an existing formation.
# this is used when evaluating full opening structures.
def _formation_pressure_against(
    coords: list[Coord],
    opponent_coords: list[Coord],
    color: PlayerColor,
) -> float:
    if not coords or not opponent_coords:
        return 0.0

    value = 0.0

    for coord in coords:
        min_distance = min(_manhattan(coord, opp) for opp in opponent_coords)

        if min_distance == 2:
            value += 12.0
        elif min_distance == 3:
            value += 8.0
        elif min_distance >= 6:
            value -= 7.0

    own_r = sum(coord.r for coord in coords) / len(coords)
    own_c = sum(coord.c for coord in coords) / len(coords)
    opp_r = sum(coord.r for coord in opponent_coords) / len(opponent_coords)
    opp_c = sum(coord.c for coord in opponent_coords) / len(opponent_coords)

    separation = abs(own_r - opp_r) + abs(own_c - opp_c)

    if 2.0 <= separation <= 5.0:
        value += 12.0

    if color == PlayerColor.RED:
        value *= 1.07

    return value


# measure how valuable the cells blocked by our formation are.
# blocking central cells is more valuable than blocking edge cells.
def _total_blocked_weight(
    board: Board,
    own_coords: list[Coord],
) -> float:
    occupied = _occupied_set(board)
    blocked: set[tuple[int, int]] = set()

    for coord in own_coords:
        for direction in CARDINAL_DIRECTIONS:
            neighbor = _offset(coord, direction)
            if neighbor is None:
                continue

            key = (neighbor.r, neighbor.c)
            if key in occupied:
                continue

            blocked.add(key)

    total = 0.0
    for r, c in blocked:
        total += _center_weight(Coord(r, c))

    return total


# return legal future placement cells for one player.
# this is used to compare our space and opponent space.
def _legal_cells_for_player(board: Board, color: PlayerColor) -> list[Coord]:
    occupied = _occupied_set(board)
    own_coords = _collect_coords(board, color)
    opponent_coords = _collect_coords(board, color.opponent)

    return _legal_cells_for_player_from_sets(
        occupied,
        own_coords,
        opponent_coords,
    )


# same as legal placement cells, but using precomputed occupied sets.
# this avoids repeatedly scanning and rebuilding the same information.
def _legal_cells_for_player_from_sets(
    occupied: set[tuple[int, int]],
    own_coords: list[Coord],
    opponent_coords: list[Coord],
) -> list[Coord]:
    cells: list[Coord] = []
    opponent_set = {(coord.r, coord.c) for coord in opponent_coords}

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if (r, c) in occupied:
                continue

            coord = Coord(r, c)
            if _is_adjacent_to_any(coord, opponent_set):
                continue

            cells.append(coord)

    return cells


# generate legal placement actions for the current player.
# this local version is used by opening lookahead.
def _legal_place_actions(board: Board) -> list[PlaceAction]:
    actions: list[PlaceAction] = []
    color = board.turn_color
    opponent_coords = _collect_coords(board, color.opponent)
    opponent_set = {(coord.r, coord.c) for coord in opponent_coords}

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)

            if not board[coord].is_empty:
                continue

            if board.turn_count > 0 and _is_adjacent_to_any(coord, opponent_set):
                continue

            actions.append(PlaceAction(coord))

    return actions


# return how central a coordinate is.
# centre cells are usually safer and more flexible.
def _center_weight(coord: Coord) -> float:
    center = (BOARD_N - 1) / 2
    distance = abs(coord.r - center) + abs(coord.c - center)
    return max(0.0, (7.0 - distance) / 7.0)


# penalise placement near the board edge.
# edge placements have less mobility and higher cascade risk later.
def _edge_penalty(coord: Coord) -> float:
    edge_dist = min(
        coord.r,
        coord.c,
        BOARD_N - 1 - coord.r,
        BOARD_N - 1 - coord.c,
    )

    if edge_dist == 0:
        return 38.0

    if edge_dist == 1:
        return 12.0

    return 0.0


# penalise placements that are too far from our existing stacks.
# isolated opening pieces usually do not support the line plan.
def _isolation_penalty(coord: Coord, own_coords: list[Coord]) -> float:
    if not own_coords:
        return 0.0

    min_cheb = _min_chebyshev(coord, own_coords)

    if min_cheb <= 1:
        return 0.0

    if min_cheb == 2:
        return 36.0

    return 120.0


# penalise formations that are too compact.
# a line or curve is preferred over a small block.
def _over_compact_penalty(coords: list[Coord]) -> float:
    if len(coords) < 3:
        return 0.0

    rows = [coord.r for coord in coords]
    cols = [coord.c for coord in coords]

    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)

    if len(coords) >= 4 and row_span <= 1 and col_span <= 1:
        return 60.0

    if len(coords) >= 4 and row_span + col_span <= 2:
        return 26.0

    return 0.0


# collect all coordinates occupied by one colour.
# this helper is reused by most opening scoring functions.
def _collect_coords(board: Board, color: PlayerColor) -> list[Coord]:
    coords: list[Coord] = []

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            if board[coord].color == color:
                coords.append(coord)

    return coords


# collect all occupied cells as simple row-column pairs.
# this makes placement-space calculations cheaper.
def _occupied_set(board: Board) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if not board[Coord(r, c)].is_empty:
                occupied.add((r, c))

    return occupied


# check whether a coordinate is adjacent to any coordinate in a set.
# this implements the placement restriction.
def _is_adjacent_to_any(
    coord: Coord,
    coord_set: set[tuple[int, int]],
) -> bool:
    for direction in CARDINAL_DIRECTIONS:
        neighbor = _offset(coord, direction)
        if neighbor is None:
            continue

        if (neighbor.r, neighbor.c) in coord_set:
            return True

    return False


# return one neighbouring coordinate.
# return None if the neighbour is outside the board.
def _offset(coord: Coord, direction: Direction) -> Coord | None:
    r = coord.r + direction.r
    c = coord.c + direction.c

    if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
        return None

    return Coord(r, c)


# return the closest Chebyshev distance to a list of coordinates.
# this is useful for checking direct chain connection.
def _min_chebyshev(coord: Coord, coords: list[Coord]) -> int:
    return min(
        max(abs(coord.r - other.r), abs(coord.c - other.c))
        for other in coords
    )


# Manhattan distance between two coordinates.
def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)