from dataclasses import dataclass

from referee.game import (
    BOARD_N,
    Action,
    Board,
    CascadeAction,
    Coord,
    EatAction,
    PlayerColor,
)

from .movegen import legal_actions
from .negamax import choose_search_action as choose_negamax_action


# clean midgame wrapper.
# It does not try to replace negamax with another decision system.
# It only removes obvious self-damaging root moves, then lets negamax decide.
def choose_search_action(
    board: Board,
    actions: list[Action],
    my_color: PlayerColor,
    time_remaining: float | None = None,
) -> Action:
    safe_actions = clean_safe_root_actions(board, actions, my_color)

    # if we are already clearly ahead, do not add a new search algorithm.
    # just order root actions in a more active way, then let negamax search normally.
    ordered_actions = _conversion_order_actions(
        board,
        safe_actions,
        my_color,
    )

    return choose_negamax_action(
        board,
        ordered_actions,
        my_color,
        time_remaining,
    )


# hard root filter.
# first keep immediate wins.
# then remove immediate losses and obvious tower-self-damage.
def clean_safe_root_actions(
    board: Board,
    actions: list[Action],
    my_color: PlayerColor,
) -> list[Action]:
    if len(actions) <= 1:
        return actions

    winning_actions: list[Action] = []
    for action in actions:
        if _action_wins_immediately(board, action, my_color):
            winning_actions.append(action)

    if winning_actions:
        return winning_actions

    filtered: list[Action] = []

    for action in actions:
        if _is_hard_bad_root_action(board, action, my_color):
            continue

        filtered.append(action)

    # do not accidentally remove every legal action.
    # if everything looks bad, let negamax pick the least bad one.
    if filtered:
        return filtered

    return actions


# check whether one root action is clearly bad.
# this function is conservative: it only blocks obvious tactical blunders.
def _is_hard_bad_root_action(
    board: Board,
    action: Action,
    my_color: PlayerColor,
) -> bool:
    before = _snapshot(board, my_color)
    source_height = _source_height(board, action, my_color)
    source_edge_distance = _source_edge_distance(board, action, my_color)

    board.apply_action(action)
    try:
        # if this action ends the game, only block it when we are not the winner
        if board.game_over:
            return board.winner_color != my_color

        after = _snapshot(board, my_color)

        # do not allow a move if opponent can immediately win next turn
        if _opponent_can_win_next(board, my_color):
            return True

        # block cascade that breaks our main structure without material gain
        if _is_no_gain_high_cascade(
            action,
            before,
            after,
            source_height,
            source_edge_distance,
            board,
            my_color,
        ):
            return True

        # block actions that give away our largest stack too cheaply
        if _gives_main_stack_to_opponent(
            board,
            before,
            after,
            my_color,
        ):
            return True

        # block actions that let opponent push our largest stack off board
        if _lets_opponent_push_main_stack(
            board,
            before,
            after,
            my_color,
        ):
            return True

        return False

    finally:
        board.undo_action()


# block cascade that splits an important tower without gaining material.
# this is the main fix for the old "throwing advantage" problem.
def _is_no_gain_high_cascade(
    action: Action,
    before: "Snapshot",
    after: "Snapshot",
    source_height: int,
    source_edge_distance: int,
    board_after_action: Board,
    my_color: PlayerColor,
) -> bool:
    if not isinstance(action, CascadeAction):
        return False

    if source_height < 3:
        return False

    enemy_removed = max(0, before.enemy_tokens - after.enemy_tokens)
    own_lost = max(0, before.own_tokens - after.own_tokens)
    largest_drop = max(0, before.largest - after.largest)
    extra_singletons = max(0, after.singletons - before.singletons)

    # if the cascade actually wins material, let negamax evaluate it.
    if enemy_removed > 0:
        return False

    # R6/R9 no-gain cascade is almost always how we throw away advantage.
    if source_height >= 6:
        return True

    # R4+ splitting into many R1s with no enemy removal is bad.
    if source_height >= 4 and largest_drop >= 2:
        return True

    # creating too many new singletons usually means the tower became fragile
    if source_height >= 4 and extra_singletons >= 2:
        return True

    # R3 near edge is also dangerous if it creates exposed fragments.
    if source_height >= 3 and source_edge_distance <= 1:
        if extra_singletons >= 1 or _opponent_can_eat_any_own_stack(board_after_action, my_color):
            return True

    # if we cascade and opponent can immediately eat one of our pieces,
    # while we removed nothing, this is usually self-damage.
    if source_height >= 3 and _opponent_can_eat_any_own_stack(board_after_action, my_color):
        return True

    # losing our own tokens with no enemy removal is hard-bad.
    if own_lost > 0:
        return True

    return False


# protect the largest stack.
# if a move lets opponent immediately eat our main tower without enough
# compensation, block it.
def _gives_main_stack_to_opponent(
    board_after_action: Board,
    before: "Snapshot",
    after: "Snapshot",
    my_color: PlayerColor,
) -> bool:
    if after.largest < 3:
        return False

    enemy_removed = max(0, before.enemy_tokens - after.enemy_tokens)
    own_removed = max(0, before.own_tokens - after.own_tokens)
    net_gain = enemy_removed - own_removed

    eaten_height = _opponent_can_eat_largest_stack_height(
        board_after_action,
        my_color,
    )

    if eaten_height <= 0:
        return False

    # do not give away R4+ unless the current move already won a lot.
    if eaten_height >= 4 and net_gain < eaten_height - 1:
        return True

    # R3 can be sacrificed only when we gained material first.
    if eaten_height >= 3 and net_gain <= 0:
        return True

    return False


# protect the largest stack from immediate enemy cascade push.
# this targets positions where a large stack is aligned with an enemy tower.
def _lets_opponent_push_main_stack(
    board_after_action: Board,
    before: "Snapshot",
    after: "Snapshot",
    my_color: PlayerColor,
) -> bool:
    if after.largest < 5:
        return False

    enemy_removed = max(0, before.enemy_tokens - after.enemy_tokens)
    own_removed = max(0, before.own_tokens - after.own_tokens)
    net_gain = enemy_removed - own_removed

    danger = _opponent_can_push_largest_stack_off(
        board_after_action,
        my_color,
    )

    if not danger:
        return False

    # if we just won significant material, allow negamax to judge.
    if net_gain >= 3:
        return False

    return True


# test whether one action wins the game immediately.
# immediate wins should always be kept before any safety filtering.
def _action_wins_immediately(
    board: Board,
    action: Action,
    my_color: PlayerColor,
) -> bool:
    board.apply_action(action)
    try:
        return board.game_over and board.winner_color == my_color
    finally:
        board.undo_action()


# check whether opponent has an immediate winning reply.
# this catches both EAT and CASCADE terminal threats.
def _opponent_can_win_next(board: Board, my_color: PlayerColor) -> bool:
    if board.turn_color != my_color.opponent:
        return False

    for reply in legal_actions(board):
        board.apply_action(reply)
        try:
            if board.game_over and board.winner_color == my_color.opponent:
                return True
        finally:
            board.undo_action()

    return False


# check whether opponent can eat any of our stacks next turn.
# this is used to detect exposed fragments after a cascade.
def _opponent_can_eat_any_own_stack(
    board: Board,
    my_color: PlayerColor,
) -> bool:
    if board.turn_color != my_color.opponent:
        return False

    for reply in legal_actions(board):
        if not isinstance(reply, EatAction):
            continue

        source = board[reply.coord]
        target_coord = _target_coord(reply.coord, reply.direction)

        if target_coord is None:
            continue

        target = board[target_coord]

        if source.color == my_color.opponent and target.color == my_color:
            return True

    return False


# return the height of our largest stack if opponent can eat it.
# return 0 if the largest stack is currently safe from immediate eat.
def _opponent_can_eat_largest_stack_height(
    board: Board,
    my_color: PlayerColor,
) -> int:
    if board.turn_color != my_color.opponent:
        return 0

    largest = _largest_stack_height(board, my_color)
    if largest <= 0:
        return 0

    for reply in legal_actions(board):
        if not isinstance(reply, EatAction):
            continue

        source = board[reply.coord]
        target_coord = _target_coord(reply.coord, reply.direction)

        if target_coord is None:
            continue

        target = board[target_coord]

        if (
            source.color == my_color.opponent
            and target.color == my_color
            and target.height == largest
        ):
            return target.height

    return 0


# check whether opponent can cascade our largest stack off the board.
# only direct row or column cascade lines are considered here.
def _opponent_can_push_largest_stack_off(
    board: Board,
    my_color: PlayerColor,
) -> bool:
    largest_coord, largest_height = _largest_stack_info(board, my_color)

    if largest_coord is None or largest_height < 5:
        return False

    opponent = my_color.opponent

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            enemy_coord = Coord(r, c)
            enemy = board[enemy_coord]

            if enemy.color != opponent:
                continue

            if enemy.height < 2:
                continue

            direction = _direction_to_target(enemy_coord, largest_coord)
            if direction is None:
                continue

            distance = _manhattan(enemy_coord, largest_coord)
            if distance > enemy.height:
                continue

            pushes = enemy.height - distance + 1
            edge_distance = _distance_to_edge_in_direction(
                largest_coord,
                direction,
            )

            if pushes > edge_distance:
                return True

    return False


# small summary of one side's material and shape.
# this avoids recalculating common root-safety values manually.
@dataclass(frozen=True)
class Snapshot:
    own_tokens: int
    enemy_tokens: int
    own_stacks: int
    enemy_stacks: int
    largest: int
    singletons: int


# collect a simple material and structure snapshot.
# it is used before and after simulating a root action.
def _snapshot(board: Board, my_color: PlayerColor) -> Snapshot:
    own_tokens = 0
    enemy_tokens = 0
    own_stacks = 0
    enemy_stacks = 0
    largest = 0
    singletons = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == my_color:
                own_tokens += cell.height
                own_stacks += 1
                largest = max(largest, cell.height)

                if cell.height == 1:
                    singletons += 1

            elif cell.color == my_color.opponent:
                enemy_tokens += cell.height
                enemy_stacks += 1

    return Snapshot(
        own_tokens=own_tokens,
        enemy_tokens=enemy_tokens,
        own_stacks=own_stacks,
        enemy_stacks=enemy_stacks,
        largest=largest,
        singletons=singletons,
    )


# return the source height for a cascade action.
# non-cascade actions do not have a source cascade height.
def _source_height(board: Board, action: Action, my_color: PlayerColor) -> int:
    if not isinstance(action, CascadeAction):
        return 0

    source = board[action.coord]
    if source.color != my_color:
        return 0

    return source.height


# return how close a cascade source is to the board edge.
# edge cascades are more likely to lose tokens or create weak fragments.
def _source_edge_distance(
    board: Board,
    action: Action,
    my_color: PlayerColor,
) -> int:
    if not isinstance(action, CascadeAction):
        return BOARD_N

    source = board[action.coord]
    if source.color != my_color:
        return BOARD_N

    return min(
        action.coord.r,
        action.coord.c,
        BOARD_N - 1 - action.coord.r,
        BOARD_N - 1 - action.coord.c,
    )


# return the largest stack height for one colour.
def _largest_stack_height(board: Board, color: PlayerColor) -> int:
    largest = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color:
                largest = max(largest, cell.height)

    return largest


# return both coordinate and height of the largest stack.
# this is used by push-off danger checks.
def _largest_stack_info(
    board: Board,
    color: PlayerColor,
) -> tuple[Coord | None, int]:
    best_coord = None
    best_height = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color == color and cell.height > best_height:
                best_coord = coord
                best_height = cell.height

    return best_coord, best_height


# return the adjacent target coordinate of an action.
# return None if the target is outside the board.
def _target_coord(coord: Coord, direction) -> Coord | None:
    r = coord.r + direction.r
    c = coord.c + direction.c

    if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
        return None

    return Coord(r, c)


# return a simple cardinal direction from source to target.
# only row-aligned or column-aligned targets can be reached by cascade.
def _direction_to_target(source: Coord, target: Coord):
    dr = target.r - source.r
    dc = target.c - source.c

    if dr == 0 and dc == 0:
        return None

    if dr == 0:
        if dc > 0:
            return _DirectionLike(0, 1)
        return _DirectionLike(0, -1)

    if dc == 0:
        if dr > 0:
            return _DirectionLike(1, 0)
        return _DirectionLike(-1, 0)

    return None


# small direction-like object used by local cascade checks.
# it only needs r and c fields.
@dataclass(frozen=True)
class _DirectionLike:
    r: int
    c: int


# count how many cells remain before the board edge in one direction.
# used to decide whether a cascade can push a stack off board.
def _distance_to_edge_in_direction(coord: Coord, direction) -> int:
    if direction.r == -1:
        return coord.r

    if direction.r == 1:
        return BOARD_N - 1 - coord.r

    if direction.c == -1:
        return coord.c

    if direction.c == 1:
        return BOARD_N - 1 - coord.c

    return BOARD_N


# Manhattan distance between two coordinates.
def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)


# winning conversion bias.
# this is not a new tactical system.
# it only reorders root actions when we are clearly ahead, so negamax breaks
# close choices toward finishing the game instead of just staying safe.
def _conversion_order_actions(
    board: Board,
    actions: list[Action],
    my_color: PlayerColor,
) -> list[Action]:
    if len(actions) <= 1:
        return actions

    before = _snapshot(board, my_color)
    enemy_largest = _largest_stack_height(board, my_color.opponent)

    # only activate when the position is already clearly good.
    # in equal or losing positions, keep original deep2 behaviour.
    if before.own_tokens < before.enemy_tokens + 3:
        return actions

    if before.largest < enemy_largest:
        return actions

    scored = [
        (
            _conversion_action_score(board, action, my_color),
            index,
            action,
        )
        for index, action in enumerate(actions)
    ]

    # keep stable order for actions with almost equal score.
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    return [action for _, _, action in scored]


# score a root action for winning conversion.
# removing enemy material, closing distance, and edge pressure are rewarded.
def _conversion_action_score(
    board: Board,
    action: Action,
    my_color: PlayerColor,
) -> float:
    before = _snapshot(board, my_color)
    before_distance = _conversion_min_distance_between_sides(board, my_color)
    before_enemy_edge = _conversion_enemy_edge_distance(board, my_color)

    score = 0.0

    board.apply_action(action)
    try:
        if board.game_over:
            if board.winner_color == my_color:
                return 100000.0
            if board.winner_color == my_color.opponent:
                return -100000.0
            return -500.0

        after = _snapshot(board, my_color)
        after_distance = _conversion_min_distance_between_sides(board, my_color)
        after_enemy_edge = _conversion_enemy_edge_distance(board, my_color)

        enemy_removed = max(0, before.enemy_tokens - after.enemy_tokens)
        own_removed = max(0, before.own_tokens - after.own_tokens)
        net_gain = enemy_removed - own_removed

        # finish the game: removing enemy material is the cleanest conversion.
        score += 90.0 * enemy_removed
        score -= 70.0 * own_removed
        score += 30.0 * net_gain

        # if opponent has very few tokens, make finishing pressure more important
        if after.enemy_tokens <= 3:
            score += 35.0

        if after.enemy_tokens <= 1:
            score += 70.0

        # chase: when ahead, closing distance matters.
        if before_distance is not None and after_distance is not None:
            score += 24.0 * (before_distance - after_distance)

            if after_distance <= 1:
                score += 45.0
            elif after_distance == 2:
                score += 25.0
            elif after_distance >= before_distance and enemy_removed == 0:
                score -= 18.0

        # edge pressure: push enemy toward board edge so it cannot run forever.
        if before_enemy_edge is not None and after_enemy_edge is not None:
            score += 12.0 * (before_enemy_edge - after_enemy_edge)

            if after_enemy_edge == 0:
                score += 25.0
            elif after_enemy_edge == 1:
                score += 10.0

        # pincer pressure: two useful own stacks near enemy is better than one stack chasing
        score += _conversion_pincer_bonus(board, my_color)

        # no-gain cascade is already filtered by root safety.
        # still, when winning, prefer not to split structure unless it has payoff
        if action.__class__.__name__ == "CascadeAction" and enemy_removed <= 0:
            score -= 35.0

        # if this move does nothing material and does not improve pressure,
        # slightly demote it. this targets "I am safe, opponent is also safe" loops.
        if enemy_removed == 0 and before_distance is not None and after_distance is not None:
            if after_distance >= before_distance:
                score -= 12.0

        return score

    finally:
        board.undo_action()


# return the closest distance between our stacks and enemy stacks,,
# smaller distance means stronger chase pressure when we are ahead.
def _conversion_min_distance_between_sides(
    board: Board,
    my_color: PlayerColor,
) -> int | None:
    own_coords = _conversion_stack_coords(board, my_color)
    enemy_coords = _conversion_stack_coords(board, my_color.opponent)

    if not own_coords or not enemy_coords:
        return None

    return min(
        _manhattan(own, enemy)
        for own in own_coords
        for enemy in enemy_coords
    )


# return how close the enemy is to the board edge
# enemies near the edge are easier to trap or push
def _conversion_enemy_edge_distance(
    board: Board,
    my_color: PlayerColor,
) -> int | None:
    enemy_coords = _conversion_stack_coords(board, my_color.opponent)

    if not enemy_coords:
        return None

    return min(_conversion_edge_distance(coord) for coord in enemy_coords)


# reward positions where two own stacks pressure the same enemy.
# this makes it harder for the enemy stack to run away
def _conversion_pincer_bonus(
    board: Board,
    my_color: PlayerColor,
) -> float:
    own_coords = _conversion_stack_coords(board, my_color)
    enemy_coords = _conversion_stack_coords(board, my_color.opponent)

    if len(own_coords) < 2 or not enemy_coords:
        return 0.0

    best_bonus = 0.0

    for enemy in enemy_coords:
        distances = sorted(_manhattan(own, enemy) for own in own_coords)

        if len(distances) < 2:
            continue

        first = distances[0]
        second = distances[1]

        # two stacks both close to enemy means less room to run
        if first <= 2 and second <= 4:
            best_bonus = max(best_bonus, 35.0)
        elif first <= 3 and second <= 5:
            best_bonus = max(best_bonus, 18.0)

    return best_bonus


# collect all stack coordinates for one colour
def _conversion_stack_coords(
    board: Board,
    color: PlayerColor,
) -> list[Coord]:
    coords: list[Coord] = []

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color == color:
                coords.append(coord)

    return coords


# return the closest distance from a coordinate to any board edge
def _conversion_edge_distance(coord: Coord) -> int:
    return min(
        coord.r,
        coord.c,
        BOARD_N - 1 - coord.r,
        BOARD_N - 1 - coord.c,
    )