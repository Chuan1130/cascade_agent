from referee.game import (
    BOARD_N,
    Action,
    Board,
    CascadeAction,
    Coord,
    MoveAction,
    PlayerColor,
)


# RED-only root safety parameters.
# these do not make RED more aggressive.
# they only discourage obvious self-damaging cascade choices.
RED_SELF_DAMAGE_CASCADE_PENALTY = 85.0
RED_HIGH_TOWER_CASCADE_PENALTY = 28.0
RED_BAD_SWING_CASCADE_PENALTY = 55.0
RED_EXPOSED_SINGLETON_PENALTY = 35.0
RED_OVERMERGE_ORDER_PENALTY = 1.0


# filter root actions for RED only.
# this removes obvious RED blunders before search and greedy fallback.
def red_safe_root_actions(
    board: Board,
    actions: list[Action],
    my_color: PlayerColor,
) -> list[Action]:
    if my_color != PlayerColor.RED:
        return actions

    if board.turn_color != PlayerColor.RED:
        return actions

    safe_actions: list[Action] = []

    # keep every action unless it is clearly bad at root level
    for action in actions:
        if _is_red_bad_root_action(board, action):
            continue

        safe_actions.append(action)

    # if everything is filtered, keep original legal actions.
    # this avoids accidentally leaving the agent with no move.
    if not safe_actions:
        return actions

    return safe_actions


# check whether a RED root action is clearly bad.
# keep this conservative: only block obvious tactical blunders.
def _is_red_bad_root_action(board: Board, action: Action) -> bool:
    if isinstance(action, CascadeAction):
        return _is_bad_red_edge_cascade(board, action)

    if isinstance(action, MoveAction):
        return _is_bad_red_root_move(board, action)

    return False


# block high RED stack edge/corner cascade when it loses material
# without removing BLUE tokens.
# this directly targets:
# R9/R12 goes to corner -> CASCADE -> loses tokens -> leaves R1s to be eaten.
def _is_bad_red_edge_cascade(board: Board, action: CascadeAction) -> bool:
    source = board[action.coord]

    if source.color != PlayerColor.RED:
        return False

    # only block high-value stacks.
    # small stacks sometimes need to cascade for mobility.
    if source.height < 6:
        return False

    # focus on edge/corner cascade.
    # this avoids over-blocking useful central cascade moves.
    if not _is_edge(action.coord):
        return False

    before = _side_snapshot(board, PlayerColor.RED)

    board.apply_action(action)
    try:
        # if this cascade wins immediately, never block it.
        if board.game_over and board.winner_color == PlayerColor.RED:
            return False

        after = _side_snapshot(board, PlayerColor.RED)
        exposed_singletons = _exposed_red_singletons(board)
    finally:
        board.undo_action()

    swing = _token_swing(before, after)
    own_lost = max(0, before[0] - after[0])
    enemy_removed = max(0, before[1] - after[1])

    # the clearest bad case:
    # RED loses tokens from a high stack, but BLUE loses nothing.
    if own_lost >= 2 and enemy_removed == 0:
        return True

    # also block if the cascade creates several exposed R1s and does not
    # remove BLUE material.
    if enemy_removed == 0 and exposed_singletons >= 2:
        return True

    # if material swing is bad and there is no concrete enemy removal,
    # do not allow this edge cascade.
    if swing < 0 and enemy_removed == 0:
        return True

    return False


# check whether a RED root move clearly walks into an immediate cascade kill.
# this targets cases like:
# R6 moves into R3 -> R9, then adjacent B6 cascades and pushes R9 off board.
def _is_bad_red_root_move(board: Board, action: MoveAction) -> bool:
    source = board[action.coord]

    if source.color != PlayerColor.RED:
        return False

    dest = _move_destination(action)
    if dest is None:
        return False

    target = board[dest]

    # calculate height after the move.
    # MOVE can either move to an empty cell or merge with a friendly stack.
    result_height = source.height
    if target.color == PlayerColor.RED:
        result_height += target.height

    # only block high-value stacks. small stacks may need to take risks.
    if result_height < 6:
        return False

    board.apply_action(action)
    try:
        # if the move itself somehow wins, never block it.
        if board.game_over and board.winner_color == PlayerColor.RED:
            return False

        # after the move, if BLUE can immediately cascade this stack off board,
        # the move is an obvious tactical blunder.
        if _can_enemy_cascade_push_off(
            board,
            dest,
            PlayerColor.RED,
            result_height,
        ):
            return True

    finally:
        board.undo_action()

    return False


# return the destination coordinate of a move action.
# return None if the destination is outside the board.
def _move_destination(action: MoveAction) -> Coord | None:
    r = action.coord.r + action.direction.r
    c = action.coord.c + action.direction.c

    if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
        return None

    return Coord(r, c)


# check whether a stack can be immediately pushed off by enemy cascade.
# this is a local tactical test used only at RED root.
def _can_enemy_cascade_push_off(
    board: Board,
    target_coord: Coord,
    color: PlayerColor,
    target_height: int,
) -> bool:
    target = board[target_coord]

    if target.color != color:
        return False

    if target.height < target_height:
        return False

    # scan all enemy stacks and check whether any one can cascade toward target
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            enemy_coord = Coord(r, c)
            enemy = board[enemy_coord]

            if enemy.color != color.opponent:
                continue

            # height-1 stack cannot cascade
            if enemy.height < 2:
                continue

            direction = _cascade_direction_to_target(enemy_coord, target_coord)
            if direction is None:
                continue

            distance = _manhattan(enemy_coord, target_coord)

            # enemy cascade only matters if it can reach the target
            if distance > enemy.height:
                continue

            pushes = enemy.height - distance + 1
            edge_distance = _distance_to_edge_in_direction(target_coord, direction)

            # need more pushes than the distance to edge to push it off board.
            if pushes > edge_distance:
                return True

    return False


# return cascade direction if source can cascade directly towards target.
# only row-aligned or column-aligned targets can be hit this way.
def _cascade_direction_to_target(source: Coord, target: Coord):
    dr = target.r - source.r
    dc = target.c - source.c

    if dr == 0 and dc == 0:
        return None

    if dr == 0:
        if dc > 0:
            return (0, 1)
        return (0, -1)

    if dc == 0:
        if dr > 0:
            return (1, 0)
        return (-1, 0)

    return None


# distance from coord to board edge in this direction.
# this is used to decide whether a cascade can push a stack off board.
def _distance_to_edge_in_direction(coord: Coord, direction) -> int:
    dr, dc = direction

    if dr == -1:
        return coord.r

    if dr == 1:
        return BOARD_N - 1 - coord.r

    if dc == -1:
        return coord.c

    if dc == 1:
        return BOARD_N - 1 - coord.c

    return BOARD_N


# check whether a coordinate is on board edge.
def _is_edge(coord: Coord) -> bool:
    return (
        coord.r == 0
        or coord.r == BOARD_N - 1
        or coord.c == 0
        or coord.c == BOARD_N - 1
    )


# count RED singletons that can be eaten by adjacent BLUE stacks.
# exposed R1s are dangerous because BLUE can clean them up easily.
def _exposed_red_singletons(board: Board) -> int:
    count = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color != PlayerColor.RED or cell.height != 1:
                continue

            # check the four adjacent cells for a BLUE stack that can eat this R1
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr = coord.r + dr
                nc = coord.c + dc

                if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
                    continue

                enemy = board[Coord(nr, nc)]

                if enemy.color == PlayerColor.BLUE and enemy.height >= cell.height:
                    count += 1
                    break

    return count


# get original cascade source height before applying the action.
def cascade_source_height(board: Board, action: Action) -> int:
    if not isinstance(action, CascadeAction):
        return 0

    return board[action.coord].height


# RED-only penalty for bad cascade.
# this targets cases like a high RED tower breaking into many R1s without gain.
def red_self_damage_cascade_penalty(
    board_after_action: Board,
    action: Action,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
    source_height: int,
) -> float:
    if not isinstance(action, CascadeAction):
        return 0.0

    # if this cascade wins the game, never punish it.
    if board_after_action.game_over and board_after_action.winner_color == PlayerColor.RED:
        return 0.0

    swing = _token_swing(before, after)
    own_lost = max(0, before[0] - after[0])
    enemy_removed = max(0, before[1] - after[1])
    lead_before = before[0] - before[1]
    lead_after = after[0] - after[1]

    if own_lost <= 0:
        return 0.0

    penalty = 0.0

    # losing own tokens without removing enemy tokens is usually a bad cascade.
    if enemy_removed == 0:
        penalty += RED_SELF_DAMAGE_CASCADE_PENALTY * own_lost
    else:
        penalty += 0.45 * RED_SELF_DAMAGE_CASCADE_PENALTY * max(
            0,
            own_lost - enemy_removed,
        )

    # high tower self-break is especially dangerous because it removes
    # RED's eating and cascade threat structure.
    if source_height >= 6 and own_lost >= 2:
        penalty += RED_HIGH_TOWER_CASCADE_PENALTY * (
            source_height + own_lost
        )

    # if the token swing is bad, punish it more.
    if swing < 0:
        penalty += RED_BAD_SWING_CASCADE_PENALTY * abs(swing)

    # if RED had at least equal material and the cascade reduces the lead,
    # discourage it further.
    if lead_before >= 0 and lead_after < lead_before:
        penalty += 25.0 * (lead_before - lead_after)

    # if a high tower cascade creates R1s that BLUE can eat immediately,
    # punish it. this targets R12 -> many R1s -> B3/B9 eats them.
    exposed_singletons = _exposed_red_singletons(board_after_action)
    if source_height >= 6 and exposed_singletons > 0:
        penalty += RED_EXPOSED_SINGLETON_PENALTY * exposed_singletons

    return penalty


# ordering penalty for RED over-merging into one huge stack.
# this only affects move ordering, not the final board evaluation.
def red_merge_order_penalty(
    board_after_action: Board,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    own_stack_reduction = max(0, before[2] - after[2])
    if own_stack_reduction <= 0:
        return 0.0

    largest = _largest_stack(board_after_action, PlayerColor.RED)
    if largest < 9:
        return 0.0

    own_stacks_after = after[2]
    penalty = 0.0

    # do not over-prioritise lines where RED immediately becomes one or two
    # large stacks while BLUE still has enough material.
    if own_stacks_after <= 2 and after[1] >= 6:
        penalty += 10.0 * (3 - own_stacks_after)

    penalty += 3.0 * (largest - 8)

    return penalty


# a local copy of side snapshot used only by RED safety checks.
# this keeps red_safety independent from negamax's internal helper.
def _side_snapshot(board: Board, color: PlayerColor) -> tuple[int, int, int, int, int]:
    own_tokens = 0
    enemy_tokens = 0
    own_stacks = 0
    enemy_stacks = 0
    own_singletons = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color == color:
                own_tokens += cell.height
                own_stacks += 1
                if cell.height == 1:
                    own_singletons += 1
            elif cell.color == color.opponent:
                enemy_tokens += cell.height
                enemy_stacks += 1

    return (
        own_tokens,
        enemy_tokens,
        own_stacks,
        enemy_stacks,
        own_singletons,
    )


# token swing from current player's perspective.
# positive value means material balance improved after the action.
def _token_swing(
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> int:
    before_balance = before[0] - before[1]
    after_balance = after[0] - after[1]
    return after_balance - before_balance


# return largest stack height for one colour.
def _largest_stack(board: Board, color: PlayerColor) -> int:
    largest = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color == color:
                largest = max(largest, cell.height)

    return largest


# Manhattan distance between two coordinates.
def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)


# RED tactical root parameters.
# these are softer than filtering: they do not remove the move.
# they only make root search prefer safer alternatives.
RED_LOW_VALUE_CASCADE_ROOT_PENALTY = 90.0
RED_CASCADE_EXPOSURE_ROOT_PENALTY = 18.0
RED_CLEANUP_FRAGMENT_ROOT_BONUS = 28.0


# extra RED-only root penalty used by negamax root search.
# it targets middle/endgame mistakes that are too specific for normal eval:
# useless RED cascade, and leaving R3/R6/R9 in BLUE's cascade line.
def red_tactical_root_penalty(
    board_after_action: Board,
    action: Action,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
    source_height: int,
) -> float:
    penalty = 0.0

    if isinstance(action, CascadeAction):
        penalty += _red_low_value_cascade_penalty(
            board_after_action,
            before,
            after,
            source_height,
        )

    penalty += _red_enemy_cascade_exposure_penalty(
        board_after_action,
        action,
        before,
        after,
    )

    return penalty


# extra RED-only root bonus for cleaning BLUE fragments.
# if BLUE has many B1/B2 fragments, RED should prefer eating them before
# they merge back into B5/B7/B9.
def red_cleanup_root_bonus(
    board_after_action: Board,
    action: Action,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    # avoid importing EatAction again: compare by class name.
    if action.__class__.__name__ != "EatAction":
        return 0.0

    enemy_removed = max(0, before[1] - after[1])
    if enemy_removed <= 0:
        return 0.0

    small_blue_count = _small_enemy_fragment_count(
        board_after_action,
        PlayerColor.RED,
    )

    bonus = RED_CLEANUP_FRAGMENT_ROOT_BONUS * enemy_removed

    # if BLUE still has several small fragments after the eat,
    # this kind of cleanup is strategically useful.
    if small_blue_count >= 3:
        bonus += 12.0 * small_blue_count

    # in endgame, each removed token matters more.
    if board_after_action.play_phase_turn_count >= 180:
        bonus *= 1.25

    return min(180.0, bonus)


# penalise RED cascade that does not remove BLUE material or improve material.
# this keeps useful cascade possible, but discourages R2/R3/R6 becoming weak R1s.
def _red_low_value_cascade_penalty(
    board_after_action: Board,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
    source_height: int,
) -> float:
    if board_after_action.game_over and board_after_action.winner_color == PlayerColor.RED:
        return 0.0

    swing = _token_swing(before, after)
    enemy_removed = max(0, before[1] - after[1])
    own_lost = max(0, before[0] - after[0])
    extra_singletons = max(0, after[4] - before[4])

    # a cascade that removes enemy material is allowed.
    if enemy_removed > 0:
        return 0.0

    penalty = 0.0

    # small/medium stack cascade is often bad if it only creates R1s.
    if source_height <= 3 and swing <= 0:
        penalty += RED_LOW_VALUE_CASCADE_ROOT_PENALTY
        penalty += 22.0 * extra_singletons

    # R4/R5/R6 cascade without material gain should also be suspicious.
    if source_height >= 4 and swing <= 0:
        penalty += 0.75 * RED_LOW_VALUE_CASCADE_ROOT_PENALTY
        penalty += 16.0 * extra_singletons

    # if RED actually loses tokens, punish more.
    if own_lost > 0:
        penalty += 35.0 * own_lost

    # near turn limit, do not casually split useful material.
    if board_after_action.play_phase_turn_count >= 200 and swing <= 0:
        penalty *= 1.25

    return min(260.0, penalty)


# penalise positions where RED leaves valuable stacks inside BLUE's next
# cascade corridor. this is softer than a hard filter to avoid over-blocking.
def _red_enemy_cascade_exposure_penalty(
    board_after_action: Board,
    action: Action,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    if board_after_action.game_over:
        return 0.0

    danger = _enemy_cascade_danger_score(
        board_after_action,
        PlayerColor.RED,
    )

    if danger <= 0.0:
        return 0.0

    enemy_removed = max(0, before[1] - after[1])
    own_tokens = after[0]
    enemy_tokens = after[1]

    penalty = RED_CASCADE_EXPOSURE_ROOT_PENALTY * danger

    # if the action also removed BLUE material, accept more tactical risk.
    if enemy_removed > 0:
        penalty *= 0.55

    # if RED is already far ahead, still avoid throwing the lead.
    if own_tokens >= enemy_tokens + 3:
        penalty *= 1.15

    # do not make this dominate terminal or clearly winning tactics.
    return min(320.0, penalty)


# score how much RED valuable material is exposed to BLUE next-turn cascade.
def _enemy_cascade_danger_score(board: Board, color: PlayerColor) -> float:
    danger = 0.0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color != color:
                continue

            # R1/R2 are less important here. the key issue is losing R3+,
            # especially R6/R9, to a BLUE long cascade.
            if cell.height < 3:
                continue

            danger += _danger_to_stack_from_enemy_cascade(board, coord, color)

    return danger


# local estimate of whether enemy cascade can hit this stack next turn.
# this is used as a soft danger score, not a hard legality filter.
def _danger_to_stack_from_enemy_cascade(
    board: Board,
    target_coord: Coord,
    color: PlayerColor,
) -> float:
    target = board[target_coord]
    danger = 0.0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            enemy_coord = Coord(r, c)
            enemy = board[enemy_coord]

            if enemy.color != color.opponent:
                continue

            # only big enemy stacks create the scary long-line cascade.
            if enemy.height < 5:
                continue

            direction = _cascade_direction_to_target(enemy_coord, target_coord)
            if direction is None:
                continue

            distance = _manhattan(enemy_coord, target_coord)
            if distance > enemy.height:
                continue

            pushes = enemy.height - distance + 1
            edge_distance = _distance_to_edge_in_direction(target_coord, direction)

            local = float(target.height)

            # more pushes means more chance to destroy structure.
            local += 0.45 * pushes

            # being near the edge makes the same cascade much worse.
            if edge_distance <= pushes:
                local *= 1.9
            elif edge_distance <= pushes + 1:
                local *= 1.45

            # R6/R9 type stacks are strategically important.
            if target.height >= 6:
                local *= 1.35

            danger += local

    return danger


# count small BLUE fragments after RED action.
# small enemy fragments are good cleanup targets.
def _small_enemy_fragment_count(board: Board, color: PlayerColor) -> int:
    count = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color.opponent and cell.height <= 2:
                count += 1

    return count


# RED dominant single-tower endgame parameters.
# this handles positions like R3 vs B2.
# in this kind of endgame RED should hold centre, keep pressure,
# and avoid being chased into a corner.
RED_DOMINANT_CENTER_BONUS = 38.0
RED_DOMINANT_EDGE_PENALTY = 85.0
RED_DOMINANT_CORNER_PENALTY = 145.0
RED_DOMINANT_CHASE_BONUS = 26.0


# RED bonus for simple stronger-tower endgames.
# it rewards centre control and chase pressure.
def red_dominant_endgame_root_bonus(
    board_after_action: Board,
    action: Action,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    # only RED uses this rule.
    # after = (own_tokens, enemy_tokens, own_stacks, enemy_stacks, own_singletons)
    own_tokens = after[0]
    enemy_tokens = after[1]
    own_stacks = after[2]
    enemy_stacks = after[3]

    if board_after_action.game_over:
        return 0.0

    # only handle simple one-stack-vs-one-stack endgame.
    if own_stacks != 1 or enemy_stacks != 1:
        return 0.0

    # RED must have material / height advantage.
    if own_tokens <= enemy_tokens:
        return 0.0

    # avoid affecting early opening/midgame tower fights.
    if own_tokens + enemy_tokens > 7 and board_after_action.play_phase_turn_count < 140:
        return 0.0

    red_info = _red_single_stack_info(board_after_action, PlayerColor.RED)
    blue_info = _red_single_stack_info(board_after_action, PlayerColor.BLUE)

    if red_info is None or blue_info is None:
        return 0.0

    red_coord, red_height = red_info
    blue_coord, blue_height = blue_info

    if red_height <= blue_height:
        return 0.0

    edge_dist = _red_edge_distance(red_coord)
    enemy_edge_dist = _red_edge_distance(blue_coord)
    distance = _red_manhattan_coord(red_coord, blue_coord)

    bonus = 0.0

    # holding the centre matters in single-tower endgame.
    # on 8x8 board, edge_dist is 0,1,2,3.
    bonus += RED_DOMINANT_CENTER_BONUS * edge_dist

    # being on edge/corner with the stronger tower is usually bad,
    # because the weaker tower can kite/cascade-pressure and force splitting.
    if edge_dist == 0:
        bonus -= RED_DOMINANT_EDGE_PENALTY
        if _red_is_corner(red_coord):
            bonus -= RED_DOMINANT_CORNER_PENALTY
    elif edge_dist == 1:
        bonus -= 0.45 * RED_DOMINANT_EDGE_PENALTY

    # keep pressure. R3 should not run away from B2 forever.
    if distance <= 1:
        bonus += 2.2 * RED_DOMINANT_CHASE_BONUS
    elif distance == 2:
        bonus += 1.5 * RED_DOMINANT_CHASE_BONUS
    elif distance == 3:
        bonus += 0.7 * RED_DOMINANT_CHASE_BONUS
    else:
        bonus -= 18.0 * (distance - 3)

    # if BLUE is close to edge, RED should keep pressure instead of retreating.
    if enemy_edge_dist <= 1:
        bonus += 22.0

    # do not over-reward passive repetition near turn limit.
    if board_after_action.play_phase_turn_count >= 220:
        bonus += 10.0 * (own_tokens - enemy_tokens)

    return max(-260.0, min(260.0, bonus))


# return the only stack of one colour if exactly one exists.
# return None if this colour has zero or multiple stacks.
def _red_single_stack_info(board: Board, color: PlayerColor):
    found = None

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color != color:
                continue

            if found is not None:
                return None

            found = (coord, cell.height)

    return found


# return distance from coordinate to the nearest board edge.
def _red_edge_distance(coord: Coord) -> int:
    return min(
        coord.r,
        coord.c,
        BOARD_N - 1 - coord.r,
        BOARD_N - 1 - coord.c,
    )


# check whether a coordinate is one of the four corners.
def _red_is_corner(coord: Coord) -> bool:
    return (
        (coord.r == 0 or coord.r == BOARD_N - 1)
        and (coord.c == 0 or coord.c == BOARD_N - 1)
    )


# Manhattan distance between two coordinates.
def _red_manhattan_coord(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)