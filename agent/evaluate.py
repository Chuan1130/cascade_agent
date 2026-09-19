from dataclasses import asdict, dataclass

from referee.game import BOARD_N, Board, Coord, PlayerColor, CARDINAL_DIRECTIONS


WIN_SCORE = 1_000_000
LOSE_SCORE = -1_000_000


# a class to store a set of weights for board evaluation
# each field controls how important one feature is in the final score
@dataclass(frozen=True)
class EvalWeights:
    token_count: float = 950.0          # weight for token number advantage
    stack_count: float = 110.0          # weight for stack number advantage
    mobility: float = 140.0             # weight for available action options
    height_power: float = 190.0         # weight for useful tower structure
    tower_shape: float = 120.0          # weight for high but not trapped towers
    capture: float = 230.0              # weight for immediate eat chances
    vulnerability: float = 260.0        # weight for risk of being eaten
    edge_risk: float = 115.0            # weight for stacks near board edge
    push_threat: float = 0.0            # weight for enemy cascade path risk
    push_off_risk: float = 0.0          # weight for immediate push-off risk
    center_control: float = 120.0       # weight for controlling centre area
    defense_shape: float = 95.0         # weight for compact defensive shape
    cascade_potential: float = 130.0    # weight for possible cascade value
    cascade_action: float = 150.0       # weight for concrete cascade options
    turn_limit_token: float = 700.0     # extra token weight near turn limit
    repetition: float = 0.0             # penalty for entering avoidable repeats
    endgame_conversion: float = 0.0     # reward for turning material into wins
    overmerge: float = 0.0              # penalty for RED becoming one huge stack too early


# BLUE is the second player, so it needs more defensive stability.
# These weights try to reduce over-eager cascade and help BLUE convert small leads.
BLUE_WEIGHTS = EvalWeights(
    token_count=950.282,
    turn_limit_token=754.623,
    stack_count=86.702,
    mobility=107.949,
    height_power=205.06,
    tower_shape=179.018,
    capture=90.246,

    # slightly stronger safety.
    # BLUE was losing some games by splitting or exposing useful towers.
    vulnerability=119.0,
    edge_risk=123.5,

    # light cascade-path safety for BLUE.
    # keep it much smaller than RED's push_off_risk, so BLUE does not become frozen.
    push_threat=18.0,
    push_off_risk=35.0,

    center_control=107.449,
    defense_shape=20.282,

    # lower BLUE cascade eagerness.
    # BLUE can still cascade, but should not overvalue B3/B6 -> many B1s.
    cascade_potential=22.75,
    cascade_action=97.1,

    repetition=159.127,
    endgame_conversion=293.712,
    overmerge=0.0,
)


# RED is the first player, so it should protect its opening advantage.
# These weights mainly help RED avoid wasting structure through overmerge or bad cascade.
RED_WEIGHTS = EvalWeights(
    token_count=958.355,
    turn_limit_token=704.969,
    stack_count=99.325,
    mobility=118.866,
    height_power=215.616,
    tower_shape=185.733,
    capture=79.328,
    vulnerability=103.978,
    edge_risk=113.407,
    push_threat=26.201,
    push_off_risk=158.7,
    center_control=107.782,
    defense_shape=20.485,
    cascade_potential=26.8,
    cascade_action=104.881,
    repetition=0.0,
    endgame_conversion=0.0,
    overmerge=127.149,
)


DEFAULT_WEIGHTS = BLUE_WEIGHTS


# all feature names used by export_feature_vector() and score_from_features()
# keeping this list fixed makes the weight calculation easier to debug
FEATURE_NAMES = (
    "token_count",
    "turn_limit_token",
    "stack_count",
    "mobility",
    "height_power",
    "tower_shape",
    "capture",
    "vulnerability",
    "edge_risk",
    "push_threat",
    "push_off_risk",
    "center_control",
    "defense_shape",
    "cascade_potential",
    "cascade_action",
    "repetition",
    "endgame_conversion",
    "overmerge",
)


# choose evaluation weights based on player role.
# RED and BLUE use different weights because they have different strategic problems.
def _weights_for(perspective: PlayerColor) -> EvalWeights:
    if perspective == PlayerColor.RED:
        return RED_WEIGHTS

    return BLUE_WEIGHTS


# evaluate the current board from perspective's point of view.
# higher score means the position is better for perspective.
# this function only checks the board state, so it can be reused by
# greedy, negamax, alpha-beta, and other future search methods.
def evaluate_board(
    board: Board,
    perspective: PlayerColor,
    weights: EvalWeights | None = None,
) -> float:
    if weights is None:
        weights = _weights_for(perspective)

    # if the game is already over, return win / lose / draw score directly
    if board.game_over:
        return _terminal_score(board, perspective)

    # only calculate cascade safety features when they are actually used.
    # this keeps evaluation cheaper when those feature weights are disabled.
    use_cascade_safety = weights.push_threat != 0.0 or weights.push_off_risk != 0.0

    # convert the board into feature values first, then apply weights
    features = export_feature_vector(
        board,
        perspective,
        include_cascade_safety=use_cascade_safety,
    )

    score = score_from_features(features, weights)

    # collect full stats again for the small conversion bonus.
    # this bonus is separate from the normal weighted feature score.
    my_stats = _collect_player_stats(
        board,
        perspective,
        use_cascade_safety,
    )
    opp_stats = _collect_player_stats(
        board,
        perspective.opponent,
        use_cascade_safety,
    )

    # add a capped bias for converting clear advantages into wins
    score += _conversion_eval_bonus(
        board,
        perspective,
        my_stats,
        opp_stats,
    )

    return score


# convert the board into a feature vector.
# each feature is kept on a bounded or normalised scale before applying weights.
def export_feature_vector(
    board: Board,
    perspective: PlayerColor,
    include_cascade_safety: bool = True,
) -> dict[str, float]:
    opponent = perspective.opponent
    my_stats = _collect_player_stats(board, perspective, include_cascade_safety)
    opp_stats = _collect_player_stats(board, opponent, include_cascade_safety)

    # token gap is the most important feature because elimination decides the game
    token_gap = _relative_gap(my_stats.tokens, opp_stats.tokens, smooth=2.0)

    # near the turn limit, token count becomes even more important
    play_turns = board.play_phase_turn_count
    turn_pressure = 0.0

    if play_turns > 220:
        turn_pressure = min(1.0, (play_turns - 220) / 80)

    # some features need extra game-specific helper logic
    overmerge_risk = _red_overmerge_risk(board, perspective, my_stats, opp_stats)
    repetition_value = _repetition_feature(board, perspective, my_stats, opp_stats)
    conversion_value = _endgame_conversion_feature(
        board,
        perspective,
        my_stats,
        opp_stats,
    )

    return {
        "token_count": token_gap,
        "turn_limit_token": turn_pressure * token_gap,
        "stack_count": _relative_gap(my_stats.stacks, opp_stats.stacks),
        "mobility": _relative_gap(my_stats.mobility, opp_stats.mobility),
        "height_power": _relative_gap(my_stats.height_power, opp_stats.height_power),
        "tower_shape": _relative_gap(my_stats.tower_shape, opp_stats.tower_shape),
        "capture": _relative_gap(my_stats.capture_value, opp_stats.capture_value),

        # risk features are negative for our risk and positive for opponent risk
        "vulnerability": -_relative_gap(
            my_stats.vulnerability,
            opp_stats.vulnerability,
        ),
        "edge_risk": -_relative_gap(my_stats.edge_risk, opp_stats.edge_risk),
        "push_threat": -_relative_gap(my_stats.push_threat, opp_stats.push_threat),
        "push_off_risk": -_relative_gap(
            my_stats.push_off_risk,
            opp_stats.push_off_risk,
        ),

        "center_control": _relative_gap(
            my_stats.center_control,
            opp_stats.center_control,
        ),
        "defense_shape": _relative_gap(
            my_stats.defense_shape,
            opp_stats.defense_shape,
        ),
        "cascade_potential": _relative_gap(
            my_stats.cascade_potential,
            opp_stats.cascade_potential,
        ),
        "cascade_action": _relative_gap(
            my_stats.cascade_action_value,
            opp_stats.cascade_action_value,
        ),
        "repetition": repetition_value,
        "endgame_conversion": conversion_value,

        # overmerge is only meaningful for RED, so the helper returns 0 otherwise
        "overmerge": -min(1.0, overmerge_risk / 180.0),
    }


# calculate the weighted sum of all evaluation features.
# this allows tuned weights and hand-written weights to use the same feature vector.
def score_from_features(
    features: dict[str, float],
    weights: EvalWeights | dict[str, float],
) -> float:
    weight_map = weights_to_dict(weights)
    return sum(weight_map[name] * features.get(name, 0.0) for name in FEATURE_NAMES)


# convert EvalWeights or a normal dictionary into a weight map.
# missing features default to zero in dictionary mode.
def weights_to_dict(weights: EvalWeights | dict[str, float]) -> dict[str, float]:
    if isinstance(weights, dict):
        return {name: float(weights.get(name, 0.0)) for name in FEATURE_NAMES}

    raw = asdict(weights)
    return {name: float(raw[name]) for name in FEATURE_NAMES}


# a class to store board stats for one player
@dataclass(frozen=True)
class PlayerStats:
    tokens: int                 # total number of tokens owned by this player
    stacks: int                 # total number of stacks owned by this player
    height_power: int           # sum of squared stack heights
    mobility: int               # estimated number of useful play actions
    tower_shape: float          # value for useful, not overgrown towers
    capture_value: int          # value of enemy stacks that can be eaten now
    vulnerability: int          # value of own stacks that can be eaten now
    edge_risk: int              # risk caused by being close to board edge
    push_threat: int            # risk caused by enemy cascade pressure
    push_off_risk: int          # immediate risk of being pushed off board
    center_control: int         # score for controlling safer centre positions
    defense_shape: float        # value for compact supporting formations
    cascade_potential: int      # rough value of possible future cascade attacks
    cascade_action_value: float # concrete value of currently useful cascades


# collect all evaluation stats for one player
def _collect_player_stats(
    board: Board,
    color: PlayerColor,
    use_cascade_safety: bool,
) -> PlayerStats:
    tokens = 0
    stacks = 0
    height_power = 0
    mobility = 0
    tower_shape = 0.0
    capture_value = 0
    vulnerability = 0
    edge_risk = 0
    push_threat = 0
    push_off_risk = 0
    center_control = 0
    owned_coords: list[Coord] = []
    cascade_potential = 0
    cascade_action_value = 0.0

    # scan every cell on the board
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            # only calculate stats for this player's stacks
            if cell.color != color:
                continue

            tokens += cell.height
            stacks += 1
            height_power += cell.height * cell.height
            mobility += _mobility_from_stack(board, coord, color)
            tower_shape += _tower_shape_value(cell.height)
            edge_risk += _edge_risk_of_stack(coord, cell.height)
            center_control += _center_score(coord, cell.height)
            owned_coords.append(coord)

            # cascade safety is relatively expensive, so only calculate it when needed
            if use_cascade_safety:
                push_threat += _enemy_push_threat_of_stack(board, coord, color)
                push_off_risk += _immediate_push_off_risk_of_stack(board, coord, color)

            capture_value += _capture_value_from(board, coord, color)
            vulnerability += _vulnerability_of_stack(board, coord, color)
            cascade_potential += _cascade_potential_from(board, coord, color)
            cascade_action_value += _cascade_action_value_from(board, coord, color)

    return PlayerStats(
        tokens=tokens,
        stacks=stacks,
        height_power=height_power,
        mobility=mobility,
        tower_shape=tower_shape,
        capture_value=capture_value,
        vulnerability=vulnerability,
        edge_risk=edge_risk,
        push_threat=push_threat,
        push_off_risk=push_off_risk,
        center_control=center_control,
        defense_shape=_defense_shape_score(owned_coords),
        cascade_potential=cascade_potential,
        cascade_action_value=cascade_action_value,
    )


# return a very large score for win, very small score for lose.
# for draws, avoid accepting a draw when we are ahead.
def _terminal_score(board: Board, perspective: PlayerColor) -> float:
    winner = board.winner_color

    if winner == perspective:
        return WIN_SCORE

    if winner == perspective.opponent:
        return LOSE_SCORE

    # draw contempt: if we are not behind, avoid settling for a repeated draw.
    # if we are behind, a draw is acceptable.
    margin = _count_tokens(board, perspective) - _count_tokens(
        board,
        perspective.opponent,
    )

    if margin > 0 and perspective == PlayerColor.RED:
        return -300.0 - 30.0 * margin

    if perspective != PlayerColor.BLUE:
        return 0.0

    if margin > 0:
        return -320.0 - 35.0 * margin

    if margin == 0 and _current_repetition_count(board) >= 3:
        return -20.0

    if margin < 0:
        return 120.0 + 20.0 * min(4, -margin)

    return 0.0


# count total tokens for one player.
def _count_tokens(board: Board, color: PlayerColor) -> int:
    total = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color:
                total += cell.height

    return total


# compare two feature values on a bounded scale.
# this keeps large raw values from dominating the score too much.
def _relative_gap(own_value: float, enemy_value: float, smooth: float = 1.0) -> float:
    total = abs(own_value) + abs(enemy_value) + smooth
    if total == 0:
        return 0.0

    return (own_value - enemy_value) / total


# evaluate repetition from BLUE's perspective.
# repetition is bad when BLUE is ahead, but can be useful when BLUE is behind.
def _repetition_feature(
    board: Board,
    perspective: PlayerColor,
    my_stats: "PlayerStats",
    opp_stats: "PlayerStats",
) -> float:
    if perspective != PlayerColor.BLUE:
        return 0.0

    if board.play_phase_turn_count <= 0:
        return 0.0

    repeat_count = _current_repetition_count(board)
    if repeat_count <= 1:
        return 0.0

    margin = my_stats.tokens - opp_stats.tokens
    pressure = min(1.0, 0.45 * (repeat_count - 1))

    if margin > 0:
        # better positions should not drift into repetition.
        return -pressure

    if margin == 0:
        # do not blow up equal early positions just to avoid a draw.
        if board.play_phase_turn_count < 120:
            return 0.0
        return -0.25 * pressure

    # if behind, a draw can be useful, but keep the reward smaller than the
    # penalty above so the agent does not seek passive repetition too early.
    return 0.35 * pressure


# evaluate BLUE's ability to convert a small advantage.
# this becomes more important when RED has few tokens left or the turn limit is close.
def _endgame_conversion_feature(
    board: Board,
    perspective: PlayerColor,
    my_stats: "PlayerStats",
    opp_stats: "PlayerStats",
) -> float:
    if perspective != PlayerColor.BLUE:
        return 0.0

    if board.play_phase_turn_count <= 0:
        return 0.0

    lead = my_stats.tokens - opp_stats.tokens
    opponent_low = max(0.0, (7.0 - opp_stats.tokens) / 7.0)
    late_pressure = min(1.0, max(0.0, (board.play_phase_turn_count - 170) / 110))
    phase_weight = max(opponent_low, late_pressure)

    if phase_weight <= 0.0:
        return 0.0

    token_security = _relative_gap(my_stats.tokens, opp_stats.tokens, smooth=1.0)
    attack_pressure = _relative_gap(
        my_stats.capture_value
        + 0.55 * my_stats.cascade_action_value
        + 0.45 * opp_stats.vulnerability
        + 0.25 * opp_stats.edge_risk,
        opp_stats.capture_value
        + 0.55 * opp_stats.cascade_action_value
        + 0.45 * my_stats.vulnerability
        + 0.25 * my_stats.edge_risk,
    )
    mobility_pressure = _relative_gap(my_stats.mobility, opp_stats.mobility)

    value = (
        0.55 * token_security
        + 0.35 * attack_pressure
        + 0.10 * mobility_pressure
    )

    # being one token behind near the turn limit is urgent.
    # make that explicit so the search does not accept quiet moves that only preserve a loss.
    if board.play_phase_turn_count > 220 and lead < 0:
        value -= min(0.5, 0.12 * -lead)

    return max(-1.0, min(1.0, phase_weight * value))


# count how many times the current board position has appeared.
# this uses the referee board's internal repetition history.
def _current_repetition_count(board: Board) -> int:
    position_history = getattr(board, "_position_history", None)
    board_hash = getattr(board, "_board_hash", None)

    if not position_history or board_hash is None:
        return 0

    return position_history.count(board_hash())


# estimate action flexibility for one stack without depending on board.turn_color.
# this lets evaluation score both players' mobility from a fixed board state.
def _mobility_from_stack(board: Board, coord: Coord, color: PlayerColor) -> int:
    cell = board[coord]
    value = 0

    for direction in CARDINAL_DIRECTIONS:
        target_coord = _step(coord, direction)
        if target_coord is None:
            continue

        target = board[target_coord]

        if target.is_empty or target.color == color:
            value += 1
        elif target.color == color.opponent and cell.height >= target.height:
            value += 2

    # cascade gives a stack extra flexibility
    if cell.height >= 2:
        value += 4

    return value


# value a tower that is high enough to matter, while avoiding runaway praise.
# very large towers are useful, but they can also become trapped.
def _tower_shape_value(height: int) -> float:
    if height <= 0:
        return 0.0

    useful_height = min(height, 7)
    value = float(useful_height) + 0.16 * useful_height * useful_height

    if height > 7:
        value -= 0.9 * (height - 7)

    return max(0.0, value)


# reward groups that can protect each other and form small defensive blocks.
def _defense_shape_score(coords: list[Coord]) -> float:
    if len(coords) <= 1:
        return 0.0

    occupied = {(coord.r, coord.c) for coord in coords}
    score = 0.0

    for coord in coords:
        # direct neighbours give stronger support
        for direction in CARDINAL_DIRECTIONS:
            neighbor = (coord.r + direction.r, coord.c + direction.c)
            if neighbor in occupied:
                score += 1.2

        # diagonal neighbours also help keep a compact formation
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (coord.r + dr, coord.c + dc) in occupied:
                    score += 0.55

    # small 2x2 blocks with several pieces are defensively useful
    for r in range(BOARD_N - 1):
        for c in range(BOARD_N - 1):
            filled = 0
            filled += (r, c) in occupied
            filled += (r + 1, c) in occupied
            filled += (r, c + 1) in occupied
            filled += (r + 1, c + 1) in occupied

            if filled >= 3:
                score += 2.5 * filled

    return score


# estimate the best local cascade from this stack.
# this checks whether cascade keeps our tokens and pushes enemy material well.
def _cascade_action_value_from(
    board: Board,
    coord: Coord,
    color: PlayerColor,
) -> float:
    cell = board[coord]
    if cell.height < 2:
        return 0.0

    best_value = 0.0

    for direction in CARDINAL_DIRECTIONS:
        kept_tokens = 0
        lost_tokens = 0
        pushed_enemy_height = 0
        pushed_enemy_count = 0
        own_path_cost = 0
        future_capture = 0

        for step in range(1, cell.height + 1):
            r = coord.r + direction.r * step
            c = coord.c + direction.c * step

            if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
                lost_tokens += 1
                continue

            kept_tokens += 1
            target_coord = Coord(r, c)
            target = board[target_coord]

            if target.color == color.opponent:
                pushed_enemy_count += 1
                pushed_enemy_height += target.height

                # pushing enemy toward edge is better than pushing it inward
                if _would_push_towards_edge(target_coord, direction):
                    pushed_enemy_height += target.height
            elif target.color == color:
                # cascading through our own stacks can disrupt our structure
                own_path_cost += target.height

            future_capture += _adjacent_small_enemy_count(board, target_coord, color)

        # very tall towers should not be split for free
        tower_loss = max(0, cell.height - 5)
        local_value = (
            1.1 * kept_tokens
            + 2.2 * pushed_enemy_height
            + 1.6 * pushed_enemy_count
            + 0.45 * future_capture
            - 2.0 * lost_tokens
            - 0.65 * own_path_cost
            - 0.45 * tower_loss
        )
        best_value = max(best_value, local_value)

    return best_value


# count small adjacent enemy stacks around one coordinate.
# these may become future eat targets after a cascade.
def _adjacent_small_enemy_count(
    board: Board,
    coord: Coord,
    color: PlayerColor,
) -> int:
    count = 0

    for direction in CARDINAL_DIRECTIONS:
        neighbor = _step(coord, direction)
        if neighbor is None:
            continue

        target = board[neighbor]
        if target.color == color.opponent and target.height <= 2:
            count += 1

    return count


# return one coordinate step in a direction.
# return None if the step leaves the board.
def _step(coord: Coord, direction) -> Coord | None:
    r = coord.r + direction.r
    c = coord.c + direction.c

    if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
        return None

    return Coord(r, c)


# estimate immediate eat chances from one stack
def _capture_value_from(board: Board, coord: Coord, color: PlayerColor) -> int:
    cell = board[coord]
    value = 0

    for direction in CARDINAL_DIRECTIONS:
        try:
            target_coord = coord + direction
        except ValueError:
            continue

        target = board[target_coord]

        # if this stack can eat an adjacent enemy stack, add target height
        if target.color == color.opponent and cell.height >= target.height:
            value += target.height

    return value


# estimate whether this stack can be eaten by the opponent immediately
def _vulnerability_of_stack(board: Board, coord: Coord, color: PlayerColor) -> int:
    cell = board[coord]

    for direction in CARDINAL_DIRECTIONS:
        try:
            enemy_coord = coord + direction
        except ValueError:
            continue

        enemy = board[enemy_coord]

        # if opponent can eat this stack, return its height as risk value
        # losing a tall stack is worse than only losing tokens,
        # because tall stacks also have stronger eat and cascade ability.
        if enemy.color == color.opponent and enemy.height >= cell.height:
            return cell.height * cell.height

    return 0


# penalise RED if it becomes one large stack too early.
# this is RED-only and moderate: high stacks are still useful,
# but a trapped single R12 should not be overvalued.
def _red_overmerge_risk(
    board: Board,
    color: PlayerColor,
    my_stats: "PlayerStats",
    opp_stats: "PlayerStats",
) -> float:
    if color != PlayerColor.RED:
        return 0.0

    # if BLUE is nearly dead, a large RED stack can be useful for finishing.
    if opp_stats.tokens <= 4:
        return 0.0

    largest, coord = _largest_stack_info(board, color)
    if largest < 9 or coord is None:
        return 0.0

    risk = 0.0

    # the main problem is becoming one or two stacks while BLUE still has
    # enough material to kite and counter-attack.
    if my_stats.stacks <= 2 and opp_stats.tokens >= 6:
        risk += 60.0 * (3 - my_stats.stacks)

    # R9/R12 is useful, but after a point more height also means more trapped risk.
    risk += 20.0 * (largest - 8)

    # a huge stack near the edge is much worse, because a later cascade can
    # lose tokens off-board and create weak R1s.
    dist_to_edge = min(
        coord.r,
        coord.c,
        BOARD_N - 1 - coord.r,
        BOARD_N - 1 - coord.c,
    )

    if dist_to_edge == 0:
        risk += 80.0 + 8.0 * (largest - 8)
    elif dist_to_edge == 1:
        risk += 35.0 + 4.0 * (largest - 8)

    # if BLUE still has multiple stacks, a single RED tower is easier to chase.
    if opp_stats.stacks >= 2:
        risk += 25.0

    # if the large stack has an immediate eat, do not punish it too much.
    # this avoids making RED too passive when the merge creates real pressure.
    if my_stats.capture_value > 0:
        risk *= 0.7

    return risk


# return largest stack height and coordinate for one player.
def _largest_stack_info(board: Board, color: PlayerColor) -> tuple[int, Coord | None]:
    largest = 0
    best_coord = None

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]

            if cell.color == color and cell.height > largest:
                largest = cell.height
                best_coord = coord

    return largest, best_coord


# calculate edge risk for one stack
def _edge_risk_of_stack(coord: Coord, height: int) -> int:
    dist_to_edge = min(
        coord.r,
        coord.c,
        BOARD_N - 1 - coord.r,
        BOARD_N - 1 - coord.c,
    )

    # stack on the edge is more dangerous
    if dist_to_edge == 0:
        return 3 * height

    # stack one cell away from edge still has some risk
    if dist_to_edge == 1:
        return height

    return 0


# estimate whether this stack is on an enemy cascade path.
# this is intentionally light-weight, so it only slightly discourages moving
# a valuable stack into an obvious cascade corridor.
def _enemy_push_threat_of_stack(
    board: Board,
    coord: Coord,
    color: PlayerColor,
) -> int:
    cell = board[coord]
    risk = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            enemy_coord = Coord(r, c)
            enemy = board[enemy_coord]

            if enemy.color != color.opponent:
                continue

            # height-1 stack cannot cascade
            if enemy.height < 2:
                continue

            direction = _cascade_direction_to_target(enemy_coord, coord)
            if direction is None:
                continue

            distance = _manhattan(enemy_coord, coord)

            # enemy cascade only matters if it can reach our stack
            if distance > enemy.height:
                continue

            # keep this small: use height, not height^2.
            threat = cell.height

            edge_distance = _distance_to_edge_in_direction(coord, direction)

            # being pushed closer to edge is more dangerous
            if edge_distance == 0:
                threat *= 3
            elif edge_distance == 1:
                threat *= 2

            risk += threat

    return risk


# estimate whether this stack can be immediately pushed off the board
# by an enemy cascade.
def _immediate_push_off_risk_of_stack(
    board: Board,
    coord: Coord,
    color: PlayerColor,
) -> int:
    cell = board[coord]

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            enemy_coord = Coord(r, c)
            enemy = board[enemy_coord]

            if enemy.color != color.opponent:
                continue

            if enemy.height < 2:
                continue

            direction = _cascade_direction_to_target(enemy_coord, coord)
            if direction is None:
                continue

            distance = _manhattan(enemy_coord, coord)

            # enemy cascade only matters if it can reach our stack
            if distance > enemy.height:
                continue

            pushes = enemy.height - distance + 1
            edge_distance = _distance_to_edge_in_direction(coord, direction)

            # if pushes is greater than edge distance, this stack can be
            # pushed directly off the board.
            if pushes > edge_distance:
                return cell.height

    return 0


# return cascade direction if source can cascade directly towards target.
def _cascade_direction_to_target(source: Coord, target: Coord):
    for direction in CARDINAL_DIRECTIONS:
        dr = target.r - source.r
        dc = target.c - source.c

        distance = abs(dr) + abs(dc)

        if distance == 0:
            continue

        if source.r + direction.r * distance == target.r and (
            source.c + direction.c * distance == target.c
        ):
            return direction

    return None


# distance from coord to board edge in this direction.
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


# calculate how good this stack's position is based on centre distance
def _center_score(coord: Coord, height: int) -> int:
    center = (BOARD_N - 1) / 2
    distance = abs(coord.r - center) + abs(coord.c - center)

    # maximum Manhattan distance from centre on 8x8 is 7
    return int((7 - distance) * height)


# cheaply estimate how useful future cascade from this stack may be
def _cascade_potential_from(board: Board, coord: Coord, color: PlayerColor) -> int:
    cell = board[coord]

    # height-1 stack cannot cascade
    if cell.height < 2:
        return 0

    value = 0

    for direction in CARDINAL_DIRECTIONS:
        for step in range(1, cell.height + 1):
            r = coord.r + direction.r * step
            c = coord.c + direction.c * step

            # stop checking this direction when it goes out of board
            if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
                break

            target_coord = Coord(r, c)
            target = board[target_coord]

            if target.color == color.opponent:
                # opponent stack in cascade path is useful
                value += target.height

                # extra value if the opponent stack is closer to the edge
                if _would_push_towards_edge(target_coord, direction):
                    value += target.height

    return value


# return True if pushing from coord in this direction moves it closer to edge
def _would_push_towards_edge(coord: Coord, direction) -> bool:
    if direction.r == -1:
        return coord.r <= 2

    if direction.r == 1:
        return coord.r >= BOARD_N - 3

    if direction.c == -1:
        return coord.c <= 2

    if direction.c == 1:
        return coord.c >= BOARD_N - 3

    return False


# Manhattan distance between two coordinates.
def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)


# small evaluation bias for converting a clear advantage.
# this is intentionally capped and phase-aware.
# it should guide negamax, not override the normal material / safety score.
def _conversion_eval_bonus(
    board: Board,
    perspective: PlayerColor,
    my_stats: "PlayerStats",
    opp_stats: "PlayerStats",
) -> float:
    token_lead = my_stats.tokens - opp_stats.tokens

    # RED keeps the original stricter conversion trigger.
    # BLUE needs an earlier trigger in small endgames:
    # for example, B4 vs R2 is already winning for BLUE,
    # but token_lead is only +2, so the old trigger did not activate.
    if perspective == PlayerColor.BLUE:
        if opp_stats.tokens <= 4:
            if token_lead < 1:
                return 0.0
        elif token_lead < 3:
            return 0.0
    else:
        if token_lead < 3:
            return 0.0

    my_largest = _conversion_largest_height(board, perspective)
    opp_largest = _conversion_largest_height(board, perspective.opponent)

    # if opponent has the stronger main tower, do not switch into chase mode.
    if my_largest < opp_largest:
        return 0.0

    min_distance = _conversion_min_distance_between_sides(board, perspective)
    enemy_edge = _conversion_enemy_edge_distance(board, perspective)
    pincer = _conversion_pincer_bonus(board, perspective)

    bonus = 0.0

    # extra reward for making opponent close to elimination.
    # token count is already in the main eval, so this stays small.
    bonus += min(70.0, max(0, 12 - opp_stats.tokens) * 6.0)

    if opp_stats.tokens <= 3:
        bonus += 35.0

    if opp_stats.tokens <= 1:
        bonus += 55.0

    # when ahead, closer pressure is better than safe but useless distance.
    if min_distance is not None:
        if min_distance <= 1:
            bonus += 45.0
        elif min_distance == 2:
            bonus += 32.0
        elif min_distance == 3:
            bonus += 18.0
        elif min_distance >= 6:
            bonus -= 18.0

    # force opponent toward edge/corner so it has fewer escape routes.
    if enemy_edge is not None:
        if enemy_edge == 0:
            bonus += 32.0
        elif enemy_edge == 1:
            bonus += 20.0
        elif enemy_edge == 2:
            bonus += 8.0

    bonus += pincer

    # BLUE-only small endgame chase pressure.
    # when RED is almost dead, BLUE should not drift into a safe repetition.
    # this targets endings like B4 vs R2 where BLUE is ahead but only by 2 tokens.
    if perspective == PlayerColor.BLUE and opp_stats.tokens <= 4 and token_lead > 0:
        if min_distance is not None:
            # smooth chase gradient: every step closer matters.
            # this makes BLUE prefer approaching RED instead of hovering forever.
            bonus += max(-32.0, min(40.0, 10.0 * (5 - min_distance)))

            if min_distance <= 2:
                bonus += 18.0

        # if the current board has already appeared, do not accept repetition
        # while BLUE is ahead in a small endgame.
        repeat_count = _current_repetition_count(board)
        if repeat_count >= 2:
            bonus -= 45.0
        elif repeat_count == 1 and token_lead >= 2:
            bonus -= 15.0

        # if BLUE's main tower is stronger, actively use it to chase.
        if my_largest >= opp_largest + 1 and min_distance is not None and min_distance <= 3:
            bonus += 16.0

    # avoid over-rewarding scattered weak structures.
    # the safety filter already blocks the worst cases; this is just a soft nudge.
    own_singletons = _conversion_singleton_count(board, perspective)
    if own_singletons >= 4 and token_lead < 5:
        bonus -= 20.0

    # keep the bonus small enough that it cannot dominate material/safety.
    if bonus > 220.0:
        return 220.0

    if bonus < -60.0:
        return -60.0

    return bonus


# return largest stack height for one colour.
def _conversion_largest_height(
    board: Board,
    color: PlayerColor,
) -> int:
    largest = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color:
                largest = max(largest, cell.height)

    return largest


# return closest distance between the two sides.
# closer distance can mean stronger chase pressure when ahead.
def _conversion_min_distance_between_sides(
    board: Board,
    color: PlayerColor,
) -> int | None:
    own_coords = _conversion_stack_coords(board, color)
    enemy_coords = _conversion_stack_coords(board, color.opponent)

    if not own_coords or not enemy_coords:
        return None

    return min(
        _conversion_manhattan(own, enemy)
        for own in own_coords
        for enemy in enemy_coords
    )


# return how close the opponent is to any board edge.
def _conversion_enemy_edge_distance(
    board: Board,
    color: PlayerColor,
) -> int | None:
    enemy_coords = _conversion_stack_coords(board, color.opponent)

    if not enemy_coords:
        return None

    return min(_conversion_edge_distance(coord) for coord in enemy_coords)


# reward two-stack pressure against the same enemy stack.
# this makes it harder for the enemy to run away.
def _conversion_pincer_bonus(
    board: Board,
    color: PlayerColor,
) -> float:
    own_coords = _conversion_stack_coords(board, color)
    enemy_coords = _conversion_stack_coords(board, color.opponent)

    if len(own_coords) < 2 or not enemy_coords:
        return 0.0

    best = 0.0

    for enemy in enemy_coords:
        distances = sorted(
            _conversion_manhattan(own, enemy)
            for own in own_coords
        )

        if len(distances) < 2:
            continue

        first = distances[0]
        second = distances[1]

        # two stacks near the same enemy means it is harder for it to run.
        if first <= 2 and second <= 4:
            best = max(best, 34.0)
        elif first <= 3 and second <= 5:
            best = max(best, 18.0)

    return best


# collect coordinates of all stacks for one colour.
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


# count height-1 stacks for one colour.
# too many singletons can make a structure fragile.
def _conversion_singleton_count(
    board: Board,
    color: PlayerColor,
) -> int:
    count = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color and cell.height == 1:
                count += 1

    return count


# return distance from a coordinate to the nearest board edge.
def _conversion_edge_distance(coord: Coord) -> int:
    return min(
        coord.r,
        coord.c,
        BOARD_N - 1 - coord.r,
        BOARD_N - 1 - coord.c,
    )


# Manhattan distance used by conversion helpers.
def _conversion_manhattan(a: Coord, b: Coord) -> int:
    return abs(a.r - b.r) + abs(a.c - b.c)