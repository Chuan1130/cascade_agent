from dataclasses import dataclass
from math import inf
from time import perf_counter

from referee.game import (
    BOARD_N,
    Action,
    Board,
    CascadeAction,
    Coord,
    EatAction,
    MoveAction,
    PlayerColor,
)

from .evaluate import evaluate_board
from .greedy import choose_greedy_action
from .movegen import legal_actions
from .red_safety import (
    cascade_source_height,
    red_merge_order_penalty,
    red_safe_root_actions,
    red_self_damage_cascade_penalty,
    red_tactical_root_penalty,
    red_cleanup_root_bonus,
    red_dominant_endgame_root_bonus,
)


# default depth for the main negamax search
DEFAULT_MAX_DEPTH = 3

# extra depth used after normal search reaches the horizon.
# only forcing actions are searched here.
QUIET_SEARCH_DEPTH = 1

# limit forcing actions in quiescence search.
# RED and BLUE use different limits because their stable versions behaved differently.
MAX_FORCING_ACTIONS_RED = 4
MAX_FORCING_ACTIONS_BLUE = 6

# maximum number of board states stored in the transposition cache
TRANSPOSITION_LIMIT = 60_000

# keep a small time buffer to avoid referee timeout
SAFETY_MARGIN = 0.05

# move-ordering memory parameters.
# actions that caused cutoffs before are searched earlier later.
CUTTING_ACTION_BONUS = 700.0
HISTORY_SCORE_CAP = 4_000.0
HISTORY_SCORE_WEIGHT = 0.35

# penalties for RED entering avoidable repetition while ahead
ROOT_REPEAT_STALL_PENALTY = 95.0
ROOT_REPEAT_DRAW_PENALTY = 360.0

# cache bound types.
# exact means the score is final for this depth.
# lower / upper are alpha-beta bound results.
CACHE_EXACT = 0
CACHE_LOWER = 1
CACHE_UPPER = 2


# one entry in the transposition cache.
# it stores the searched depth, score, bound type, and best action.
@dataclass(frozen=True)
class SearchCacheEntry:
    depth: int
    score: float
    bound: int
    best_action: Action | None


# memory used by move ordering during one root search.
# it records actions that caused cutoffs and actions with good history scores.
@dataclass
class OrderingMemory:
    cutting_actions: dict[int, list[tuple]]
    history_scores: dict[tuple, float]


# exception used when the search is close to timeout
class SearchTimeout(Exception):
    pass


# choose an action by using iterative deepening negamax with alpha-beta pruning.
# greedy result is kept as a safe fallback if search times out.
def choose_search_action(
    board: Board,
    actions: list[Action],
    my_color: PlayerColor,
    time_remaining: float | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Action:
    # actions should not be empty, otherwise the agent has no valid move
    if not actions:
        raise RuntimeError("No legal actions available")

    # RED should not choose obvious self-damaging root actions if there are
    # safer alternatives.
    root_actions = red_safe_root_actions(board, actions, my_color)

    # BLUE also has a small conservative guard.
    # it only blocks clear self-splitting cascade mistakes.
    if my_color == PlayerColor.BLUE:
        root_actions = _blue_minimal_safe_root_actions(board, root_actions, my_color)

    # BLUE needs another guard for immediate terminal loss.
    # this checks whether RED can win immediately after BLUE's action.
    if my_color == PlayerColor.BLUE:
        root_actions = _blue_avoid_immediate_cascade_loss(
            board,
            root_actions,
            my_color,
        )

    # greedy action is used as a safe fallback.
    # if search timeout happens, we can still return a reasonable legal action.
    greedy_fallback = choose_greedy_action(board, root_actions, my_color)

    # deadline is used to avoid spending too much time on one action.
    deadline = _deadline(time_remaining, board, root_actions)

    # iterative deepening keeps the last fully searched result as a safe answer.
    depth_limit = _depth_limit(max_depth, time_remaining, len(root_actions))
    best_action = greedy_fallback
    completed_depth = 0

    # cache and ordering memory only live for this one action decision.
    # this avoids carrying stale ordering information across unrelated turns.
    search_cache: dict[tuple, SearchCacheEntry] = {}
    ordering_memory = OrderingMemory({}, {})

    try:
        for depth in range(1, depth_limit + 1):
            action, score = _search_root(
                board,
                root_actions,
                depth,
                my_color,
                deadline,
                best_action if completed_depth > 0 else None,
                search_cache,
                ordering_memory,
            )
            best_action = action
            completed_depth = depth

    except SearchTimeout:
        return best_action if completed_depth > 0 else greedy_fallback

    return best_action


# search all root actions at one fixed depth.
# root level also applies colour-specific tactical bonuses or penalties.
def _search_root(
    board: Board,
    actions: list[Action],
    depth: int,
    my_color: PlayerColor,
    deadline: float | None,
    preferred_action: Action | None,
    search_cache: dict[tuple, SearchCacheEntry],
    ordering_memory: OrderingMemory,
) -> tuple[Action, float]:
    _check_timeout(deadline)

    best_action = actions[0]
    best_score = -inf
    alpha = -inf
    beta = inf

    # the previous completed depth's best move is tried first when available.
    # this usually improves alpha-beta pruning.
    for action in _ordered_actions(
        board,
        actions,
        my_color,
        preferred_action,
        use_result_bonus=False,
        ordering_memory=ordering_memory,
        ply=0,
    ):
        _check_timeout(deadline)

        before = _side_snapshot(board, my_color)
        source_height = cascade_source_height(board, action)

        board.apply_action(action)
        try:
            # after applying our root action, opponent is to move.
            # negamax returns value from the next side's perspective, so negate it.
            score = -_negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                my_color,
                deadline,
                search_cache,
                ordering_memory,
                1,
            )

            after = _side_snapshot(board, my_color)

            # RED has extra root corrections to avoid throwing first-player advantage.
            if my_color == PlayerColor.RED:
                score -= red_self_damage_cascade_penalty(
                    board,
                    action,
                    before,
                    after,
                    source_height,
                )
                score -= red_tactical_root_penalty(
                    board,
                    action,
                    before,
                    after,
                    source_height,
                )
                score += red_cleanup_root_bonus(
                    board,
                    action,
                    before,
                    after,
                )
                score += red_dominant_endgame_root_bonus(
                    board,
                    action,
                    before,
                    after,
                )

            # BLUE gets extra help for converting small endgame advantages.
            if my_color == PlayerColor.BLUE:
                score += _endgame_root_bonus(board, action, my_color, before, after)

            # RED should not repeat quiet moves while ahead.
            score -= _root_repeat_stall_penalty(
                board,
                action,
                my_color,
                before,
                after,
            )
        finally:
            board.undo_action()

        if score > best_score:
            best_score = score
            best_action = action

        alpha = max(alpha, best_score)

    return best_action, best_score


# recursive negamax search with alpha-beta pruning
def _negamax(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    root_color: PlayerColor,
    deadline: float | None,
    search_cache: dict[tuple, SearchCacheEntry],
    ordering_memory: OrderingMemory,
    ply: int,
) -> float:
    _check_timeout(deadline)

    # if the game is over, evaluate from root player's perspective
    if board.game_over:
        return _relative_evaluation(board, root_color)

    # when normal search reaches its horizon, keep resolving immediate forcing
    # moves so a capture or token-winning cascade is not evaluated too early.
    if depth <= 0:
        return _quiet_search(
            board,
            alpha,
            beta,
            QUIET_SEARCH_DEPTH,
            root_color,
            deadline,
        )

    alpha_start = alpha
    beta_start = beta
    cache_key = _state_key(board)
    cached = search_cache.get(cache_key)
    cached_action = cached.best_action if cached is not None else None

    # reuse cached result if this board was already searched deeply enough
    if cached is not None and cached.depth >= depth:
        if cached.bound == CACHE_EXACT:
            return cached.score

        if cached.bound == CACHE_LOWER:
            alpha = max(alpha, cached.score)
        elif cached.bound == CACHE_UPPER:
            beta = min(beta, cached.score)

        if alpha >= beta:
            return cached.score

    # generate legal actions for the current player
    actions = legal_actions(board)

    # if no legal actions, evaluate current board directly
    if not actions:
        score = _relative_evaluation(board, root_color)
        _store_search_cache(
            search_cache,
            cache_key,
            depth,
            score,
            CACHE_EXACT,
            None,
        )
        return score

    best_score = -inf
    best_action = actions[0]

    # search promising actions first to make pruning more effective
    for action in _ordered_actions(
        board,
        actions,
        root_color,
        cached_action,
        ordering_memory=ordering_memory,
        ply=ply,
    ):
        board.apply_action(action)  # simulate this action
        try:
            # negamax idea:
            # current player's score = negative of opponent's best score
            score = -_negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                root_color,
                deadline,
                search_cache,
                ordering_memory,
                ply + 1,
            )
        finally:
            # undo action after getting its search score
            board.undo_action()

        if score > best_score:
            best_score = score
            best_action = action

        alpha = max(alpha, score)

        # alpha-beta pruning:
        # if alpha is already no worse than beta, this branch is not needed
        if alpha >= beta:
            _record_cutting_action(ordering_memory, ply, action, depth)
            break

    # store the search result for future transposition reuse
    _store_search_cache(
        search_cache,
        cache_key,
        depth,
        best_score,
        _cache_bound_for(best_score, alpha_start, beta_start),
        best_action,
    )

    return best_score


# continue searching forcing moves after normal depth is reached.
# this reduces horizon effect around immediate eats and useful cascades.
def _quiet_search(
    board: Board,
    alpha: float,
    beta: float,
    remaining_depth: int,
    root_color: PlayerColor,
    deadline: float | None,
) -> float:
    _check_timeout(deadline)

    # stand_pat is the evaluation if we stop searching here
    stand_pat = _relative_evaluation(board, root_color)

    if board.game_over or remaining_depth <= 0:
        return stand_pat

    if stand_pat >= beta:
        return stand_pat

    alpha = max(alpha, stand_pat)

    # only continue with forcing actions, not every quiet move
    forcing_actions = _forcing_actions(board, legal_actions(board), root_color)
    if not forcing_actions:
        return stand_pat

    for action in forcing_actions:
        _check_timeout(deadline)

        board.apply_action(action)
        try:
            score = -_quiet_search(
                board,
                -beta,
                -alpha,
                remaining_depth - 1,
                root_color,
                deadline,
            )
        finally:
            board.undo_action()

        if score >= beta:
            return score

        alpha = max(alpha, score)

    return alpha


# select a small list of forcing actions for quiescence search.
# only useful EAT and positive-gain CASCADE actions are kept.
def _forcing_actions(board: Board, actions: list[Action], root_color: PlayerColor) -> list[Action]:
    scored_actions: list[tuple[float, Action]] = []

    for action in actions:
        score = _forcing_action_score(board, action)
        if score > 0:
            scored_actions.append((score, action))

    scored_actions.sort(key=lambda item: item[0], reverse=True)

    # RED keeps a smaller forcing set because its stable version was more conservative.
    # BLUE gets more forcing actions because it needs help converting endgames.
    if root_color == PlayerColor.BLUE:
        limit = MAX_FORCING_ACTIONS_BLUE
    else:
        limit = MAX_FORCING_ACTIONS_RED

    return [
        action
        for _, action in scored_actions[:limit]
    ]


# score whether an action is forcing enough for quiet search.
# actions with no material pressure are ignored.
def _forcing_action_score(board: Board, action: Action) -> float:
    mover = board.turn_color

    # EAT actions are forcing because they immediately remove material
    if isinstance(action, EatAction):
        target = _eat_target(board, action)
        if target is None:
            return 0.0

        return (100.0 + 20.0 * target.height) * _endgame_tactical_multiplier(
            board,
            mover,
        )

    # CASCADE is only forcing if it improves token swing
    if isinstance(action, CascadeAction):
        before = _side_snapshot(board, mover)
        board.apply_action(action)
        try:
            after = _side_snapshot(board, mover)
        finally:
            board.undo_action()

        if _token_swing(before, after) <= 0:
            return 0.0

        return _cascade_result_score(before, after) * _endgame_tactical_multiplier(
            board,
            mover,
        )

    return 0.0


# return the target cell of an eat action.
# return None if the target coordinate is invalid.
def _eat_target(board: Board, action: EatAction):
    try:
        return board[action.coord + action.direction]
    except ValueError:
        return None


# collect a compact material snapshot for one side.
# used by ordering, tactical penalties, and endgame bonuses.
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


# calculate material swing from one side's snapshot.
# positive value means the action improved the material balance.
def _token_swing(
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> int:
    before_balance = before[0] - before[1]
    after_balance = after[0] - after[1]
    return after_balance - before_balance


# score the result of a cascade action.
# good cascades remove enemy material without creating too many weak pieces.
def _cascade_result_score(
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    swing = _token_swing(before, after)
    enemy_removed = max(0, before[1] - after[1])
    own_lost = max(0, before[0] - after[0])
    enemy_stack_reduction = max(0, before[3] - after[3])
    own_extra_singletons = max(0, after[4] - before[4])

    return (
        35.0 * swing
        + 25.0 * enemy_removed
        - 35.0 * own_lost
        + 5.0 * enemy_stack_reduction
        - 3.0 * own_extra_singletons
    )


# increase tactical action value for BLUE in late or small endgames.
# this helps BLUE convert instead of drifting into draws.
def _endgame_tactical_multiplier(board: Board, mover: PlayerColor) -> float:
    if mover != PlayerColor.BLUE:
        return 1.0

    snapshot = _side_snapshot(board, mover)
    enemy_tokens = snapshot[1]

    if enemy_tokens <= 4:
        return 1.75

    if enemy_tokens <= 7 or board.play_phase_turn_count >= 180:
        return 1.35

    return 1.0


# BLUE root bonus for endgame conversion.
# it rewards moves that remove material or improve late token margin.
def _endgame_root_bonus(
    board_after_action: Board,
    action: Action,
    mover: PlayerColor,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    enemy_before = before[1]
    enemy_after = after[1]

    # do not activate too early unless the position is already a small endgame
    if enemy_before > 7 and board_after_action.play_phase_turn_count < 180:
        return 0.0

    bonus = _endgame_action_bonus(board_after_action, action, before, after)
    margin_after = after[0] - after[1]

    # near turn limit, preserving a one-token loss is not acceptable.
    if board_after_action.play_phase_turn_count >= 220:
        bonus += 24.0 * margin_after
        if margin_after < 0:
            bonus += 38.0 * margin_after

    # if enemy is close to elimination, each remaining token matters more
    if enemy_after <= 3:
        bonus += 32.0 * (4 - enemy_after)

    return max(-320.0, min(320.0, bonus))


# score one endgame action based on material swing and enemy reduction.
# this is used by BLUE root bonus and BLUE move ordering.
def _endgame_action_bonus(
    board_after_action: Board,
    action: Action,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    enemy_before = before[1]

    # avoid forcing endgame conversion logic in normal middle-game positions
    if enemy_before > 7 and board_after_action.play_phase_turn_count < 180:
        return 0.0

    swing = _token_swing(before, after)
    enemy_removed = max(0, before[1] - after[1])
    own_lost = max(0, before[0] - after[0])
    enemy_stack_reduction = max(0, before[3] - after[3])

    bonus = (
        45.0 * swing
        + 38.0 * enemy_removed
        - 58.0 * own_lost
        + 10.0 * enemy_stack_reduction
    )

    # EAT is a clean conversion action
    if isinstance(action, EatAction):
        bonus += 24.0 + 12.0 * enemy_removed

    # positive-gain cascade can also finish small endgames
    if isinstance(action, CascadeAction) and swing > 0:
        bonus += 24.0

    return bonus


# adjust action ordering based on repetition risk.
# BLUE should avoid repetition when ahead but may accept it when behind.
def _repetition_order_adjustment(board: Board, root_color: PlayerColor) -> float:
    if root_color != PlayerColor.BLUE:
        return 0.0

    repeat_count = _current_repetition_count(board)
    if repeat_count <= 1:
        return 0.0

    margin = _token_margin(board, root_color)
    pressure = repeat_count - 1

    if margin > 0:
        return -140.0 * pressure

    if margin == 0:
        if board.play_phase_turn_count < 120:
            return 0.0
        return -45.0 * pressure

    return 45.0 * pressure


# penalise RED quiet moves that repeat while ahead.
# this tries to avoid turning winning positions into draw loops.
def _root_repeat_stall_penalty(
    board_after_action: Board,
    action: Action,
    root_color: PlayerColor,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    if root_color != PlayerColor.RED:
        return 0.0

    if not isinstance(action, MoveAction):
        return 0.0

    if board_after_action.play_phase_turn_count <= 0:
        return 0.0

    repeat_count = _current_repetition_count(board_after_action)
    if repeat_count <= 1:
        return 0.0

    # only quiet root moves are punished.
    # if the move changes material, merges stacks, or creates immediate tactical
    # progress, let search judge it.
    if before != after:
        return 0.0

    margin = after[0] - after[1]
    if margin <= 0:
        return 0.0

    pressure = repeat_count - 1
    penalty = ROOT_REPEAT_STALL_PENALTY * pressure

    # a third occurrence ends the game immediately, so it needs to be
    # noticeably worse than keeping the game alive when not behind.
    if repeat_count >= 3:
        penalty += ROOT_REPEAT_DRAW_PENALTY

    return penalty


# count token margin for one colour.
def _token_margin(board: Board, color: PlayerColor) -> int:
    own = 0
    enemy = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color == color:
                own += cell.height
            elif cell.color == color.opponent:
                enemy += cell.height

    return own - enemy


# encode the board into a hashable key for the transposition cache.
# include turn, phase, repetition count, and all cell states.
def _state_key(board: Board) -> tuple:
    cells: list[tuple[int, int]] = []

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            color_code = -1 if cell.color is None else cell.color.value
            cells.append((color_code, cell.height))

    return (
        board.turn_color.value,
        board.phase.value,
        board.turn_count,
        _current_repetition_count(board),
        tuple(cells),
    )


# count current position repetitions using board history.
def _current_repetition_count(board: Board) -> int:
    position_history = getattr(board, "_position_history", None)
    board_hash = getattr(board, "_board_hash", None)

    if not position_history or board_hash is None:
        return 0

    return position_history.count(board_hash())


# decide whether a cached score is exact, lower bound, or upper bound.
def _cache_bound_for(score: float, alpha_start: float, beta_start: float) -> int:
    if score <= alpha_start:
        return CACHE_UPPER

    if score >= beta_start:
        return CACHE_LOWER

    return CACHE_EXACT


# store one search result in the transposition cache.
# old deeper entries are kept over shallow entries.
def _store_search_cache(
    search_cache: dict[tuple, SearchCacheEntry],
    key: tuple,
    depth: int,
    score: float,
    bound: int,
    best_action: Action | None,
):
    current = search_cache.get(key)
    if current is not None and current.depth > depth:
        return

    if current is None and len(search_cache) >= TRANSPOSITION_LIMIT:
        return

    search_cache[key] = SearchCacheEntry(
        depth=depth,
        score=score,
        bound=bound,
        best_action=best_action,
    )


# record actions that caused alpha-beta cutoffs.
# these actions are searched earlier later in the same move search.
def _record_cutting_action(
    ordering_memory: OrderingMemory,
    ply: int,
    action: Action,
    depth: int,
):
    key = _action_key(action)
    recent_actions = ordering_memory.cutting_actions.setdefault(ply, [])

    if key in recent_actions:
        recent_actions.remove(key)

    recent_actions.insert(0, key)
    del recent_actions[2:]

    current_score = ordering_memory.history_scores.get(key, 0.0)
    bonus = float(depth * depth)
    ordering_memory.history_scores[key] = min(
        HISTORY_SCORE_CAP,
        current_score + bonus,
    )


# return ordering bonus from history and cutoff memory.
def _ordering_memory_bonus(
    ordering_memory: OrderingMemory | None,
    action: Action,
    ply: int,
) -> float:
    if ordering_memory is None:
        return 0.0

    key = _action_key(action)
    score = ordering_memory.history_scores.get(key, 0.0) * HISTORY_SCORE_WEIGHT

    if key in ordering_memory.cutting_actions.get(ply, []):
        score += CUTTING_ACTION_BONUS

    return score


# convert an action into a simple tuple key.
# this is used by ordering memory dictionaries.
def _action_key(action: Action) -> tuple:
    coord = action.coord
    key = [action.__class__.__name__, coord.r, coord.c]

    direction = getattr(action, "direction", None)
    if direction is not None:
        key.extend([direction.r, direction.c])

    return tuple(key)


# convert evaluate_board(root_color perspective) into current side perspective.
# this is needed because negamax always flips the sign between players.
def _relative_evaluation(board: Board, root_color: PlayerColor) -> float:
    score = evaluate_board(board, root_color)

    # if it is root player's turn, use score directly
    if board.turn_color == root_color:
        return score

    # otherwise it is opponent's turn, so reverse the score for negamax
    return -score


# order actions before search, so promising actions are searched first.
# better ordering makes alpha-beta pruning happen earlier.
def _ordered_actions(
    board: Board,
    actions: list[Action],
    root_color: PlayerColor,
    preferred_action: Action | None = None,
    use_result_bonus: bool = True,
    ordering_memory: OrderingMemory | None = None,
    ply: int = 0,
) -> list[Action]:
    scored: list[tuple[float, Action]] = []

    # mover is the player who is going to apply these actions
    mover = board.turn_color

    for action in actions:
        before = _side_snapshot(board, mover) if use_result_bonus else None

        # simulate action for ordering
        board.apply_action(action)
        try:
            # evaluate the board from root player's perspective
            score = evaluate_board(board, root_color)

            # if the mover is opponent, a high root score is bad for the mover,
            # so reverse the score for ordering
            if mover != root_color:
                score = -score

            # small priority only helps ordering actions with similar scores
            if use_result_bonus and before is not None:
                after = _side_snapshot(board, mover)
                score += _action_order_bonus(board, action, mover, before, after)
            else:
                score += _basic_action_priority(action)

            # adjust ordering if repetition is strategically good or bad
            score += _repetition_order_adjustment(board, root_color)

            # try the best action from the previous completed depth first
            if preferred_action is not None and action == preferred_action:
                score += 10_000.0

            # add bonus from cutoff and history memory
            score += _ordering_memory_bonus(ordering_memory, action, ply)
        finally:
            # undo action after getting its ordering score
            board.undo_action()

        scored.append((score, action))

    # larger score means this action is more promising, so search it earlier
    scored.sort(key=lambda item: item[0], reverse=True)
    return [action for _, action in scored]


# rough action priority used when full result scoring is disabled.
def _basic_action_priority(action: Action) -> float:
    if isinstance(action, EatAction):
        return 3.0

    if isinstance(action, CascadeAction):
        return 1.0

    if isinstance(action, MoveAction):
        return 0.2

    return 0.0


# add action-specific ordering bonus after simulating the move.
# EAT, CASCADE, and MOVE use different local signals.
def _action_order_bonus(
    board_after_action: Board,
    action: Action,
    mover: PlayerColor,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    if isinstance(action, EatAction):
        enemy_removed = max(0, before[1] - after[1])
        bonus = 25.0 + 15.0 * enemy_removed

        # BLUE values endgame conversion more in ordering
        if mover == PlayerColor.BLUE:
            bonus += 0.45 * _endgame_action_bonus(
                board_after_action,
                action,
                before,
                after,
            )
        return bonus

    if isinstance(action, CascadeAction):
        bonus = _cascade_result_score(before, after)

        # BLUE gets extra endgame conversion signal for useful cascades
        if mover == PlayerColor.BLUE:
            bonus += 0.45 * _endgame_action_bonus(
                board_after_action,
                action,
                before,
                after,
            )
        return bonus

    if isinstance(action, MoveAction):
        bonus = _move_result_bonus(board_after_action, mover, before, after)

        # BLUE can use quiet moves to approach and convert small endgames
        if mover == PlayerColor.BLUE:
            bonus += 0.25 * _endgame_action_bonus(
                board_after_action,
                action,
                before,
                after,
            )
        return bonus

    return 0.0


# order merge moves based on stack reduction and largest stack size.
# RED receives an extra penalty for dangerous overmerge.
def _move_result_bonus(
    board_after_action: Board,
    mover: PlayerColor,
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> float:
    own_stack_reduction = max(0, before[2] - after[2])
    if own_stack_reduction <= 0:
        return 0.2

    # merging stacks is often useful, but for RED it can over-encourage
    # becoming one trapped R12 too early.
    score = 4.0 * own_stack_reduction + 0.25 * _largest_stack(
        board_after_action,
        mover,
    )

    if mover == PlayerColor.RED:
        score -= red_merge_order_penalty(
            board_after_action,
            before,
            after,
        )

    return score


# return largest stack height for one colour.
def _largest_stack(board: Board, color: PlayerColor) -> int:
    largest = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]
            if cell.color == color:
                largest = max(largest, cell.height)

    return largest


# calculate a small deadline for this action search.
# if no time limit is provided, search can run without this local deadline.
def _deadline(
    time_remaining: float | None,
    board: Board,
    actions: list[Action],
) -> float | None:
    # if referee does not provide time limit, do not set deadline
    if time_remaining is None:
        return None

    budget = _move_time_budget(time_remaining, board, actions)

    # keep a safety margin to avoid timeout
    return perf_counter() + max(0.01, budget - SAFETY_MARGIN)


# decide how much time to spend on the current move.
# tactical or high-branching positions receive a little more time.
def _move_time_budget(
    time_remaining: float,
    board: Board,
    actions: list[Action],
) -> float:
    # spend a small fraction of remaining time per move.
    # the cap stays low so one complicated turn does not spend everything.
    budget = min(1.0, max(0.08, time_remaining * 0.015))

    if len(actions) >= 24 or _has_eat_action(actions):
        budget *= 1.2

    if board.play_phase_turn_count > 220:
        budget *= 1.1

    if time_remaining < 30:
        budget = min(budget, 0.35)

    if time_remaining < 10:
        budget = min(budget, 0.12)

    return min(1.2, budget)


# choose search depth based on remaining time and number of root actions.
def _depth_limit(
    max_depth: int,
    time_remaining: float | None,
    action_count: int,
) -> int:
    if time_remaining is not None:
        if time_remaining < 5:
            return 1

        if time_remaining < 20:
            return min(2, max_depth)

    if action_count > 32:
        return min(2, max_depth)

    return max(1, max_depth)


# check whether any root action is an EAT action.
def _has_eat_action(actions: list[Action]) -> bool:
    for action in actions:
        if isinstance(action, EatAction):
            return True

    return False


# check whether current search is close to timeout
def _check_timeout(deadline: float | None):
    if deadline is not None and perf_counter() >= deadline:
        raise SearchTimeout


# BLUE-only minimal root guard.
# this is intentionally small and conservative:
# it only blocks clearly bad BLUE cascades that split a useful tower
# without reducing RED material or RED main tower.
def _blue_minimal_safe_root_actions(
    board: Board,
    actions,
    my_color: PlayerColor,
):
    original_actions = list(actions)
    kept_actions = []

    for action in original_actions:
        if not _is_cascade_action(action):
            kept_actions.append(action)
            continue

        if not _blue_bad_self_split_cascade(board, action, my_color):
            kept_actions.append(action)

    # never allow the guard to remove every move.
    # if all actions look bad, trust the original search instead.
    if kept_actions:
        return kept_actions

    return original_actions


# check whether an action is a CASCADE action.
# this uses string form to keep the guard simple and local.
def _is_cascade_action(action) -> bool:
    return str(action).startswith("CASCADE")


# check whether BLUE cascade splits a useful tower without hurting RED.
# this blocks B3/B4/B5/B6 becoming many weak B1s for no gain.
def _blue_bad_self_split_cascade(
    board: Board,
    action,
    my_color: PlayerColor,
) -> bool:
    before_my_tokens = _guard_token_count(board, my_color)
    before_opp_tokens = _guard_token_count(board, my_color.opponent)
    before_my_largest = _guard_largest_stack(board, my_color)
    before_opp_largest = _guard_largest_stack(board, my_color.opponent)
    before_my_singletons = _guard_singleton_count(board, my_color)

    try:
        board.apply_action(action)

        # if the cascade directly wins, never block it.
        if board.game_over and board.winner_color == my_color:
            return False

        after_my_tokens = _guard_token_count(board, my_color)
        after_opp_tokens = _guard_token_count(board, my_color.opponent)
        after_my_largest = _guard_largest_stack(board, my_color)
        after_opp_largest = _guard_largest_stack(board, my_color.opponent)
        after_my_singletons = _guard_singleton_count(board, my_color)

    finally:
        board.undo_action()

    my_token_loss = before_my_tokens - after_my_tokens
    opp_token_loss = before_opp_tokens - after_opp_tokens
    my_largest_drop = before_my_largest - after_my_largest
    opp_largest_drop = before_opp_largest - after_opp_largest
    extra_singletons = after_my_singletons - before_my_singletons

    # good cascade: it removes enemy material.
    if opp_token_loss >= 1:
        return False

    # good cascade: it significantly damages RED's main tower.
    if opp_largest_drop >= 2:
        return False

    # bad BLUE cascade:
    # B3/B4/B5/B6 becomes many B1s, but RED keeps its material and main stack.
    if my_largest_drop >= 2 and extra_singletons >= 2:
        return True

    # also block cases where BLUE loses material and does not hurt RED.
    if my_token_loss >= 1 and opp_token_loss <= 0:
        return True

    return False


# count total tokens for one colour inside BLUE guard logic.
def _guard_token_count(board: Board, color: PlayerColor) -> int:
    total = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color:
                total += cell.height

    return total


# return largest stack height for one colour inside BLUE guard logic.
def _guard_largest_stack(board: Board, color: PlayerColor) -> int:
    largest = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color:
                largest = max(largest, cell.height)

    return largest


# count height-1 stacks for one colour inside BLUE guard logic.
def _guard_singleton_count(board: Board, color: PlayerColor) -> int:
    count = 0

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = board[Coord(r, c)]

            if cell.color == color and cell.height == 1:
                count += 1

    return count


# BLUE-only immediate loss guard.
# it blocks BLUE moves that allow RED to win immediately by EAT or CASCADE.
# the key fix is CASCADE: BLUE used to miss positions where RED can push
# the last BLUE stack off the board.
def _blue_avoid_immediate_cascade_loss(
    board: Board,
    actions,
    my_color: PlayerColor,
):
    original_actions = list(actions)
    safe_actions = []

    for action in original_actions:
        if not _blue_allows_immediate_terminal_loss(board, action, my_color):
            safe_actions.append(action)

    # never remove every legal action.
    # if all actions lose immediately, let negamax choose the least bad one.
    if safe_actions:
        return safe_actions

    return original_actions


# test whether one BLUE action allows RED to win immediately.
# both EAT and CASCADE replies are checked.
def _blue_allows_immediate_terminal_loss(
    board: Board,
    action,
    my_color: PlayerColor,
) -> bool:
    board.apply_action(action)

    try:
        # if BLUE's own action wins, keep it.
        if board.game_over:
            return board.winner_color == my_color.opponent

        # after BLUE moves, check every RED reply.
        # do not only check EAT. CASCADE can also immediately eliminate BLUE.
        for reply in legal_actions(board):
            board.apply_action(reply)

            try:
                if board.game_over and board.winner_color == my_color.opponent:
                    return True
            finally:
                board.undo_action()

        return False

    finally:
        board.undo_action()