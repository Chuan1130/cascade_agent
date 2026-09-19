from referee.game import Action, Board, CascadeAction, EatAction, MoveAction, PlayerColor

from .evaluate import evaluate_board


# choose the action with the best immediate evaluation.
# this is mainly used as a simple baseline and as a safe fallback for search.
def choose_greedy_action(
    board: Board,
    actions: list[Action],
    my_color: PlayerColor,
) -> Action:
    best_action = actions[0]
    best_score = float("-inf")

    # try every legal action once
    for action in actions:
        applied = False
        score = float("-inf")

        try:
            # simulate this action on the local board
            board.apply_action(action)
            applied = True

            # evaluate the board after this action from our perspective
            score = evaluate_board(board, my_color)

            # add a very small tie-break so equal scores are more stable
            score += _small_action_tie_break(action)

        finally:
            # always undo the simulation before trying the next action
            if applied:
                board.undo_action()

        # keep the action with the highest immediate score
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


# small tie-break between actions with very similar evaluation.
# the real decision should still come from evaluate_board().
# this should stay small so it does not distort negamax-style evaluation later.
def _small_action_tie_break(action: Action) -> float:
    # eating is usually more concrete than a quiet move
    if isinstance(action, EatAction):
        return 1.0

    # cascade can be useful, but should not be over-preferred by greedy alone
    if isinstance(action, CascadeAction):
        return 0.3

    # normal move gets only a tiny bonus
    if isinstance(action, MoveAction):
        return 0.1

    return 0.0