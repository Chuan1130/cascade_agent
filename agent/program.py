from referee.game import Action, GamePhase, PlayerColor

from .movegen import legal_actions
from .opening import choose_opening_action
from .clean_search import choose_search_action
from .state import GameState



class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._state = GameState(color)

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """
        actions = legal_actions(self._state.board)
        if not actions:
            raise RuntimeError("No legal actions available")

        if self._state.board.phase == GamePhase.PLACEMENT:
            return choose_opening_action(self._state.board, actions, self._color)

        return choose_search_action(
            self._state.board,
            actions,
            self._color,
            referee.get("time_remaining"),
        )

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self._state.apply_action(color, action)
