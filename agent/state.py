from referee.game import Action, Board, PlayerColor


class GameState:
    """
    Minimal internal game-state wrapper for stage 1.

    The referee owns the official board. Our agent must keep a local board in
    sync by applying every action received through update().
    """

    def __init__(self, color: PlayerColor):
        self.color = color
        self.board = Board()

    def apply_action(self, color: PlayerColor, action: Action):
        """
        Apply a referee-notified action to the local board.

        Board.apply_action() uses board.turn_color to validate ownership, so a
        mismatch here means our internal state has fallen out of sync.
        """
        if self.board.turn_color != color:
            raise ValueError(
                "Internal board is out of sync: "
                f"expected {self.board.turn_color}, got {color}"
            )

        self.board.apply_action(action)
