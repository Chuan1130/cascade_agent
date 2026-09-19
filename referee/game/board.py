# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from dataclasses import dataclass
from enum import Enum

from .coord import Coord, Direction, CARDINAL_DIRECTIONS, Vector2
from .player import PlayerColor
from .actions import Action, PlaceAction, MoveAction, EatAction, CascadeAction
from .exceptions import IllegalActionException
from .constants import *


@dataclass(frozen=True, slots=True)
class CellState:
    """
    A structure representing the state of a cell on the game board. A cell can
    be empty or contain a stack of tokens of a given player colour and height.
    """
    color: PlayerColor | None = None
    height: int = 0

    def __post_init__(self):
        if self.color is None and self.height != 0:
            raise ValueError("Empty cell cannot have non-zero height")
        if self.color is not None and self.height <= 0:
            raise ValueError("Stack must have positive height")

    @property
    def is_empty(self) -> bool:
        return self.color is None

    @property
    def is_stack(self) -> bool:
        return self.color is not None

    def __str__(self):
        if self.is_empty:
            return "."
        color_char = "R" if self.color == PlayerColor.RED else "B"
        return f"{color_char}{self.height}"


class GamePhase(Enum):
    """
    Enum representing the two phases of the game.
    """
    PLACEMENT = 0
    PLAY = 1


@dataclass(frozen=True, slots=True)
class CellMutation:
    """
    A structure representing a change in the state of a single cell on the game
    board after an action has been played.
    """
    cell: Coord
    prev: CellState
    next: CellState

    def __str__(self):
        return f"CellMutation({self.cell}, {self.prev}, {self.next})"


@dataclass(frozen=True, slots=True)
class BoardMutation:
    """
    A structure representing a change in the state of the game board after an
    action has been played. Each mutation consists of a set of cell mutations.
    """
    action: Action
    cell_mutations: set[CellMutation]

    def __str__(self):
        return f"BoardMutation({self.cell_mutations})"


class Board:
    """
    A class representing the game board for internal use in the referee.

    NOTE: Don't assume this class is an "ideal" board representation for your
    own agent; you should think carefully about how to design data structures
    for representing the state of a game with respect to your chosen strategy.
    This class has not been optimised beyond what is necessary for the referee.
    """
    def __init__(
        self,
        initial_state: dict[Coord, CellState] | None = None,
        initial_player: PlayerColor = PlayerColor.RED
    ):
        """
        Create a new board. It is optionally possible to specify an initial
        board state (in practice this is only used for testing).
        """
        self._state: dict[Coord, CellState] = {
            Coord(r, c): CellState()
            for r in range(BOARD_N)
            for c in range(BOARD_N)
        }
        if initial_state:
            self._state.update(initial_state)

        self._turn_color: PlayerColor = initial_player
        self._history: list[BoardMutation] = []
        self._position_history: list[int] = []  # For threefold repetition
        self._placement_count: int = 0

        # Count initial placements if initial_state provided
        if initial_state:
            for cell in initial_state.values():
                if cell.is_stack:
                    self._placement_count += 1
                elif not cell.is_empty:
                    raise ValueError("Invalid initial state: cell must be empty or a stack")

    def __getitem__(self, cell: Coord) -> CellState:
        """
        Return the state of a cell on the board.
        """
        if not self._within_bounds(cell):
            raise IndexError(f"Cell position '{cell}' is invalid.")
        return self._state[cell]

    @property
    def phase(self) -> GamePhase:
        """
        The current game phase (PLACEMENT or PLAY).
        """
        return GamePhase.PLACEMENT if self._placement_count < PLACEMENT_TURNS else GamePhase.PLAY

    def _count_tokens(self, color: PlayerColor) -> int:
        """
        Count the total number of tokens for a player.
        """
        return sum(
            cell.height for cell in self._state.values()
            if cell.color == color
        )

    def _count_stacks(self, color: PlayerColor) -> int:
        """
        Count the number of stacks for a player.
        """
        return sum(
            1 for cell in self._state.values()
            if cell.color == color
        )

    @property
    def red_tokens(self) -> int:
        """Total tokens for RED (first player)."""
        return self._count_tokens(PlayerColor.RED)

    @property
    def blue_tokens(self) -> int:
        """Total tokens for BLUE (second player)."""
        return self._count_tokens(PlayerColor.BLUE)

    def _board_hash(self) -> int:
        """
        Compute a hash of the current board position for threefold repetition.
        """
        cells = tuple(
            (self._state[coord].color, self._state[coord].height)
            for coord in sorted(self._state)
        )
        return hash((cells, self._turn_color))

    def apply_action(self, action: Action) -> BoardMutation:
        """
        Apply an action to a board, mutating the board state. Throws an
        IllegalActionException if the action is invalid.
        """
        match action:
            case PlaceAction(coord):
                mutation = self._resolve_place_action(action)
            case MoveAction(coord, direction):
                mutation = self._resolve_move_action(action)
            case EatAction(coord, direction):
                mutation = self._resolve_eat_action(action)
            case CascadeAction(coord, direction):
                mutation = self._resolve_cascade_action(action)
            case _:
                raise IllegalActionException(
                    f"Unknown action {action}", self._turn_color)

        for cell_mutation in mutation.cell_mutations:
            self._state[cell_mutation.cell] = cell_mutation.next

        self._history.append(mutation)

        # Track placement count
        if isinstance(action, PlaceAction):
            self._placement_count += 1

        self._turn_color = self._turn_color.opponent

        # Track position history for threefold repetition (only in play phase)
        # Must be after turn switch so hash matches what _is_threefold_repetition sees
        if self.phase == GamePhase.PLAY:
            self._position_history.append(self._board_hash())

        return mutation

    def undo_action(self) -> BoardMutation:
        """
        Undo the last action played, mutating the board state. Throws an
        IndexError if no actions have been played.
        """
        if len(self._history) == 0:
            raise IndexError("No actions to undo.")

        mutation: BoardMutation = self._history.pop()

        self._turn_color = self._turn_color.opponent

        # Undo position history BEFORE undoing placement count, so that
        # phase still reflects the state when the hash was recorded.
        # (The 8th placement transitions to PLAY phase and records a hash;
        # if we decrement placement_count first, phase reverts to PLACEMENT
        # and the hash is never popped, leaking into minimax search.)
        if self._position_history and self.phase == GamePhase.PLAY:
            self._position_history.pop()

        # Undo placement count
        if isinstance(mutation.action, PlaceAction):
            self._placement_count -= 1

        for cell_mutation in mutation.cell_mutations:
            self._state[cell_mutation.cell] = cell_mutation.prev

        return mutation

    def render(self, use_color: bool = False, use_unicode: bool = False) -> str:
        """
        Returns a visualisation of the game board as a multiline string, with
        optional ANSI color codes and Unicode characters (if applicable).
        """
        def apply_ansi(text, bold=True, color=None):
            bold_code = "\033[1m" if bold else ""
            color_code = ""
            if color == "BLUE":
                color_code = "\033[34m"  # Blue
            if color == "RED":
                color_code = "\033[31m"  # Red
            return f"{bold_code}{color_code}{text}\033[0m"

        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                cell = self._state[Coord(r, c)]
                if cell.is_stack:
                    # PlayerColor.RED displays as RED (R), PlayerColor.BLUE displays as BLUE (B)
                    color_char = "R" if cell.color == PlayerColor.RED else "B"
                    text = f"{color_char}{cell.height}"
                    if use_color:
                        color_name = "RED" if cell.color == PlayerColor.RED else "BLUE"
                        output += apply_ansi(text, color=color_name, bold=True)
                    else:
                        output += text
                else:
                    output += ". "
                output += " "
            output += "\n"
        return output

    @property
    def turn_count(self) -> int:
        """
        The number of actions that have been played so far.
        """
        return len(self._history)

    @property
    def play_phase_turn_count(self) -> int:
        """
        The number of turns played in the play phase.
        """
        return max(0, self.turn_count - PLACEMENT_TURNS)

    @property
    def turn_limit_reached(self) -> bool:
        """
        True iff the maximum number of turns has been reached in the play phase.
        """
        return self.play_phase_turn_count >= MAX_TURNS

    @property
    def turn_color(self) -> PlayerColor:
        """
        The player whose turn it is (represented as a colour).
        """
        return self._turn_color

    def _is_threefold_repetition(self) -> bool:
        """
        Check if the current position has occurred three times.
        """
        if len(self._position_history) < 3:
            return False
        current_hash = self._board_hash()
        return self._position_history.count(current_hash) >= 3

    def _has_legal_actions(self) -> bool:
        """
        Check whether the current player has at least one legal action.
        Short-circuits as soon as any legal action is found.
        """
        color = self._turn_color
        opponent = color.opponent

        for coord, cell in self._state.items():
            if cell.color != color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                dest_r = coord.r + direction.r
                dest_c = coord.c + direction.c

                if self._is_within_bounds(dest_r, dest_c):
                    dest = self._state[Coord(dest_r, dest_c)]

                    # MOVE: destination is empty or friendly
                    if dest.is_empty or dest.color == color:
                        return True

                    # EAT: destination is enemy with height <= ours
                    if dest.color == opponent and cell.height >= dest.height:
                        return True

            # CASCADE: stack height >= 2 (always valid in some direction)
            if cell.height >= 2:
                return True

        return False

    @property
    def game_over(self) -> bool:
        """
        True iff the game is over.
        """
        # No win conditions during placement phase
        if self.phase == GamePhase.PLACEMENT:
            return False

        # Check elimination (only in play phase)
        if self._count_tokens(PlayerColor.RED) == 0:
            return True
        if self._count_tokens(PlayerColor.BLUE) == 0:
            return True

        # Check turn limit
        if self.turn_limit_reached:
            return True

        # Check threefold repetition
        if self._is_threefold_repetition():
            return True

        # Check stalemate (current player has no legal moves)
        if not self._has_legal_actions():
            return True

        return False

    @property
    def winner_color(self) -> PlayerColor | None:
        """
        The player (color) who won the game, or None if no player has won (draw).
        """
        if not self.game_over:
            return None

        white_tokens = self._count_tokens(PlayerColor.RED)
        black_tokens = self._count_tokens(PlayerColor.BLUE)

        # Elimination
        if white_tokens == 0:
            return PlayerColor.BLUE
        if black_tokens == 0:
            return PlayerColor.RED

        # Threefold repetition is always a draw
        if self._is_threefold_repetition():
            return None

        # Stalemate is always a draw
        if not self._has_legal_actions():
            return None

        # Turn limit - compare token counts
        if white_tokens > black_tokens:
            return PlayerColor.RED
        elif black_tokens > white_tokens:
            return PlayerColor.BLUE

        # Equal tokens = draw
        return None

    def _within_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N

    def _is_within_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N

    def _cell_empty(self, coord: Coord) -> bool:
        return self._state[coord].is_empty

    def _assert_coord_valid(self, coord: Coord):
        if type(coord) != Coord or not self._within_bounds(coord):
            raise IllegalActionException(
                f"'{coord}' is not a valid coordinate.", self._turn_color)

    def _assert_coord_occupied_by(self, coord: Coord, color: PlayerColor):
        cell = self._state[coord]
        if cell.is_empty or cell.color != color:
            raise IllegalActionException(
                f"Coord {coord} is not occupied by player {color}.",
                    self._turn_color)

    def _assert_coord_empty(self, coord: Coord):
        if not self._cell_empty(coord):
            raise IllegalActionException(
                f"Coord {coord} is already occupied.", self._turn_color)

    def _assert_direction_cardinal(self, direction: Direction):
        if direction not in CARDINAL_DIRECTIONS:
            raise IllegalActionException(
                f"Direction {direction} is not cardinal (must be up/down/left/right).",
                self._turn_color)

    def _assert_has_attr(self, action: Action, attr: str):
        if not hasattr(action, attr):
            raise IllegalActionException(
                f"Action '{action}' is missing '{attr}' attribute.",
                    self._turn_color)

    def _assert_phase(self, expected_phase: GamePhase):
        if self.phase != expected_phase:
            raise IllegalActionException(
                f"Action not allowed in {self.phase.name} phase.",
                self._turn_color)

    def _is_adjacent_to_opponent(self, coord: Coord) -> bool:
        """Check if coord is adjacent to any opponent stack."""
        for direction in CARDINAL_DIRECTIONS:
            try:
                neighbor = coord + direction
            except ValueError:
                continue  # Off board
            if self._state[neighbor].color == self._turn_color.opponent:
                return True
        return False

    # ========================
    # PLACE Action Resolution
    # ========================
    def _resolve_place_action(self, action: PlaceAction) -> BoardMutation:
        """Resolve a PLACE action during placement phase."""
        self._assert_phase(GamePhase.PLACEMENT)
        self._assert_coord_valid(action.coord)
        self._assert_coord_empty(action.coord)

        # After first placement, cannot place adjacent to opponent
        if self._placement_count > 0 and self._is_adjacent_to_opponent(action.coord):
            raise IllegalActionException(
                f"Cannot place adjacent to opponent's stack after first turn",
                self._turn_color)

        cell_mutations = {
            CellMutation(
                action.coord,
                self._state[action.coord],
                CellState(self._turn_color, INITIAL_STACK_HEIGHT)
            )
        }

        return BoardMutation(action, cell_mutations)

    # ========================
    # MOVE Action Resolution
    # ========================
    def _resolve_move_action(self, action: MoveAction) -> BoardMutation:
        """Resolve a MOVE action: relocate or merge."""
        self._assert_phase(GamePhase.PLAY)
        self._assert_coord_valid(action.coord)
        self._assert_coord_occupied_by(action.coord, self._turn_color)
        self._assert_direction_cardinal(action.direction)

        src = action.coord
        try:
            dest = src + action.direction
        except ValueError:
            raise IllegalActionException(
                f"Move from {src} in direction {action.direction} goes out of bounds.",
                self._turn_color)

        self._assert_coord_valid(dest)

        src_cell = self._state[src]
        dest_cell = self._state[dest]

        cell_mutations = set()

        if dest_cell.is_empty:
            # Relocate: move stack to empty cell
            cell_mutations.add(CellMutation(src, src_cell, CellState()))
            cell_mutations.add(CellMutation(dest, dest_cell, src_cell))
        elif dest_cell.color == self._turn_color:
            # Merge: add heights
            new_height = src_cell.height + dest_cell.height
            cell_mutations.add(CellMutation(src, src_cell, CellState()))
            cell_mutations.add(CellMutation(
                dest, dest_cell, CellState(self._turn_color, new_height)))
        else:
            # Enemy stack - illegal
            raise IllegalActionException(
                f"Cannot MOVE onto enemy stack at {dest}. Use EAT instead.",
                self._turn_color)

        return BoardMutation(action, cell_mutations)

    # ========================
    # EAT Action Resolution
    # ========================
    def _resolve_eat_action(self, action: EatAction) -> BoardMutation:
        """Resolve an EAT action: capture adjacent enemy."""
        self._assert_phase(GamePhase.PLAY)
        self._assert_coord_valid(action.coord)
        self._assert_coord_occupied_by(action.coord, self._turn_color)
        self._assert_direction_cardinal(action.direction)

        src = action.coord
        try:
            dest = src + action.direction
        except ValueError:
            raise IllegalActionException(
                f"Eat from {src} in direction {action.direction} goes out of bounds.",
                self._turn_color)

        self._assert_coord_valid(dest)

        src_cell = self._state[src]
        dest_cell = self._state[dest]

        if dest_cell.is_empty:
            raise IllegalActionException(
                f"Cannot EAT empty cell at {dest}.",
                self._turn_color)

        if dest_cell.color == self._turn_color:
            raise IllegalActionException(
                f"Cannot EAT own stack at {dest}.",
                self._turn_color)

        # Enemy stack - check height requirement
        if src_cell.height < dest_cell.height:
            raise IllegalActionException(
                f"Cannot EAT: attacker height ({src_cell.height}) < target height ({dest_cell.height}).",
                self._turn_color)

        # Capture: attacker moves to target cell, target removed
        cell_mutations = {
            CellMutation(src, src_cell, CellState()),
            CellMutation(dest, dest_cell, src_cell)
        }

        return BoardMutation(action, cell_mutations)

    # ========================
    # CASCADE Action Resolution
    # ========================
    def _resolve_cascade_action(self, action: CascadeAction) -> BoardMutation:
        """
        Resolve a CASCADE action: spread tokens in a direction, pushing stacks.

        Algorithm:
        1. Remove original stack
        2. For each token i (1 to height), place at origin + direction * i
        3. If stack in the way, push recursively
        4. Stacks pushed off board are eliminated
        5. Tokens that would land off board are discarded
        """
        self._assert_phase(GamePhase.PLAY)
        self._assert_coord_valid(action.coord)
        self._assert_coord_occupied_by(action.coord, self._turn_color)
        self._assert_direction_cardinal(action.direction)

        src = action.coord
        src_cell = self._state[src]

        if src_cell.height < 2:
            raise IllegalActionException(
                f"Cannot CASCADE stack with height < 2 (height is {src_cell.height}).",
                self._turn_color)

        height = src_cell.height
        direction = action.direction

        # Build working state for simulation
        working_state = dict(self._state)

        # Remove original stack
        working_state[src] = CellState()

        # Place tokens and handle pushes
        for i in range(1, height + 1):
            target_r = src.r + direction.r * i
            target_c = src.c + direction.c * i

            # Check if target is in bounds
            if not self._is_within_bounds(target_r, target_c):
                # Token is discarded (falls off board)
                continue

            target_coord = Coord(target_r, target_c)
            target_cell = working_state[target_coord]

            if target_cell.is_empty:
                # Place token
                working_state[target_coord] = CellState(self._turn_color, 1)
            else:
                # Stack in the way - push it first, then place token
                working_state = self._push_stack(
                    working_state, target_coord, direction)
                # Now place the token
                working_state[target_coord] = CellState(self._turn_color, 1)

        # Build cell mutations
        cell_mutations = set()
        for coord in self._state:
            if self._state[coord] != working_state[coord]:
                cell_mutations.add(CellMutation(
                    coord, self._state[coord], working_state[coord]))

        return BoardMutation(action, cell_mutations)

    def _push_stack(
        self,
        working_state: dict[Coord, CellState],
        coord: Coord,
        direction: Direction
    ) -> dict[Coord, CellState]:
        """
        Push a stack at coord in the given direction. If the destination
        has a stack, recursively push it first. Stacks pushed off board
        are eliminated.
        """
        cell = working_state[coord]
        if cell.is_empty:
            return working_state

        dest_r = coord.r + direction.r
        dest_c = coord.c + direction.c

        if not self._is_within_bounds(dest_r, dest_c):
            # Pushed off board - eliminated
            working_state[coord] = CellState()
            return working_state

        dest_coord = Coord(dest_r, dest_c)
        dest_cell = working_state[dest_coord]

        if dest_cell.is_stack:
            # Recursively push the stack at destination
            working_state = self._push_stack(working_state, dest_coord, direction)

        # Move current stack to destination
        working_state[dest_coord] = cell
        working_state[coord] = CellState()

        return working_state

    def set_cell_state(self, cell: Coord, state: CellState):
        self._state[cell] = state

    def set_turn_color(self, color: PlayerColor):
        self._turn_color = color

    def set_placement_count(self, count: int):
        self._placement_count = count
