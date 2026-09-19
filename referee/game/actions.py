# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from dataclasses import dataclass

from .coord import Coord, Direction


@dataclass(frozen=True, slots=True)
class PlaceAction:
    """
    A dataclass representing a "place action" during the placement phase,
    which places a new stack at the specified coordinate.
    """
    coord: Coord

    def __str__(self) -> str:
        return f"PLACE({self.coord})"


@dataclass(frozen=True, slots=True)
class MoveAction:
    """
    A dataclass representing a "move action", which moves a stack one cell
    in a cardinal direction (up, down, left, or right).
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"MOVE({self.coord}, {self.direction})"


@dataclass(frozen=True, slots=True)
class EatAction:
    """
    A dataclass representing an "eat action", which captures an adjacent
    enemy stack. Requires attacker height >= target height.
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"EAT({self.coord}, {self.direction})"


@dataclass(frozen=True, slots=True)
class CascadeAction:
    """
    A dataclass representing a "cascade action", which spreads a stack's tokens
    in a cardinal direction, potentially pushing other stacks.
    """
    coord: Coord
    direction: Direction

    def __str__(self) -> str:
        return f"CASCADE({self.coord}, {self.direction})"


Action = PlaceAction | MoveAction | EatAction | CascadeAction
