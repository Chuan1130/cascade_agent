# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Generator

from .constants import BOARD_N


@dataclass(frozen=True, slots=True)
class Vector2:
    """
    A simple 2D vector "helper" class with basic arithmetic operations
    overloaded for convenience.
    """
    r: int
    c: int

    def __lt__(self, other: 'Vector2') -> bool:
        return (self.r, self.c) < (other.r, other.c)
    
    def __hash__(self) -> int:
        return hash((self.r, self.c))
    
    def __str__(self) -> str:
        return f"Vector2({self.r}, {self.c})"

    def __add__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r + other.r, self.c + other.c)

    def __sub__(self, other: 'Vector2|Direction') -> 'Vector2':
        return self.__class__(self.r - other.r, self.c - other.c)

    def __neg__(self) -> 'Vector2':
        return self.__class__(self.r * -1, self.c * -1)

    def __mul__(self, n: int) -> 'Vector2':
        return self.__class__(self.r * n, self.c * n)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.r
        yield self.c


class Direction(Enum):
    """
    An `enum` capturing the eight directions on the square grid-based board.
    """
    Down      = Vector2(1, 0)
    DownLeft  = Vector2(1, -1)
    DownRight = Vector2(1, 1)
    Up        = Vector2(-1, 0)
    UpLeft    = Vector2(-1, -1)
    UpRight   = Vector2(-1, 1)
    Left      = Vector2(0, -1)
    Right     = Vector2(0, 1)

    @classmethod
    def _missing_(cls, value: tuple[int, int]):
        for item in cls:
            if item.value == Vector2(*value):
                return item
        raise ValueError(f"Invalid direction: {value}")

    def __neg__(self) -> 'Direction':
        return Direction(-self.value)

    def __mul__(self, n: int) -> 'Vector2':
        return self.value * n

    def __str__(self) -> str:
        return {
            Direction.Down:      "[↓]",
            Direction.DownLeft:  "[↙]",
            Direction.DownRight: "[↘]",
            Direction.Up:        "[↑]",
            Direction.UpLeft:    "[↖]",
            Direction.UpRight:   "[↗]",
            Direction.Left:      "[←]",
            Direction.Right:     "[→]",
        }[self]
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.value)

    def __getattribute__(self, __name: str) -> int:
        match __name:
            case "r":
                return self.value.r
            case "c":
                return self.value.c
            case _:
                return super().__getattribute__(__name)


CARDINAL_DIRECTIONS = frozenset([
    Direction.Up,
    Direction.Down,
    Direction.Left,
    Direction.Right
])


@dataclass(order=True, frozen=True)
class Coord(Vector2):
    """
    A specialisation of the `Vector2` class, representing a coordinate on the
    game board. This class also enforces that the coordinates are within the
    bounds of the game board, or in the case of addition/subtraction, using
    modulo arithmetic to "wrap" the coordinates at the edges of the board.
    """

    def __post_init__(self):
        if not (0 <= self.r < BOARD_N) or not (0 <= self.c < BOARD_N):
            raise ValueError(f"Out-of-bounds coordinate: {self}")

    def __str__(self):
        return f"{self.r}-{self.c}"

    def __add__(self, other: 'Direction|Vector2') -> 'Coord':
        return self.__class__(
            (self.r + other.r), 
            (self.c + other.c),
        )

    def __sub__(self, other: 'Direction|Vector2') -> 'Coord':
        return self.__class__(
            (self.r - other.r), 
            (self.c - other.c)
        )
