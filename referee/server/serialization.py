# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from dataclasses import asdict
from typing import Literal

from ..game import *
from ..game.board import CellState


def serialize_game_board(board: Board) -> list[list[dict]]:
    """
    Serialize a game board to a 2D array of cell states.
    Each cell is represented as {"color": int, "height": int}
    color: 1 = RED, -1 = BLUE, 0 = empty
    """
    sz_board = [[None for _ in range(BOARD_N)] for _ in range(BOARD_N)]
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            sz_board[r][c] = serialize_game_board_cell(board[Coord(r, c)])

    return sz_board


def serialize_game_board_cell(cell: CellState) -> dict:
    """
    Serialize a game board cell to a dictionary.
    """
    if cell.is_empty:
        return {"color": 0, "height": 0}
    elif cell.color == PlayerColor.RED:
        return {"color": 1, "height": cell.height}
    elif cell.color == PlayerColor.BLUE:
        return {"color": -1, "height": cell.height}
    else:
        raise ValueError(f"Invalid cell state: {cell}")


def serialize_game_player(player: Player | PlayerColor | None) -> int:
    """
    Serialize a game player to a dictionary.
    """
    if isinstance(player, Player):
        player = player.color

    return int(player) if player != None else 0


def serialize_game_action(action: Action) -> dict:
    """
    Serialize a game action to a dictionary.
    """
    match action:
        case PlaceAction(coord):
            return {
                "type": "PlaceAction",
                "coord": [coord.r, coord.c],
            }

        case MoveAction(coord, direction):
            return {
                "type": "MoveAction",
                "coord": [coord.r, coord.c],
                "direction": [direction.r, direction.c],
            }

        case EatAction(coord, direction):
            return {
                "type": "EatAction",
                "coord": [coord.r, coord.c],
                "direction": [direction.r, direction.c],
            }

        case CascadeAction(coord, direction):
            return {
                "type": "CascadeAction",
                "coord": [coord.r, coord.c],
                "direction": [direction.r, direction.c],
            }


def serialize_game_update(update: GameUpdate) -> dict:
    """
    Serialize a game update to a dictionary.
    """
    update_cls_name = update.__class__.__name__
    update_payload = {}

    match update:
        case PlayerInitialising(player):
            update_payload = {
                "player": serialize_game_player(player),
            }

        case GameBegin(board):
            update_payload = {
                "board": serialize_game_board(board),
                "phase": board.phase.name,
            }

        case TurnBegin(turn_id, player):
            update_payload = {
                "turnId": turn_id,
                "player": serialize_game_player(player),
            }

        case TurnEnd(turn_id, player, action):
            update_payload = {
                "turnId": turn_id,
                "player": serialize_game_player(player),
                "action": serialize_game_action(action),
            }

        case BoardUpdate(board):
            update_payload = {
                "board": serialize_game_board(board),
                "phase": board.phase.name,
            }

        case GameEnd(winner):
            update_payload = {
                "winner": serialize_game_player(winner),
            }

    return {
        "type": f"GameUpdate:{update_cls_name}",
        **update_payload,
    }
