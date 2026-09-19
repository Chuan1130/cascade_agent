# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """
    Represents a message sent to or received from a connected client.
    """
    type: str
    message: dict
