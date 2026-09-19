# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

import asyncio
from dataclasses import asdict
from time import time
from typing import AsyncGenerator

from .server import RemoteServer
from .serialization import serialize_game_update
from ..game import GameUpdate, PlayerColor, GameBegin, GameEnd


class RemoteGame:
    """
    A remote game instance that can sync updates with a client.
    """
    def __init__(
            self, 
            server: RemoteServer,
            player_names: list[str],
            game_log_lines: list[str]
        ):
        self._server = server
        self._player_names = player_names
        self._history = []

    async def event_handler(self) -> AsyncGenerator:
        """
        Process game updates as they occur and forward them to any listeners.
        """
        while True:
            update: GameUpdate | None = yield
            assert update is not None

            match update:
                case GameBegin(board):
                    self._history.clear()
                    self._server._log.debug("syncing game metadata...")
                    await self.sync_game_metadata()

            try:
                serialized_update = serialize_game_update(update)
        
                await self._server.sync(serialized_update, len(self._history))

                self._server._log.debug(f"broadcasted game update: {serialized_update}")
                self._history.append(update)
                
            except Exception as e:
                self._server._log.error(f"error broadcasting game update: {e}")
                raise e
    
    async def sync_game_metadata(self):
        """
        Send game metadata to the client, e.g. player names. 
        """
        message = {
            "type": "GameMetadata",
            "players": self._player_names,
        }
        await self._server.sync(message)
        self._server._log.debug(f"sent game metadata: {message}")
