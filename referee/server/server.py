# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from collections import Counter
from math import inf
import json
import asyncio
import websockets

from ..log import LogStream, NullLogger
from .message import Message


INCOMING_POLL_INTERVAL = 0.1
LISTEN_HOST = "0.0.0.0" # For use within dev container
LISTEN_PORT = 8765


class InvalidAckError(Exception):
    pass


class RemoteServer:
    def __init__(self, 
            host: str=LISTEN_HOST, 
            port: int=LISTEN_PORT,
            log_stream: LogStream | None = None, 
        ):
        self._host = host
        self._port = port
        self._log = log_stream or NullLogger()

        self._server = None
        self._incoming_messages: list[Message] = []
    
    async def send(self, message: dict, id: int | None = None):
        """
        Send a message to the client.
        """
        assert self._server, "Server not running."

        message_str = json.dumps({
            **message,
            "id": id,
        })
        self._log.debug(f"sending message: {message_str}")

        if self._server is None:
            self._log.warning("attempted to send message when server not running.")
            return
        
        for conn in self._server.connections:
            await conn.send(message_str)
            self._log.debug(f"sent message: {message_str}")

    async def sync(self, message: dict, expect_id: str | None = None):
        """
        Send a message to the client and wait for a response.
        """
        await self.send(message, expect_id)
        self._log.debug("waiting for <ack>...")
        response = await self.receive('<ack>')
        if expect_id and response and expect_id != response.get("id"):
            await self.stop()
            raise InvalidAckError(f"expected ack ID {expect_id}, got {response}")
        self._log.debug("received <ack>")

    async def receive(
        self, 
        message_type: str | None = None,
    ) -> dict | None:
        """
        Take the next message from the queue of the specified type and return
        it. Waits until a message of the specified type is available if none.
        """
        while self._server and len(self._server.connections) > 0:
            if message_type is not None:
                for i, message in enumerate(self._incoming_messages):
                    if message.type == message_type:
                        return self._incoming_messages.pop(i).message
            else:
                if self._incoming_messages:
                    return self._incoming_messages.pop(0).message

            await asyncio.sleep(INCOMING_POLL_INTERVAL)

    async def _handler(self, websocket):
        """
        Handle incoming then outgoing messages.
        """
        async for message in websocket:
            self._log.debug(f"received message: {message}")
            
            try:
                message = json.loads(message)
                message_type = message["type"]
            except json.JSONDecodeError as e:
                self._log.error(f"failed to parse message: {e}")
                return
            except KeyError as e:
                self._log.error(f"missing message type: {e}")
                return

            self._incoming_messages.append(Message(message_type, message))

    async def run(self):
        """
        Run the server.
        """
        async with websockets.serve(self._handler, self._host, self._port) as server:
            self._log.info(f"server listening on ws://{self._host}:{self._port}...")
            self._server = server
            self._future = asyncio.Future()
            try:
                await self._future
            except asyncio.CancelledError:
                self._log.debug("cancelled error occurred")
                pass

    async def stop(self):
        """
        Stop the server gracefully.
        """
        if self._future:
            self._log.info("stopping server...")
            self._future.cancel()

    async def wait_for_client(self):
        """
        Wait for a client to connect.
        """
        while self._server is None:
            await asyncio.sleep(INCOMING_POLL_INTERVAL)

        self._log.info("waiting for client to connect...")

        while len(list(self._server.connections)) == 0:
            await asyncio.sleep(INCOMING_POLL_INTERVAL)

        await self.sync({"type": "<ping>"})

        self._log.info("client connected")

    async def sync_match_metadata(
            self, 
            info: str,
            player_names: list[str],
            player_wins: Counter[str | None],
            match_winner: str | None = None,
        ):
        """
        Send match metadata to the client, e.g. player names. 
        """
        message = {
            "type": "MatchMetadata",
            "info": info,
            "winner": match_winner or '',
            "players": player_names,
            "scores": {
                str(player or 'Draw'): wins for player, wins in player_wins.items()
            }
        }
        await self.sync(message)
        self._log.debug(f"sent match metadata: {message}")
