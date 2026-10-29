"""Versioned one-way state protocol for a live Python-authoritative race.

The browser receives presentation state only. It never sends actions, reset,
pause, scoring, or track-selection commands back to the simulation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import threading
import time
from typing import Any


LIVE_SCHEMA_VERSION = 1
LIVE_MAX_MESSAGE_BYTES = 256 * 1024
LIVE_PATH = "/live/v1"


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not (value == value and abs(value) != float("inf")):
        raise ValueError(f"{name} must be finite")
    return value


def _nonempty_string(value: Any, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= limit:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value


def encode_live_message(message: dict[str, Any]) -> str:
    """Validate the public envelope and serialize without NaN/Infinity."""

    if not isinstance(message, dict):
        raise ValueError("live message must be an object")
    kind = _nonempty_string(message.get("kind"), "kind")
    if message.get("schema_version") != LIVE_SCHEMA_VERSION:
        raise ValueError("unsupported live schema version")
    _nonempty_string(message.get("run_id"), "run_id", 128)
    if kind in {"synapse.live.state", "synapse.live.end"}:
        if type(message.get("seq")) is not int or message["seq"] < 0:
            raise ValueError("live sequence must be a non-negative integer")
    if kind == "synapse.live.state":
        if type(message.get("tick")) is not int or message["tick"] < 0:
            raise ValueError("live tick must be a non-negative integer")
        _finite_number(message.get("sim_time_s"), "sim_time_s")
        cars = message.get("cars")
        if not isinstance(cars, list) or not 1 <= len(cars) <= 4:
            raise ValueError("live state must contain 1 to 4 cars")
    elif kind == "synapse.live.hello":
        if type(message.get("physics_hz")) is not int or message["physics_hz"] <= 0:
            raise ValueError("hello physics_hz must be positive")
        if type(message.get("control_hz")) is not int or message["control_hz"] <= 0:
            raise ValueError("hello control_hz must be positive")
        if not isinstance(message.get("track"), dict) or not isinstance(message.get("entrants"), list):
            raise ValueError("hello must include track and entrants")
    elif kind == "synapse.live.end":
        if type(message.get("tick")) is not int or message["tick"] < 0:
            raise ValueError("end tick must be a non-negative integer")
        if not isinstance(message.get("results"), list):
            raise ValueError("end must include results")
    else:
        raise ValueError("unsupported live message kind")
    try:
        encoded = json.dumps(message, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("live message contains unsupported or non-finite data") from exc
    if len(encoded.encode("utf-8")) > LIVE_MAX_MESSAGE_BYTES:
        raise ValueError("live message exceeds byte limit")
    return encoded


def parse_live_message(raw: str | bytes, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse a message for protocol tests and non-browser clients."""

    if isinstance(raw, bytes):
        if len(raw) > LIVE_MAX_MESSAGE_BYTES:
            raise ValueError("live message exceeds byte limit")
        raw = raw.decode("utf-8")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > LIVE_MAX_MESSAGE_BYTES:
        raise ValueError("live message exceeds byte limit")
    try:
        value = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid live JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("live message must be an object")
    encode_live_message(value)
    if previous and value["kind"] != "synapse.live.hello":
        if previous.get("run_id") and value["run_id"] != previous["run_id"]:
            raise ValueError("live run identity changed")
        if "seq" in previous and type(value.get("seq")) is int and value["seq"] <= previous["seq"]:
            raise ValueError("live sequence is stale")
    return value


@dataclass(eq=False)
class _Client:
    websocket: Any
    controls: deque[str]
    latest_state: str | None
    wake: asyncio.Event


class LiveStatePublisher:
    """Loopback WebSocket publisher with latest-state backpressure handling.

    The simulation thread never awaits a client. Intermediate state messages
    are replaced per client, while hello/end control messages are retained.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("live publisher must bind to loopback")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("live port must be 0..65535")
        self.host = host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._stop_requested = threading.Event()
        self._hello: str | None = None
        self._latest_state: str | None = None
        self._server: Any = None
        self._clients: set[_Client] = set()
        self._stop_event: asyncio.Event | None = None

    @property
    def url(self) -> str:
        if self._thread is None or self.port == 0:
            raise RuntimeError("live publisher has not started")
        return f"ws://{self.host}:{self.port}{LIVE_PATH}"

    @property
    def connected_clients(self) -> int:
        return len(self._clients)

    def start(self, hello: dict[str, Any], *, timeout: float = 5.0) -> str:
        if self._thread is not None:
            raise RuntimeError("live publisher cannot be restarted")
        self._hello = encode_live_message(hello)
        self._thread = threading.Thread(target=self._thread_main, name="synapse-live-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError("live publisher did not start")
        if self._failure is not None:
            failure = self._failure
            self.close()
            raise RuntimeError("live publisher failed to start") from failure
        return self.url

    def publish_state(self, message: dict[str, Any]) -> None:
        encoded = encode_live_message(message)
        if self._loop is None:
            return
        self._latest_state = encoded
        self._loop.call_soon_threadsafe(self._enqueue_state, encoded)

    def publish_end(self, message: dict[str, Any]) -> None:
        encoded = encode_live_message(message)
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._enqueue_control, encoded)

    def close(self, *, grace_seconds: float = 0.15) -> None:
        if self._loop is None or self._thread is None:
            return
        if grace_seconds < 0 or grace_seconds > 5:
            raise ValueError("live publisher grace must be 0..5 seconds")
        self._stop_requested.set()
        if grace_seconds:
            time.sleep(grace_seconds)
        loop = self._loop
        if self._stop_event is not None:
            loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=max(1.0, grace_seconds + 1.0))
        if self._thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=1.0)
        self._thread = None
        self._loop = None

    def _thread_main(self) -> None:
        try:
            from websockets.asyncio.server import serve

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(self._serve(serve))
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
        finally:
            try:
                loop.close()  # type: ignore[name-defined]
            except UnboundLocalError:
                pass

    async def _serve(self, serve: Any) -> None:
        self._stop_event = asyncio.Event()
        self._server = await serve(
            self._handler,
            self.host,
            self.port,
            max_size=LIVE_MAX_MESSAGE_BYTES,
            max_queue=2,
            ping_interval=10,
            ping_timeout=5,
            compression=None,
            server_header="synapse-live",
        )
        socket = self._server.sockets[0]
        self.port = int(socket.getsockname()[1])
        self._ready.set()
        await self._stop_event.wait()
        self._server.close()
        await self._server.wait_closed()
        for client in tuple(self._clients):
            await client.websocket.close(code=1001, reason="live race ended")

    async def _handler(self, websocket: Any) -> None:
        client = _Client(websocket, deque(), None, asyncio.Event())
        self._clients.add(client)
        if self._hello is not None:
            client.controls.append(self._hello)
        if self._latest_state is not None:
            client.latest_state = self._latest_state
        client.wake.set()
        sender = asyncio.create_task(self._sender(client))
        try:
            async for _message in websocket:
                await websocket.close(code=1008, reason="live endpoint is read-only")
                break
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            self._clients.discard(client)

    async def _sender(self, client: _Client) -> None:
        while True:
            await client.wake.wait()
            while client.controls:
                control = client.controls.popleft()
                # A terminal control packet must not overtake the newest
                # authoritative state. Hello is intentionally sent first;
                # end is sent after the latest state slot is drained.
                if '"kind":"synapse.live.end"' in control and client.latest_state is not None:
                    state = client.latest_state
                    client.latest_state = None
                    await client.websocket.send(state)
                await client.websocket.send(control)
            if client.latest_state is not None:
                state = client.latest_state
                client.latest_state = None
                await client.websocket.send(state)
            if not client.controls and client.latest_state is None:
                client.wake.clear()

    def _enqueue_state(self, encoded: str) -> None:
        self._latest_state = encoded
        for client in tuple(self._clients):
            client.latest_state = encoded
            client.wake.set()

    def _enqueue_control(self, encoded: str) -> None:
        for client in tuple(self._clients):
            if len(client.controls) >= 4:
                asyncio.create_task(client.websocket.close(code=1013, reason="live client too slow"))
                continue
            client.controls.append(encoded)
            client.wake.set()


__all__ = [
    "LIVE_MAX_MESSAGE_BYTES",
    "LIVE_PATH",
    "LIVE_SCHEMA_VERSION",
    "LiveStatePublisher",
    "encode_live_message",
    "parse_live_message",
]
