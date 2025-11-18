# ghostcode.ipc
# inter process communication layer
from typing import *
from dataclasses import dataclass, field
from . import types
from .ipc_message import IPCMessage, IPCMessageAdapter

class IPCServer:
    """Inter process communication layer.
    Spawns a small web server that can receive HTTP requests with ghostcode messages."""

    host: str = field(default = "localhost")
    port: int = field(default = 0)

    def start(self) -> Tuple[str, int]:
        """Starts the IPC server.
        Returns a pair of < hostname, port >.
        Raises exception if spawning of http server was unsuccesful."""
        pass

    def stop(self) -> None:
        """Halts the server."""
        pass


    
