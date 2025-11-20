from typing import *
from dataclasses import dataclass, field
from threading import Thread
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.requests import Request
import socket
import logging
import asyncio
import os # Added for os.devnull

from .ipc_message import IPCMessage, IPCMessageAdapter, IPCResponse, IPCResponseAdapter

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.ipc")

type IPCMessageCallback = Callable[[IPCMessage], Optional[IPCResponse]]

@dataclass
class IPCServer:
    """Inter process communication layer.
    Spawns a small web server that can receive HTTP requests with ghostcode messages.
    """

    host: str = field(default="localhost")
    port: int = field(default=0)

    _server: Optional[uvicorn.Server] = field(default=None, init=False)
    _server_thread: Optional[Thread] = field(default=None, init=False)
    _actual_host: str = field(default="", init=False)
    _actual_port: int = field(default=0, init=False)

    async def _message_endpoint(self, request: Request, handle_message: IPCMessageCallback) -> JSONResponse:
        try:
            body = await request.json()
            ipc_message = IPCMessageAdapter.validate_python(body)
            logger.debug(f"Received IPC message: {ipc_message.type}")
            handle_message(ipc_message)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            logger.error(f"Error processing IPC message: {e}", exc_info=True)
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)

    def start(self, handle_message: IPCMessageCallback) -> Tuple[str, int]:
        """Starts the IPC server.
        Returns a pair of < hostname, port >.
        Raises exception if spawning of http server was unsuccesful.
        """
        if self._server_thread and self._server_thread.is_alive():
            logger.warning("IPCServer already running.")
            return self._actual_host, self._actual_port

        async def route_handler(request: Request) -> JSONResponse:
            return await self._message_endpoint(request, handle_message)


        app = Starlette(routes=[
            Route("/message", route_handler, methods=["POST"])
        ])

        # Try to bind to the specified port, with a fallback to port 0
        current_port = self.port
        max_retries = 2 # Initial attempt + one retry with port 0
        retries = 0

        while retries < max_retries:
            try:
                config = uvicorn.Config(app, host=self.host, port=current_port, log_config=None, log_level="warning", loop="asyncio")
                self._server = uvicorn.Server(config)

                # Uvicorn's run() is blocking, so we need to run it in a separate thread
                # We need to explicitly set the event loop for the new thread
                def run_server() -> None:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                    if self._server is not None:
                        self._server.run()
                    else:
                        logger.error(f"Failed to run server: Server is None.")

                self._server_thread = Thread(target=run_server, daemon=True)
                self._server_thread.start()

                # Wait a moment for the server to start and bind its socket
                # This is a heuristic, but necessary to get the actual port if 0 was used
                # A more robust solution would involve Uvicorn's startup hooks if exposed easily
                import time
                time.sleep(0.5)

                # Retrieve the actual bound host and port
                # This assumes Uvicorn has started at least one server instance and socket
                if self._server.servers and self._server.servers[0].sockets:
                    sock = self._server.servers[0].sockets[0]
                    self._actual_host, self._actual_port = sock.getsockname()
                    # If host is '0.0.0.0' (listen on all interfaces), report '127.0.0.1' for local clients
                    if self._actual_host == '0.0.0.0':
                        self._actual_host = '127.0.0.1'
                    logger.info(f"IPCServer started on {self._actual_host}:{self._actual_port}")
                    return self._actual_host, self._actual_port
                else:
                    raise RuntimeError("Uvicorn server did not expose bound socket information.")

            except OSError as e:
                if "Address already in use" in str(e) and self.port != 0 and retries == 0:
                    logger.warning(f"Port {self.port} already in use. Retrying with an OS-assigned port (0).")
                    current_port = 0 # Try with OS-assigned port
                    retries += 1
                else:
                    logger.error(f"Failed to start IPCServer on {self.host}:{current_port}: {e}", exc_info=True)
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during IPCServer startup: {e}", exc_info=True)
                raise

        raise RuntimeError("Failed to start IPCServer after multiple retries.")

    def stop(self) -> None:
        """Halts the server."""
        if self._server:
            logger.info("Shutting down IPCServer.")
            # Uvicorn's shutdown is async, but we're in a sync context
            # We need to run it in the event loop of the server thread
            if self._server_thread and self._server_thread.is_alive():
                # Schedule the shutdown coroutine to be run in the server's event loop
                # This is a bit tricky as we don't have direct access to the loop from here
                # A simpler approach for a daemon thread is to just stop the server
                # and let the thread die with the main process, but a clean shutdown is better.
                # To gracefully shut down a uvicorn server running in a separate thread,
                # we set its internal 'should_exit' flag and then join the thread.
                # This signals the server's event loop (running in _server_thread) to begin shutdown.
                self._server.should_exit = True
            self._server_thread.join(timeout=5) # Wait for the thread to finish gracefully
            if self._server_thread.is_alive():
                logger.warning("IPCServer thread did not terminate gracefully.")
            self._server = None
            self._server_thread = None
            self._actual_host = ""
            self._actual_port = 0
        else:
            logger.debug("IPCServer not running, no need to stop.")
