# progress_printer.py
from dataclasses import dataclass, field
from typing import *
import threading
import time
import logging


@dataclass
class ProgressPrinter:
    """Indicates progress during long running operations by printing to the screen or an arbitrary destination, similar to a spinner in a web page.
    You can provide a custom print function with print_function, but it needs to conform to the python print function interface (support end, flush, etc)."""

    style: Literal["spinner", "dots"] = "spinner"

    # will be displayed in front of the spinning element
    message: str = "" 
    print_function: Callable[[str], None] = field(default=print)

    _print_thread: Optional[threading.Thread] = None
    _is_running: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._is_running.clear()

    def _print_spinner(self) -> None:
        spinner_chars = ['|', '/', '-', '\\']
        i = 0
        while self._is_running.is_set():
            self.print_function(f"\r{self.message}{spinner_chars[i % len(spinner_chars)]} ", end="", flush=True)
            i += 1
            time.sleep(0.1)
        self.print_function("\r", end="", flush=True) # Clear the line

    def _print_dots(self) -> None:
        dot_patterns = ['.  ', '.. ', '...']
        i = 0
        while self._is_running.is_set():
            self.print_function(f"\r{self.message}{dot_patterns[i % len(dot_patterns)]}", end="", flush=True)
            i += 1
            time.sleep(0.5)
        self.print_function("\r", end="", flush=True) # Clear the line

    def start(self) -> None:
        """Beings the progress indicator by calling the provided print function (python's print by default) with a progress indicating string.
        To stop the progress printing, simply call the stop() method.
        """
        if self._is_running.is_set():
            return # Already running

        self._is_running.set()
        style_to_target = {"spinner": self._print_spinner, "dots": self._print_dots}

        target = style_to_target.get(self.style, self._print_dots) # Default to dots if style is unknown
        self._print_thread = threading.Thread(target=target, daemon=True)
        self._print_thread.start()

    def stop(self) -> None:
        if not self._is_running.is_set():
            return # Not running

        self._is_running.clear()
        if self._print_thread:
            self._print_thread.join(timeout=1) # Wait for the thread to finish
            if self._print_thread.is_alive():
                # If thread is still alive, it might be stuck, log a warning
                logging.warning("ProgressPrinter thread did not terminate gracefully.")
        self._print_thread = None
        self.print_function("\r" + " " * 4 + "\r", end="", flush=True) # Clear the line after stopping

    def __enter__(self) -> "ProgressPrinter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


