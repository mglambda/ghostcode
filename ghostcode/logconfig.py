# ghostcode.logconfig
from typing import *
import os
import sys
import traceback
import logging
from queue import Queue, Empty



class ExceptionListHandler(logging.Handler):
    """Stores exceptions in a global list."""

    def __init__(self, level=logging.NOTSET):  # type: ignore
        super().__init__(level)
        self.exceptions: Queue[str] = Queue()

    def emit(self, record):  # type: ignore
        if record.exc_info:
            # record.exc_info is (type, value, tb)
            formatted_exc = "".join(traceback.format_exception(*record.exc_info))
            self.exceptions.put(formatted_exc)
        # You could also store the raw record if needed
        # self.exceptions.put(record)

    def try_get_last_traceback(self) -> Optional[str]:
        try:
            return self.exceptions.get_nowait()
        except Empty:
            return None


# Global exception handler. kWould be nicer to store in Program type but unfortunately we might handle exceptions ebfore program is fully constructed.
EXCEPTION_HANDLER = ExceptionListHandler()  # type: ignore


def _configure_logging(
    log_mode: str,
    project_root: Optional[str],
    is_init: bool = False,
    secondary_log_filepath: Optional[str] = None,
    is_debug: bool = False,
) -> None:
    """
    Configures the root logger based on the specified mode and project root.
    Primary logging is always directed to .ghostcode/log.txt if a project root exists and is writable.
    If primary file logging fails or no project root is found, stderr becomes the primary target.
    A secondary logging target can be configured via `log_mode` and `secondary_log_filepath`.
    """
    # Clear existing handlers to prevent duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # global exception tracking
    global EXCEPTION_HANDLER
    logging.root.addHandler(EXCEPTION_HANDLER)

    # Custom filter to suppress exc_info for other handlers
    class SuppressExcInfoFilter(logging.Filter):
        def filter(self, record):  # type: ignore
            # This filter is applied to handlers *after* EXCEPTION_HANDLER has processed it.
            # So, EXCEPTION_HANDLER gets the original record with exc_info.
            # For subsequent handlers, we clear exc_info so they don't print it.
            record.exc_info = None
            return True

    suppress_exc_info_filter = SuppressExcInfoFilter()

    # define our own timing log level
    TIMING_LEVEL_NUM = 25
    logging.addLevelName(TIMING_LEVEL_NUM, "TIMING")

    # Add a convenience method to the Logger class
    def timing_log(self, message, *args, **kwargs):  # type: ignore
        if self.isEnabledFor(TIMING_LEVEL_NUM):
            self._log(TIMING_LEVEL_NUM, message, args, **kwargs)

    logging.Logger.timing = timing_log  # type: ignore

    # Set the root logger level
    if is_debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # --- Primary Logging: Always to .ghostcode/log.txt if project_root exists, else stderr ---
    primary_file_handler_added = False
    if project_root:
        ghostcode_dir = os.path.join(project_root, ".ghostcode")
        primary_log_filepath = os.path.join(ghostcode_dir, "log.txt")
        try:
            os.makedirs(os.path.dirname(primary_log_filepath), exist_ok=True)
            file_handler = logging.FileHandler(primary_log_filepath)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(suppress_exc_info_filter)
            logging.root.addHandler(file_handler)
            primary_file_handler_added = True
        except Exception as e:
            print(
                f"WARNING: Failed to set up primary file logging to {primary_log_filepath}: {e}. Primary logs will go to stderr.",
                file=sys.stderr,
            )
            stderr_error_handler = logging.StreamHandler(sys.stderr)
            stderr_error_handler.setFormatter(formatter)
            stderr_error_handler.setLevel(
                logging.ERROR
            )  # Only show errors/critical from this handler
            stderr_error_handler.addFilter(suppress_exc_info_filter)
            logging.root.addHandler(stderr_error_handler)
            logging.root.error(
                f"Failed to set up primary file logging: {e}", exc_info=True
            )

    # If primary file logging was not possible, ensure a default stderr handler is present.
    if not primary_file_handler_added:
        # Only add if no stderr handler is already present (e.g., from the error fallback above)
        if not any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
            for h in logging.root.handlers
        ):
            default_stderr_handler = logging.StreamHandler(sys.stderr)
            default_stderr_handler.setFormatter(formatter)
            default_stderr_handler.addFilter(suppress_exc_info_filter)
            logging.root.addHandler(default_stderr_handler)

    # --- Secondary Logging: Based on --logging parameter ---
    if log_mode == "stderr":
        # Add a secondary stderr handler. Avoid duplicates if stderr is already a primary/default.
        if not any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
            for h in logging.root.handlers
        ):
            secondary_stderr_handler = logging.StreamHandler(sys.stderr)
            secondary_stderr_handler.setFormatter(formatter)
            secondary_stderr_handler.addFilter(suppress_exc_info_filter)
            logging.root.addHandler(secondary_stderr_handler)
    elif log_mode == "file":
        if secondary_log_filepath:
            try:
                os.makedirs(os.path.dirname(secondary_log_filepath), exist_ok=True)
                secondary_file_handler = logging.FileHandler(secondary_log_filepath)
                secondary_file_handler.setFormatter(formatter)
                secondary_file_handler.addFilter(suppress_exc_info_filter)
                logging.root.addHandler(secondary_file_handler)
            except Exception as e:
                print(
                    f"WARNING: Failed to set up secondary file logging to {secondary_log_filepath}: {e}. No secondary file logging will occur.",
                    file=sys.stderr,
                )
                logging.root.error(
                    f"Failed to set up secondary file logging: {e}", exc_info=True
                )
        else:
            print(
                "WARNING: '--logging file' was specified, but no --secondary-log-filepath was provided. No secondary file logging will occur.",
                file=sys.stderr,
            )
    elif log_mode == "off":
        pass


    
