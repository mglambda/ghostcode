# ghostcode.idle_worker
from typing import *
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from threading import Thread, Event
from . import types
from .program import Program
from time import time
import logging
from enum import StrEnum

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.idle_worker")

IdleTask = StrEnum(
    "IdleTask",
    "summarize_interactions update_directory_file"
)

@dataclass
class IdleWorker:
    # Program backpointer, this is necessary so that the idle worker can affect state
    prog_bp: Program

    # the worker will go through these in order froml eft to right, moving on to a different type of task if the first is finished
    # tasks that are not in the priority list are essentially disabled
    task_priorities: List[IdleTask] = field(
        default_factory = lambda: list(IdleTask)
    )
    
    # duration of time in seconds after which we start to work on tasks. This may be user configurable.
    idle_timeout: float = 30.0

    # time that we got the last update, signifying that activity is happening and we should not work on idle tasks
    _last_update_t: float = field(
        default_factory = time
    )

    _worker_thread: Optional[Thread] = None
    _stop_flag: Event = field(
        init = False
    )
    
    def __post_init__(self) -> None:
        # make sure this is false
        self._stop_flag.clear()
        # and we explicitly do not start the idle worker atuomatically. user has to call start()
        
    def start(self) -> None:
        """Starts the timer and begins working on idle tasks after the timeout is reached."""
        self.update()
        self._stop_flag.clear()
        # fill this in

    def stop(self) -> None:
        """Stop the idle worker.
        This is a blocking method that will wait for ongoing tasks to finish before stoping the worker.
        After calling stop, you must explicitly call start() to enable the countdown and subsequent task working again."""
        self._stop_flag.set()
        
    def update(self) -> None:
        """Called to signify that user activity is happening. This method will reset the internal idle timeout."""
        self._last_update_t = time()

    def check_idle_timeout(self) -> bool:
        """Checks to see if enough time has passed without calls to update() and the idle worker should begin to work on tasks."""
        return (time() - self._last_update_t) > self.idle_timeout
