# ghostcode.idle_worker
from typing import *
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from threading import Thread, Event
from . import types
from .program import Program
from time import time, sleep
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

    _running: bool = False
    _worker_thread: Optional[Thread] = None
    _stop_flag: Event = field(
        init = False
    )
    
    def __post_init__(self) -> None:
        # make sure this is false
        self._stop_flag = Event()
        self._stop_flag.clear()
        # and we explicitly do not start the idle worker atuomatically. user has to call start()
        logger.info(f"Idle worker initialized with a timeout duration of {self.idle_timeout:2f} seconds. Waiting for stat() call.")
        
    def start(self) -> None:
        """Starts the timer and begins working on idle tasks after the timeout is reached."""
        self.update()
        self._stop_flag.clear()
        self._running = True
        logger.info(f"Starting idle worker with an idle timeout of {self.idle_timeout:2f}s and task priorities: {self.task_priorities}.") 
        # fill this in

    def stop(self) -> None:
        """Stop the idle worker.
        This is a blocking method that will wait for ongoing tasks to finish before stoping the worker.
        After calling stop, you must explicitly call start() to enable the countdown and subsequent task working again."""
        self._stop_flag.set()
        self._running = False
        
    def update(self) -> None:
        """Called to signify that user activity is happening. This method will reset the internal idle timeout."""
        self._last_update_t = time()

    def check_idle_timeout(self) -> bool:
        """Checks to see if enough time has passed without calls to update() and the idle worker should begin to work on tasks."""
        return (time() - self._last_update_t) > self.idle_timeout

    def _dispatch(self, idle_task: IdleTask) -> None:
        """Main entry point for the thread worker. Goes through the list of priorities and dispatches a function depending on what task is next in line."""
        # if you like python indentation nesting, you will **love** this!
        # don't loop over nothing
        if self.task_priorities == []:
            # since there is nothing to do we just shut down the worker
            logger.warning(f"Idle worker shutting down because there is nothing to do (empty task priority list).")
            self.stop()
            return
        
        while not self._stop_flag.is_set():
            for idle_task in self.task_priorities:
                match idle_task:
                    case IdleTask.summarize_interactions:
                        if self._summarize_interactions() == "done":
                            continue
                        else:
                            # not done
                            break
                    case IdleTask.update_directory_file:
                        if self._update_directory_file() == "done":
                            continue
                        else:
                            break
                    case _ as unreachable:
                        assert_never(unreachable)

    def _summarize_interactions(self) -> Literal["done", "not_done"]:
        """Goes through the past interactions and tries to use the worker LLM to fill the 'summarize' and 'title' attribute if they are none.
        This is done one interaction history at a time. If there appear to be no more unsummarized or untitled interaction histories, the function returns "done", and "not_done" otherwise."""
        # fill this in
        return "done"

    def _update_directory_file(self) -> Literal["done", "not_done"]:
        """Tries to see if the directory file needs updating, and does so if necessary."""
        # this isn't implemented yet
        return "done"
        pass
