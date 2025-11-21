# ghostcode.idle_worker
from typing import *
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from threading import Thread, Event
import hashlib
from . import types
from .program import Program
from . import worker
from . import prompts
from .utility import timestamp_now_iso8601
from time import time, sleep
import logging
from enum import StrEnum

# --- Logging Setup ---
logger = logging.getLogger("ghostcode.idle_worker")

IdleTask = StrEnum(
    "IdleTask",
    "summarize_interactions summarize_context_files update_directory_file"
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

    # problematic interactions are added in here so the worker doesn't loop forever on them but skips 
    _skip_interaction_ids: Set[str] = field(
        default_factory = set
    )

    skip_context_file_filepaths: Set[str] = Field(
        default_factory = set,
        description = "Problematic context files are added here so we don't get stuck on summarizing them."
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
        self._worker_thread = Thread(target=self._run, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the idle worker.
        This is a blocking method that will wait for ongoing tasks to finish before stoping the worker.
        After calling stop, you must explicitly call start() to enable the countdown and subsequent task working again."""
        self._stop_flag.set()
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5) # Wait for the thread to finish gracefully
            if self._worker_thread.is_alive():
                logger.warning("IdleWorker thread did not terminate gracefully.")
        
    def update(self) -> None:
        """Called to signify that user activity is happening. This method will reset the internal idle timeout."""
        self._last_update_t = time()
        self._stop_flag.set()

    def wait(self) -> None:
        """Updates and blocks until the current task (if any) is done."""
        self.update()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5) # Wait for the thread to finish gracefully
            if self._worker_thread.is_alive():
                logger.warning("IdleWorker thread did not terminate gracefully.")

        # time may have passed so why not
        self.update()

    def check_idle_timeout(self) -> bool:
        """Checks to see if enough time has passed without calls to update() and the idle worker should begin to work on tasks."""
        return (time() - self._last_update_t) > self.idle_timeout

    def _dispatch(self) -> None:
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
                    case IdleTask.summarize_context_files:
                        if self._summarize_context_files() == "done":
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

    def _run(self) -> None:
        logger.debug(f"Idle worker thread started. Checking for idle timeout every 1 second.")
        while self._running and not self._stop_flag.is_set():
            sleep(1) # Check every second
            if self.check_idle_timeout():
                logger.debug(f"Idle timeout reached ({self.idle_timeout:.2f}s). Dispatching idle tasks.")
                self._dispatch()
        logger.debug(f"Idle worker thread stopped.")

    def _summarize_interactions(self) -> Literal["done", "not_done"]:
        if self.prog_bp.project is None:
            logger.error(f"Null project in idle worker summary task.")
            self.stop()
            return "done"
        
        # Retrieve the ID of the currently active interaction, if any, to skip it.
        current_interaction_id = self.prog_bp.lock_read()
        
        # Iterate through interactions to find those needing a title or summary.
        # We iterate in reverse to prioritize more recent interactions, which are often more relevant.
        for interaction in reversed(self.prog_bp.project.interactions):
            # Skip the currently active interaction to prevent conflicts.
            if current_interaction_id and interaction.unique_id == current_interaction_id:
                logger.debug(f"Skipping current interaction {interaction.unique_id} for summarization.")
                continue

            # skip problematic interactions
            if interaction.unique_id in self._skip_interaction_ids:
                continue
            
            made_changes = False

            # 1. Generate title if missing
            if not interaction.title.strip():
                logger.debug(f"Generating title for interaction {interaction.unique_id} (Tag: {interaction.tag}).")
                new_title = worker.worker_generate_title(self.prog_bp, interaction)
                if new_title:
                    interaction.title = new_title
                    made_changes = True
                    logger.debug(f"Generated title for {interaction.unique_id}: '{new_title}'.")
                else:
                    logger.warning(f"Failed to generate title for interaction {interaction.unique_id}. Skipping next time (maybe try setting worker to a more powerful LLM).")
                    self._skip_interaction_ids.add(interaction.unique_id)

            # 2. Generate summary if missing
            if not interaction.summary.strip():
                logger.debug(f"Generating summary for interaction {interaction.unique_id} (Tag: {interaction.tag}).")
                try:
                    self.prog_bp.worker_box.clear_history()
                    summary_prompt = prompts.make_prompt_interaction_summary(interaction)
                    new_summary = self.prog_bp.worker_box.text(summary_prompt)
                    if new_summary:
                        interaction.summary = new_summary
                        made_changes = True
                        logger.debug(f"Generated summary for {interaction.unique_id}: '{new_summary[:100]}...'.")
                    else:
                        logger.warning(f"Worker returned empty summary for interaction {interaction.unique_id}. Skipping next time (maybe try with a more powerful LLM for the worker).")
                        self._skip_interaction_ids.add(interaction.unique_id)
                except Exception as e:
                    logger.debug(f"Error generating summary for interaction {interaction.unique_id} (skipping next time): {e}")
                    self._skip_interaction_ids.add(interaction.unique_id)                    

            # If any changes were made, save the project and return 'not_done' to allow other tasks a turn.
            if made_changes:
                logger.debug(f"Saving project after updating interaction {interaction.unique_id}.")
                # FIXME: make this a proper Program.save_interaction_history call
                self.prog_bp.project.save_to_root(self.prog_bp.project_root) # type: ignore
                return "not_done"

        logger.debug("No more interactions found needing titles or summaries.")
        return "done"


    def _summarize_context_files(self) -> Literal["done","not_done"]:
        logger.debug(f"Summarizing context files.")
        if self.prog_bp.project is None:
            logger.error(f"Null project while trying to summarize context files.")
            self.stop()
            return "done"

        for context_file in self.prog_bp.project.context_files.data:
            #previously problematic context files are skipped
            if context_file.filepath in self.skip_context_file_filepaths:
                continue
            
            if context_file.config.is_ignored_by(types.AIAgent.WORKER):
                # not intended for worker, so we skip it
                continue

            if (new_hash := context_file.try_hash()) is None:
                # couldnt read or smth, skip it
                logger.debug(f"Skipping context file {context_file.filepath} because it couldn't be hashed.")
                self.skip_context_file_filepaths.add(context_file.filepath)
                continue

            if context_file.content_hash != new_hash:
                # file has changed
                logger.debug(f"Detected hash change in context file {context_file.filepath}")
                if (new_summary := worker.generate_context_file_summary(self.prog_bp, context_file, headless=True)) is None:
                    logger.debug(f"Skipping context file {context_file.filepath} because summary generation failed.")
                    self.skip_context_file_filepaths.add(context_file.filepath)                    
                    continue

                # ok, update the file
                context_file.config.summary = new_summary
                context_file.content_hash = new_hash
                return "not_done"

        # end of all files
        return "done"
                
    def _update_directory_file(self) -> Literal["done", "not_done"]:
        """Tries to see if the directory file needs updating, and does so if necessary."""
        # this isn't implemented yet
        return "done"
        pass
