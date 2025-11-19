# ghostcode.program
from typing import *
from dataclasses import dataclass, field
import atexit
import traceback
from contextlib import contextmanager
from threading import Thread
import os
import json
import appdirs
import yaml
import logging
import ghostbox.definitions
from ghostbox.definitions import LLMBackend
from ghostbox import Ghostbox
import ghostcode
from .types import *
from . import git
from . import shell
from .soundmanager import SoundManager
from .ansi_colors import Color256, colored
from .utility import (
    show_model_nt,
    timestamp_now_iso8601
)
if TYPE_CHECKING:
    from .idle_worker import IdleWorker, IdleTask
from .ipc import IPCServer
from .ipc_message import IPCMessage, IPCNotification, IPCActions
# --- Logging Setup ---
logger = logging.getLogger("ghostcode.program")

@dataclass
class Program:
    """Holds program state for the main function.
    This instance is passed to command run methods."""

    project_root: Optional[str]
    project: Optional[Project]
    worker_box: Ghostbox
    coder_box: Ghostbox

    tty: shell.VirtualTerminal = field(
        default_factory=lambda: shell.VirtualTerminal(),
    )

    sound_manager: SoundManager = field(
        init = False
    )

    # executes background tasks during interactions
    idle_worker: 'IdleWorker' = field(
        init = False
    )

    # starlette server for inter process communication
    # not every subcommand starts one so it's optional
    ipc_server: Optional[IPCServer] = field(
        default = None
    )

    # holds the actions that are processes during single interactions. FIFO style
    action_queue: List[Action] = field(
        default_factory=lambda: [],
    )

    
    # flag indicating wether the user wants all actions in the queue to be auto-confirmed during a single interaction.
    action_queue_yolo: bool = False

    user_config: UserConfig = field(
        default_factory=UserConfig,
    )

    # holds the tag of the last message that was printed, useful for internal print logic
    last_print_tag: Optional[str] = None
    # always holds the last printed text, or empty string
    last_print_text: str = ""

    # used to color the interface and cannot be relied on except for cosmetics
    cosmetic_state: CosmeticProgramState = field(
        default_factory=lambda: CosmeticProgramState.IDLE
    )

    _DEBUG_DIR: ClassVar[str] = ".ghostcode/debug"
    _LOCKFILE: ClassVar[str] = "interaction.lock"
    _IPC_SERVER_FILE: ClassVar[str] = "ipc_server"
    def __post_init__(self) -> None:
        from .idle_worker import IdleWorker, IdleTask        
        sound_dir = ghostcode.get_ghostcode_data("sounds")
        self.sound_manager = SoundManager(
            sound_directory=sound_dir,
            sound_enabled = self.user_config.sound_enabled,
            volume_multiplier = self.user_config.sound_volume
        )
        # set up idle worker but don't start it yet
        # by default, it does all the tasks in the priority defined in IdleTask
        # FIXME: use user config options for duration etc.
        self.idle_worker = IdleWorker(
            prog_bp = self,
            idle_timeout = 30.0
        )

        
        # Register shutdown to ensure PyAudio resources are released on program exit
        atexit.register(self.sound_manager.shutdown)

    def lock_acquire(self, interaction_history_id: str) -> bool:
        """Acquires a lock on the .ghostcode directory for the current interaction.
        Returns True if the lock could not be acquired (e.g., already locked), False otherwise.
        """
        if self.project_root is None:
            logger.error("Cannot acquire lock: Project root is not set.")
            return True

        lock_filepath = os.path.join(self.get_data(""), self._LOCKFILE)

        if os.path.exists(lock_filepath):
            logger.warning(f"Lock file already exists at {lock_filepath}. Interaction already in progress?")
            return True # Lock already held

        try:
            with open(lock_filepath, "w") as f:
                f.write(interaction_history_id)
            logger.info(f"Lock acquired for interaction {interaction_history_id} at {lock_filepath}.")
            return False # Lock acquired successfully
        except IOError as e:
            logger.error(f"Failed to create lock file {lock_filepath}: {e}")
            return True # Failed to acquire lock

    def lock_release(self) -> None:
        """Releases the lock on the .ghostcode directory.
        """
        if self.project_root is None:
            logger.warning("Cannot release lock: Project root is not set.")
            return

        lock_filepath = os.path.join(self.get_data(""), self._LOCKFILE)

        if os.path.exists(lock_filepath):
            try:
                os.remove(lock_filepath)
                logger.info(f"Lock file {lock_filepath} released.")
            except IOError as e:
                logger.error(f"Failed to delete lock file {lock_filepath}: {e}")
        else:
            logger.debug(f"No lock file found at {lock_filepath} to release.")

    def lock_read(self) -> Optional[str]:
        """Reads the interaction ID from the lock file.
        Returns the interaction ID if the lock file exists, None otherwise or on error.
        """
        if self.project_root is None:
            logger.debug("Cannot read lock: Project root is not set.")
            return None

        lock_filepath = os.path.join(self.get_data(""), self._LOCKFILE)

        if os.path.exists(lock_filepath):
            try:
                with open(lock_filepath, "r") as f:
                    return f.read().strip()
            except IOError as e:
                logger.error(f"Failed to read lock file {lock_filepath}: {e}")
                return None
        return None


    @contextmanager
    def interaction_lock(self, interaction_history_id: str, should_lock: bool = True) -> Generator[None, None, None]:
        """
        Context manager for managing interaction locks.
        Acquires a lock if `should_lock` is True and no other lock is held.
        Releases the lock upon exiting the context.

        Args:
            interaction_history_id (str): The ID of the interaction attempting to acquire the lock.
            should_lock (bool): If True, attempts to acquire a lock. If False, the context manager
                                proceeds without acquiring a lock.

        Yields:
            None

        Raises:
            InteractionLockError: If a lock is required but cannot be acquired (e.g., another interaction is locked).
        """
        if not should_lock:
            logger.debug("Interaction lock skipped as 'should_lock' is False.")
            yield
            return

        # Attempt to acquire the lock
        if self.lock_acquire(interaction_history_id):
            # Lock could not be acquired (another lock exists)
            current_lock_id = self.lock_read()
            error_msg = f"Another interaction (ID: {current_lock_id}) is already active. Please finish or quit it before starting a new interactive session."
            logger.error(error_msg)
            raise InteractionLockError(error_msg)

        try:
            yield
        finally:
            self.lock_release()
    
    def has_git_integration(self) -> bool:
        """Returns true if git integration is enabled for the current project.
            You still have to check for availability yourself, this just allows the project to disable it."""
        if self.project is None:
            return False

        return self.project.config.git_integration
            
    def _get_cli_prompt(self) -> str:
        """Returns the CLI prompt used in the interact command and any other REPL like interactions with the LLMs."""
        git_str = ""
        if self.has_git_integration():
            root = self.project_root if self.project_root else ""
            repo_gr = git.get_repo_name(root)
            branch_gr = git.get_current_branch(root)
            if repo_gr.value or branch_gr.value:
                git_str = f"[{repo_gr.value}:{branch_gr.value}] "
        
        # some ghostbox internal magic to get the token count
        coder_tokens = self.coder_box._plumbing._get_last_result_tokens()
        worker_tokens = self.worker_box._plumbing._get_last_result_tokens()
        return f" 👻{coder_tokens} 🔧{worker_tokens} {git_str}>"

    def _has_api_keys(self) -> Dict[LLMBackend, bool]:
        """
        Compares the chosen backends to the user config and checks for required API keys.

        Returns:
            Dict[LLMBackend, bool]: A dictionary where keys are LLMBackend enum members
                                    and values are True if the API key is present, False otherwise.
                                    Only includes backends that are actually used and require keys.
        """
        if self.project is None:
            error_msg = (
                "Attempting to verify API keys with uninitialized project. Aborting."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        missing_keys: Dict[LLMBackend, bool] = {}

        # Check Coder LLM backend
        coder_backend_str = self.project.config.coder_backend
        if coder_backend_str == LLMBackend.google.name:
            if not self.user_config.google_api_key:
                missing_keys[LLMBackend.google] = False
            # else: # No need to add if key is present, we only care about missing ones
            #     missing_keys[LLMBackend.google] = True
        elif coder_backend_str == LLMBackend.openai.name:
            if not self.user_config.openai_api_key:
                missing_keys[LLMBackend.openai] = False
            # else:
            #     missing_keys[LLMBackend.openai] = True
        elif coder_backend_str == LLMBackend.generic.name:
            # For generic, if the endpoint is OpenAI or Google, we might need the general api_key
            # This is a heuristic, as generic can point to anything.
            # For now, we'll only check if the endpoint looks like OpenAI/Google official ones
            if "openai.com" in self.project.config.coder_endpoint:
                if not self.user_config.openai_api_key and not self.user_config.api_key:
                    missing_keys[LLMBackend.openai] = (
                        False  # Assume OpenAI key is preferred for OpenAI endpoints
                    )
                # else:
                #     missing_keys[LLMBackend.openai] = True
            elif "googleapis.com" in self.project.config.coder_endpoint:
                if not self.user_config.google_api_key and not self.user_config.api_key:
                    missing_keys[LLMBackend.google] = (
                        False  # Assume Google key is preferred for Google endpoints
                    )
                # else:
                #     missing_keys[LLMBackend.google] = True
            # If generic points to a local server (e.g., localhost:8080), no API key is expected.

        # Check Worker LLM backend
        worker_backend_str = self.project.config.worker_backend
        if worker_backend_str == LLMBackend.google.name:
            if not self.user_config.google_api_key:
                # Only add if not already marked as missing by coder_backend
                if LLMBackend.google not in missing_keys:
                    missing_keys[LLMBackend.google] = False
        elif worker_backend_str == LLMBackend.openai.name:
            if not self.user_config.openai_api_key:
                # Only add if not already marked as missing by coder_backend
                if LLMBackend.openai not in missing_keys:
                    missing_keys[LLMBackend.openai] = False
        elif worker_backend_str == LLMBackend.generic.name:
            if "openai.com" in self.project.config.worker_endpoint:
                if not self.user_config.openai_api_key and not self.user_config.api_key:
                    if LLMBackend.openai not in missing_keys:
                        missing_keys[LLMBackend.openai] = False
            elif "googleapis.com" in self.project.config.worker_endpoint:
                if not self.user_config.google_api_key and not self.user_config.api_key:
                    if LLMBackend.google not in missing_keys:
                        missing_keys[LLMBackend.google] = False

        # Filter out True entries, only return missing ones
        return {k: v for k, v in missing_keys.items() if not v}

    def discard_actions(self) -> None:
        """Empties the action queue."""
        if not (self.action_queue):
            logger.debug(f"Discard on empty action queue.")
            return

        actions_str = "\n".join(
            [json.dumps(action.model_dump(), indent=4) for action in self.action_queue]
        )
        logger.debug(
            f"Discard on action queue. Will discard the following actions:\n{actions_str}"
        )
        self.action_queue = []

    def queue_action(self, action: Action) -> None:
        """Queues an action at the end of the action queue."""
        logger.info(f"Queueing action {type(action)}.")
        self.idle_worker.update()
        self.action_queue.append(action)

    def push_front_action(self, action: Action) -> None:
        """Pushes an action to the front of the queue. An action at the front will be executed before the remaining ones."""
        logger.info(
            f"Pushing action {action_show_short(action)} to the front of the action queue."
        )
        self.idle_worker.update()
        self.action_queue = [action] + self.action_queue

    def confirm_action(
        self,
        action: Action,
        agent_clearance_level: ClearanceRequirement = ClearanceRequirement.AUTOMATIC,
        agent_name: str = "System",
    ) -> UserConfirmation:
        """Interactively acquire user confirmation for a given action.
        When an action needs to be confirmed, there is usually some agent who wants to perform the action (e.g. ghostcoder or ghostworker). The agent's current clearance level is provided for context.
        """
        self.sound_notify()
        self.print(f"{agent_name} wants to perform {action_show_short(action)}.")
        abridge = 80  # type: Optional[int]
        try:
            while True:
                self.idle_worker.update()                
                choice = input(
                    "Permit? yes (y), no (n), yes to all (a), cancel all (q), or show more info (?, default):"
                )
                match choice:
                    case "y":
                        logger.info("User confirmed action.")
                        return UserConfirmation.YES
                    case "n":
                        logger.info("User denied action.")
                        return UserConfirmation.NO
                    case "a":
                        # FIXME: this should implement a default confirm for all confirmation dialogs that happen during the current run_action_queue execution.
                        logger.info("YOLO")
                        return UserConfirmation.ALL
                    case "q":
                        logger.info("User canceled all action requests.")
                        return UserConfirmation.CANCEL
                    case "d":
                        # secret debug option
                        print(
                            show_model_nt(
                                action, heading=action.__class__.__name__, abridge=None
                            )
                        )
                        continue
                    case _:
                        # cover "?", empty input, and anything else, as showing more nformation is generally a safe option.
                        self.print(
                            action_show_confirmation_data(
                                action,
                                first_showing=(abridge is not None),
                                abridge=abridge,
                            )
                        )
                        # we only show abridge once. user can do enter twice to see full strings
                        abridge = None
                        continue
        except EOFError as e:
            self.print(f"Canceled.")
            logger.info(
                f"User exited confirmation dialog with EOF. Defaulting to deny."
            )
            return UserConfirmation.CANCEL
        except Exception as e:
            self.print(f"Action canceled. Error: {e}.")
            logger.error(
                f"Encountered error during user confirmation dialog. Defaulting to deny. Error: {e}"
            )
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return UserConfirmation.CANCEL

        # unreachable but ok
        logger.info(f"Follow the white rabbit.")
        return UserConfirmation.CANCEL

    def print(
        self, text: str, end: str = "\n", tag: Optional[str] = None, flush: bool = True
    ) -> None:
        """Print a message to stdout.
        If you tag your print message, this print remembers and an automatic newline is inserted before another message that does not share the tag. This allows you to build up a line without interfereing in other print actions.
        """
        # we hijack the normal print so that we can more easily change things in the future
        # e.g. make it thread safe or keep a log.
        # right now it's just easy print

        if tag != self.last_print_tag and self.last_print_tag is not None:
            if self.last_print_tag == "action_queue":
                print("\r", end="", flush=True)
            else:
                print("")

        print(text, end=end, flush=flush)
        self.last_print_text = text
        self.last_print_tag = tag

    def make_tagged_printer(self, tag: str) -> Callable[[str], None]:
        return lambda text, end="\n", flush=True: self.print(text, end=end, flush=flush, tag=tag)  # type: ignore

    def announce(self, text: str, *, agent: AIAgent) -> None:
        """Alternative to print that announces some kind of system event to the user from the perspective of an AI agent.
        This method is intended to be used by deeply nested processing in the action queue to keep the user informed when unusual things happen. Announcements should therefore use simple language, and be used in addition to logging, not instead of it.
        """
        delimiter = colored(">>=", self.cosmetic_state.to_color())
        match agent:
            case AIAgent.WORKER:
                self.print(f" 🔧 {delimiter} {text}")
            case AIAgent.CODER:
                self.print(f" 👻 {delimiter} {text}")
            case _ as unreachable:
                assert_never(unreachable)

    def get_data(self, relative_filepath: str) -> str:
        """Returns a filepath relative to the hidden ghostcode directory.
            Example: get_data("log.txt") -> ".ghostcode/log.txt"
        If no project root directory is defined, this function raises a runtime error.
        If the filepath that this function returns does not exist, no error is raised.
        """
        if self.project_root is None or not (
            os.path.isdir(
                ghostcode_dir := os.path.join(self.project_root, ".ghostcode")
            )
        ):
            msg = "Could not find .ghostcode directory. Please run `ghostcode init` and retry."
            logger.error(msg)
            raise RuntimeError(msg)

        return os.path.join(ghostcode_dir, relative_filepath)

    def get_config_or_default(self) -> ProjectConfig:
        """Returns the project config or a default config if retrieving the project config fails."""
        fail = ProjectConfig()

        if self.project is None:
            logger.warning(
                "Null project when trying to get config. This means that a default config is used."
            )
            return fail

        return self.project.config

    def show_log(self, tail: Optional[int] = None) -> str:
        """Returns the current event log if available.
        If tail is given, show only the tail latest number of log lines.
        """
        if self.project is None:
            logger.warn("Null project while trying to read logs.")
            return ""
        try:
            with open(self.get_data(self.project._LOG_FILE), "r") as f:
                log = f.read()

            if tail is None:
                return log

            return "\n".join([line for line in log.splitlines()][(-1) * tail :])

        except Exception as e:
            logger.warning(f"Failed to read log file. Reason: {e}")
            return ""


    @contextmanager        
    def sound_clicks(self, mean: float = 0.7) -> Generator[None, None, None]:
        """Plays clicking sounds if sound_enabled is true on the sound manager."""
        clicking_sounds = "clicks1.wav clicks2.wav clicks3.wav clicks_double.wav clicks_triple.wav".split(" ")
        with self.sound_manager.continuous_playback(
                clicking_sounds,
                mean = mean):
            yield

    def sound_error(self) -> None:
        self.sound_manager.play("error.wav")

    def sound_notify(self) -> None:
        self.sound_manager.play("opening1.wav")

    @contextmanager
    def idle_work(self) -> Generator[None, None, None]:
        """Context manager that starts the idle worker on entry and stops it on exit."""
        logger.debug("Entering idle_work context. Starting idle worker.")
        self.idle_worker.start()
        try:
            yield
        finally:
            logger.debug("Exiting idle_work context. Stopping idle worker.")
            self.idle_worker.stop()

    def debug_dump(self) -> None:
        """Save some debugging output into .ghostcode/debug/"""
        # FIXME: make this conditional on self.debug which should be set with --debug
        if self.project_root is None:
            logger.error(f"Cannot dump debug information: Project root is null.")
            return

        debug_dir = os.path.join(self.project_root, self._DEBUG_DIR)
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception as e:
            logger.exception(
                f"Couldn't create directories {debug_dir} while dumping debug information. Reason: {e}"
            )

        # we dump the logs for both boxes
        def save_box_history(box_name: str, box: Ghostbox) -> None:
            filename = f"{box_name}-{timestamp_now_iso8601()}"
            content = "\n---\n".join(
                [show_model_nt(chat_message) for chat_message in box.get_history()]
            )
            try:
                with open(os.path.join(debug_dir, filename), "w") as f:
                    f.write(content)
            except Exception as e:
                logger.exception(f"Couldn't dump {box_name} history. Reason: {e}")

        save_box_history("coder_box", self.coder_box)
        save_box_history("worker_box", self.worker_box)

    def get_current_backend_coder(self) -> LLMBackend:
        """Returns the current LLM backend for the coder.
        The current backend is determined in the order of options in: command line > user config > project config > defaults"""
        return self.coder_box.get("backend")
    
    def get_current_backend_worker(self) -> LLMBackend:
        """Returns the current LLM backend for the worker.
        The current backend is determined in the order of options in: command line > user config > project config > defaults"""
        return self.worker_box.get("backend")

    def get_current_model_coder(self) -> str:
        """Returns the currently used LLM model for the coder (if any is set). This is not the same as the backend.
        The model is determined in the order of: command line > user config > project config > default"""
        return self.coder_box.get("model")

    def get_current_model_worker(self) -> str:
        """Returns the currently used LLM model used by the worker (if any is set). This is not the same as the backend.
        The model is determined in the order of: command line > user config > project config > default"""
        return self.worker_box.get("model")    

    def get_branch_interactions(self, target_branch: Optional[str] = None) -> List[InteractionHistory]:
        """Returns interactions for a given branch, excluding the current interaction by default if on is ongoing.
        If no branch name is provided, returns interactions for the current git branch by default.
        If git is disabled or branch name cannot be determined, all interactions are returned."""
        # sanity
        if self.project is None:
            logger.error(f"Null project during interaction retrieval.")
            return []

        current_interaction_id = self.lock_read()
        exclude_ids = [current_interaction_id] if current_interaction_id is not None else []
        if exclude_ids == []:
            logger.debug(f"Unable to determine current interaction ID while retrieving branch interactions.")

        # git stuff is handled by the project method
        return self.project.get_branch_interactions(exclude_interaction_ids = exclude_ids)    

    def get_speaker_box(self) -> Ghostbox:
        """Returns a clone of the worker box, configured for TTS.
        This is useful as the speaker box will have various TTS options set, and is ideal for text streaming directly to TTS."""
        logger.info(f"Constructing speaker box.")
        
        if self.project is None:
            logger.error(f"Null project on trying to create speaker box.")
            raise RuntimeError(f"No project root. Please initialize a ghostbox project with `ghostbox init`")

        # assemble user overrides
        user_options = default_tts_options
        user_options["tts_model"] = self.user_config.tts_model
        user_options["tts_voice"] = self.user_config.tts_voice
        if not self.user_config.sound_enabled:
            logger.warning(f"Disabling sound on speaker box because of user disabled sound option.")
            user_options["tts"] = False
            user_options["quiet"] = True
        
        worker_options = self.worker_box.get_options()
        # FIXME: bug inghostbox where it must be initialized with tts = True
        # FIXME: we should probably, wether ghostbox fixes the bug or not, initialize speaker from a dedicated speaker character folder
        speaker_box = Ghostbox(**(worker_options | {"tts":True}))
        # we do it this way because part of the worker options is 'character_folder' which will override options in the kwargs
        # this we we know 100% that the tts user options are in
        for k, v in user_options.items():
            speaker_box.set(k, v)
        logger.debug(f"Initializing speaker_box with the following options:\n{json.dumps(speaker_box.get_options(), indent=4)}")
        return speaker_box

    def _ipc_server_info_save(self, host: str, port: int) -> None:
        """Writes the connection information for a running IPC server to a file."""
        if self.project_root is None:
            logger.error("Cannot save IPC server info: Project root is not set.")
            return

        ipc_info_filepath = os.path.join(self.get_data(""), self._IPC_SERVER_FILE)
        try:
            with open(ipc_info_filepath, "w") as f:
                json.dump({"host": host, "port": port}, f)
            logger.info(f"IPC server info saved to {ipc_info_filepath}: {host}:{port}")
        except IOError as e:
            logger.error(f"Failed to save IPC server info to {ipc_info_filepath}: {e}")

    def _ipc_server_info_get(self) -> Optional[Tuple[str, int]]:
        """Tries to read the IPC server info from the _IPC_SERVER_FILE and returns a pair of < host, port >, and None if it can't be read."""
        if self.project_root is None:
            logger.debug("Cannot get IPC server info: Project root is not set.")
            return None

        ipc_info_filepath = os.path.join(self.get_data(""), self._IPC_SERVER_FILE)

        if not os.path.exists(ipc_info_filepath):
            logger.debug(f"IPC server info file not found at {ipc_info_filepath}.")
            return None

        try:
            with open(ipc_info_filepath, "r") as f:
                data = json.load(f)
            host = data.get("host")
            port = data.get("port")
            if isinstance(host, str) and isinstance(port, int):
                logger.debug(f"Retrieved IPC server info from {ipc_info_filepath}: {host}:{port}")
                return host, port
            else:
                logger.error(f"Invalid data in IPC server info file {ipc_info_filepath}.")
                return None
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read or decode IPC server info from {ipc_info_filepath}: {e}")
            return None

    def _ipc_server_info_clear(self) -> None:
        """Removes the IPC server info file if it exists."""
        if self.project_root is None:
            logger.debug("Cannot clear IPC server info: Project root is not set.")
            return

        ipc_info_filepath = os.path.join(self.get_data(""), self._IPC_SERVER_FILE)

        if os.path.exists(ipc_info_filepath):
            try:
                os.remove(ipc_info_filepath)
                logger.info(f"IPC server info file {ipc_info_filepath} cleared.")
            except IOError as e:
                logger.error(f"Failed to remove IPC server info file {ipc_info_filepath}: {e}")
        else:
            logger.debug(f"No IPC server info file found at {ipc_info_filepath} to clear.")

    def _ipc_server_handle_message(self, message: IPCMessage) -> None:
        """Default message handler that can be passed to the IPC server as a callback."""
        import worker
        logger.debug(f"Handling IPC message: {message}")

        def client_str(client_name: str) -> str:
            return f"[{client_name}] " if client_name else ""
        
        match message:
            case IPCNotification() as notification_msg:
                self.print(f"{client_str(notification_msg.client)}{notification_msg.text}")
            case IPCActions() as ipc_actions_msg:
                if ipc_actions_msg.text:
                    self.print(f"{client_str(ipc_actions_msg.client)}{ipc_actions_msg.text}")

                # All actions received via IPC are considered to be from the current interaction
                # This ensures they are properly associated if an interaction is active
                current_interaction_id = self.lock_read()
                for action in ipc_actions_msg.actions:
                    # For Query actions, set the interaction_history_id if an interaction is active
                    if isinstance(action, (ActionQueryCoder, ActionQueryWorker)) and current_interaction_id:
                        action.interaction_history_id = current_interaction_id
                    self.queue_action(action)
                    
                # Run the action queue in a separate thread to avoid blocking the IPC server's event loop
                # This is crucial because run_action_queue can be long-running and interactive.
                action_thread = Thread(target=worker.run_action_queue, args=(self,))
                action_thread.daemon = True # Allow the main program to exit even if this thread is still running
                action_thread.start()
                
    def start_ipc_server(self) -> None:
        """Initializes the IPC server with a default message handler and writes host/port to the IPC server info file."""
        if self.ipc_server is not None:
            logger.warning("IPC server already running.")
            return

        self.ipc_server = IPCServer()
        try:
            host, port = self.ipc_server.start(self._ipc_server_handle_message)
            self._ipc_server_info_save(host, port)
            logger.info(f"IPC server started successfully on {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start IPC server: {e}", exc_info=True)
            self.ipc_server = None # Clear server if startup failed
            self._ipc_server_info_clear() # Ensure no stale info is left
