from typing import *
import json
import traceback
from ghostcode import types, worker
from . import git
from ghostcode import slash_commands
from ghostcode.progress_printer import ProgressPrinter
from ghostcode import prompts
from ghostcode.utility import show_model_nt, EXTENSION_TO_LANGUAGE_MAP, language_from_extension
# don't want the types. prefix for these
from ghostcode.types import Program
import ghostbox
import os
from ghostbox import Ghostbox, ChatMessage
from ghostbox.definitions import BrokenBackend
from ghostbox.commands import showTime
from dataclasses import dataclass, field
import argparse
import logging
import os
import glob
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import sys
import json
import yaml
import appdirs
from ghostbox.definitions import LLMBackend  # Added for API key checks
from queue import Queue, Empty
import re # Added for DiscoverCommand regex filtering

# logger will be configured after argument parsing
logger: logging.Logger  # Declare logger globally, will be assigned later


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


# --- Command Interface and Implementations ---


class CommandOutput(BaseModel):
    """Helper class to package output from commands.
    Mostly will just contain text, but this exists so we can add JSON etc. in the future.
    """

    text: str = ""
    data: Dict[str, Any] = Field(default_factory=lambda: {})

    def print(self, w: str, end: str = "\n") -> None:
        """Mimics print function by concatinating text."""
        self.text += w + end


class CommandInterface(ABC):
    """Abstract base class for all ghostcode commands."""

    @abstractmethod
    def run(self, prog: Program) -> CommandOutput:
        pass


class InitCommand(BaseModel, CommandInterface):
    """Initializes a new ghostcode project."""

    path: Optional[str] = Field(
        default=None,
        description="Optional path to initialize the project. Defaults to current directory.",
    )

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        target_root = os.path.abspath(self.path if self.path else os.getcwd())
        try:
            types.Project.init(target_root)
            result.text += (
                f"Ghostcode project initialized successfully in {target_root}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize project in {target_root}: {e}")
            sys.exit(1)

        return result


class ConfigCommand(BaseModel, CommandInterface):
    """Manages ghostcode project configuration."""

    subcommand: Literal["set", "get", "ls"]
    key: Optional[str] = Field(
        default=None,
        description="The configuration key to retrieve or set (e.g., 'coder_llm_config.model', 'config.coder_backend').",
    )
    value: Optional[str] = Field(
        default=None, description="The new value for the configuration key."
    )

    def _get_config_value(self, project: types.Project, key: str) -> Any:
        """Helper to get config value from nested dictionaries/models."""
        parts = key.split(".", 1)
        if len(parts) == 1:  # Top-level key (e.g., 'name' in metadata)
            if hasattr(project.project_metadata, key):
                return getattr(project.project_metadata, key)
            logger.warning(
                f"Config key '{key}' not found at top level of project metadata."
            )
            return None

        # Nested key (e.g., 'coder_llm_config.model', 'config.coder_backend')
        parent_key, sub_key = parts
        if parent_key == "coder_llm_config":
            return project.coder_llm_config.get(sub_key)
        elif parent_key == "worker_llm_config":
            return project.worker_llm_config.get(sub_key)
        elif parent_key == "project_metadata" and project.project_metadata:
            return getattr(project.project_metadata, sub_key, None)
        elif parent_key == "config":  # Handle ProjectConfig
            if hasattr(project.config, sub_key):
                # Now that backend fields are strings, just return the value directly
                return getattr(project.config, sub_key)
            logger.warning(f"Sub-key '{sub_key}' not found in project.config.")
            return None

        logger.warning(
            f"Config key '{key}' not found or invalid parent key '{parent_key}'."
        )
        return None

    def _set_config_value(
        self, project: types.Project, key: str, value_str: str
    ) -> None:
        """Helper to set config value in nested dictionaries/models."""
        parsed_value = self._parse_value(value_str)
        parts = key.split(".", 1)

        if len(parts) == 1:  # Top-level key in metadata
            if hasattr(project.project_metadata, key):
                setattr(project.project_metadata, key, parsed_value)
                logger.debug(f"Set project_metadata.{key} to {parsed_value}")
            else:
                logger.warning(
                    f"Config key '{key}' not found at top level of project metadata for setting."
                )
            return

        parent_key, sub_key = parts
        if parent_key == "coder_llm_config":
            project.coder_llm_config[sub_key] = parsed_value
            logger.debug(f"Set coder_llm_config.{sub_key} to {parsed_value}")
        elif parent_key == "worker_llm_config":
            project.worker_llm_config[sub_key] = parsed_value
            logger.debug(f"Set worker_llm_config.{sub_key} to {parsed_value}")
        elif parent_key == "project_metadata" and project.project_metadata:
            if hasattr(project.project_metadata, sub_key):
                setattr(project.project_metadata, sub_key, parsed_value)
                logger.debug(f"Set project_metadata.{sub_key} to {parsed_value}")
            else:
                logger.warning(
                    f"Sub-key '{sub_key}' not found in project_metadata for setting."
                )
        elif parent_key == "config":  # Handle ProjectConfig
            if hasattr(project.config, sub_key):
                # No special conversion needed here, as ProjectConfig fields are now strings
                setattr(project.config, sub_key, parsed_value)
                logger.debug(f"Set project.config.{sub_key} to {parsed_value}")
            else:
                logger.warning(
                    f"Sub-key '{sub_key}' not found in project.config for setting."
                )
        else:
            logger.warning(f"Config key '{key}' not recognized for setting.")

    def _parse_value(self, value_str: str) -> Any:
        """Attempt to parse string to int, float, bool, or keep as string."""
        try:
            return json.loads(
                value_str
            )  # Try parsing as JSON (handles bool, int, float, list, dict, null)
        except json.JSONDecodeError:
            return value_str  # If not JSON, treat as string

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        project = prog.project

        if self.subcommand == "ls":
            result.print("Project Configuration (config.yaml):")
            for k, v in project.config.model_dump().items():
                # Now that backend fields are strings, just print them directly
                result.print(f"  {k}: {v}")
            result.print("\nCoder LLM Config (coder/config.json):")
            for k, v in project.coder_llm_config.items():
                result.print(f"  {k}: {v}")
            result.print("\nWorker LLM Config (worker/config.json):")
            for k, v in project.worker_llm_config.items():
                result.print(f"  {k}: {v}")
            result.print("\nProject Metadata (project_metadata.yaml):")
            if project.project_metadata:
                for k, v in project.project_metadata.model_dump().items():
                    result.print(f"  {k}: {v}")

            # Also list user config
            result.print("\nUser Configuration (.ghostcodeconfig):")
            for k, v in prog.user_config.model_dump().items():
                # Mask API keys for display
                if "api_key" in k and v:
                    result.print(f"  {k}: {v[:4]}...{v[-4:]}")
                else:
                    result.print(f"  {k}: {v}")

        elif self.subcommand == "get":
            if not self.key:
                logger.error("Key required for 'config get'.")
                sys.exit(1)

            # First try project config
            val = self._get_config_value(project, self.key)
            if val is not None:
                result.print(f"{self.key}: {val}")
            else:
                # Then try user config
                if hasattr(prog.user_config, self.key):
                    val = getattr(prog.user_config, self.key)
                    if "api_key" in self.key and val:
                        result.print(f"{self.key}: {val[:4]}...{val[-4:]}")
                    else:
                        result.print(f"{self.key}: {val}")
                else:
                    logger.warning(
                        f"Config key '{self.key}' not found in project or user configuration."
                    )
        elif self.subcommand == "set":
            if not self.key or self.value is None:
                logger.error("Key and value required for 'config set'.")
                sys.exit(1)

            # Try setting in project config first
            # This is a bit tricky as _set_config_value doesn't return success/failure
            # We'll assume if it's not in project config, it might be user config
            initial_project_config_dump = project.config.model_dump_json()
            initial_coder_llm_config_dump = json.dumps(project.coder_llm_config)
            initial_worker_llm_config_dump = json.dumps(project.worker_llm_config)
            initial_project_metadata_dump = (
                project.project_metadata.model_dump_json()
                if project.project_metadata
                else "{}"
            )

            self._set_config_value(project, self.key, self.value)

            # Check if any project config was actually changed
            project_config_changed = (
                project.config.model_dump_json() != initial_project_config_dump
                or json.dumps(project.coder_llm_config) != initial_coder_llm_config_dump
                or json.dumps(project.worker_llm_config)
                != initial_worker_llm_config_dump
                or (
                    project.project_metadata
                    and project.project_metadata.model_dump_json()
                    != initial_project_metadata_dump
                )
            )

            if project_config_changed:
                project.save_to_root(prog.project_root)
                result.print(
                    f"Project config key '{self.key}' set to '{self.value}' and saved."
                )
            elif hasattr(prog.user_config, self.key):
                # If not a project config key, try user config
                setattr(prog.user_config, self.key, self._parse_value(self.value))
                prog.user_config.save()
                result.print(
                    f"User config key '{self.key}' set to '{self.value}' and saved."
                )
            else:
                logger.warning(
                    f"Config key '{self.key}' not found in project or user configuration for setting."
                )
        return result


class ContextCommand(BaseModel, CommandInterface):
    """Manages files included in the project context."""

    subcommand: Literal["add", "rm", "remove", "ls", "clean", "lock", "unlock", "wipe"]
    filepaths: List[str] = Field(
        default_factory=list,
        description="File paths to add or remove, can include wildcards.",
    )
    rag: bool = Field(
        default=False,
        description="Enable Retrieval Augmented Generation for added files.",
    )

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        project = prog.project
        current_context_files = {cf.filepath: cf for cf in project.context_files.data}

        if self.subcommand == "ls":
            if not project.context_files.data:
                result.print("No context files tracked.")
            else:
                result.print("Tracked Context Files (relative to project root):")
                result.print(project.context_files.show_overview())
        elif self.subcommand in ["add", "rm", "remove"]:
            if not self.filepaths:
                logger.error(f"File paths required for 'context {self.subcommand}'.")
                sys.exit(1)

            resolved_paths = set()
            for pattern in self.filepaths:
                # Resolve relative to CWD, then convert to relative to project_root
                abs_pattern = os.path.abspath(pattern)

                # Use glob to handle wildcards, recursive for '**'
                for matched_path in glob.glob(abs_pattern, recursive=True):
                    if os.path.isfile(matched_path):
                        # Convert absolute path to path relative to project_root
                        relative_to_project_root = os.path.relpath(
                            matched_path, start=prog.project_root
                        )
                        resolved_paths.add(relative_to_project_root)
                    else:
                        logger.warning(
                            f"Skipping '{matched_path}': Not a file or does not exist."
                        )

            if self.subcommand == "add":
                for fp in resolved_paths:
                    if fp not in current_context_files:
                        project.context_files.add(
                            fp
                        )
                        result.print(f"Added '{fp}' to context (RAG={self.rag}).")
            elif self.subcommand in ["rm", "remove"]:
                initial_count = len(project.context_files.data)
                project.context_files.data = [
                    cf
                    for cf in project.context_files.data
                    if cf.filepath not in resolved_paths
                ]
                removed_count = initial_count - len(project.context_files.data)
                if removed_count > 0:
                    result.print(f"Removed {removed_count} file(s) from context.")
                else:
                    result.print("No matching files found in context to remove.")

            project.save_to_root(prog.project_root)
            result.print("Context files updated and saved.")
        elif self.subcommand in ["clean"]:
            result.print("Removing bogus files from context.")
            cfs = prog.project.context_files.data
            new_cfs = []
            for cf in cfs:
                if not os.path.isfile(cf.get_abs_filepath()):
                    result.print(f" - Removing {cf.filepath}")
                    continue
                else:
                    new_cfs.append(cf)
            prog.project.context_files.data = new_cfs
            project.save_to_root(prog.project_root)
            result.print("Context files updated and saved.")
        elif self.subcommand in ["lock", "unlock"]:
            if not self.filepaths:
                logger.error(f"File paths required for 'context {self.subcommand}'.")
                sys.exit(1)

            resolved_paths = set()
            for pattern in self.filepaths:
                abs_pattern = os.path.abspath(pattern)
                for matched_path in glob.glob(abs_pattern, recursive=True):
                    if os.path.isfile(matched_path):
                        relative_to_project_root = os.path.relpath(
                            matched_path, start=prog.project_root
                        )
                        resolved_paths.add(relative_to_project_root)
                    else:
                        logger.warning(
                            f"Skipping '{matched_path}': Not a file or does not exist."
                        )

            action_word = "locked" if self.subcommand == "lock" else "unlocked"
            target_lock_status = True if self.subcommand == "lock" else False
            modified_count = 0

            for fp in resolved_paths:
                found = False
                for cf in project.context_files.data:
                    if cf.filepath == fp:
                        if cf.config.locked != target_lock_status:
                            cf.config.locked = target_lock_status
                            result.print(f"File '{fp}' {action_word}.")
                            modified_count += 1
                        else:
                            result.print(f"File '{fp}' is already {action_word}. Skipping.")
                        found = True
                        break
                if not found:
                    result.print(f"File '{fp}' not found in context. Skipping.")
            
            if modified_count > 0:
                project.save_to_root(prog.project_root)
                result.print("Context file lock statuses updated and saved.")
            else:
                result.print("No context files modified.")
        elif self.subcommand == "wipe":
            if not project.context_files.data:
                result.print("Context is already empty. No files to wipe.")
            else:
                n = len(project.context_files.data)
                project.context_files.data = []
                project.save_to_root(prog.project_root)
                result.print("{n} file(s) removed from context.")
        return result

class DiscoverCommand(BaseModel, CommandInterface):    
    """Intelligently discovers files associated with the project and adds them to the context."""

    filepath: str = Field(
        description = "The filepath that points to a directory in which files to add to the context will be recursively discovered."
    )
    
    languages_enabled: List[str] = Field(
        default_factory = list,
        description = "Programming languages (e.g. 'python', 'javascript') for which files in the respective language will be added to the context."
    )


    languages_disabled: List[str] = Field(
        default_factory = list,
        description = "Providing e.g. --no-python --no-javascript will exclude languages from the discovery."
    )
    min_lines: Optional[int] = Field(
        default = None,
        description = "Minimum amount of lines that a file must have in order to be added to the context."
    )

    max_lines: Optional[int] = Field(
        default = None,
        description = "Maximum number of lines a file can have in order to be added to the context."
    )

    size_heuristic: Optional[Literal["small", "medium", "large"]] = Field(
        default = None,
        description = "If provided with --small, --medium, or --large, will automatically set the min_lines and max_lines parameters. Small means 0 - 1000 lines. Medium is 300 - 3000 lines. Large is 3000+. The small, medium, and large parameters are mutually exclusive."
    )

    exclude_pattern: str = Field(
        default = "",
        description = "A regex that can be provided. File names that match against it are excluded from the context. The exclude pattern is applied at the very end of the discovery process."
    )
    
    all: bool = Field(
        default = False,
        description = "If provided, will add all (non-hidden) files to the context that are found in subdirectories of the given path. Providing --all is like providing all of the language parameters."
    )

    # this containts e.g. "python", "javascript, "cpp". It should be used to create argparse parameters, like --python, --javascript, --cpp and so on 
    possible_languages: ClassVar[List[str]] = list(set(list(EXTENSION_TO_LANGUAGE_MAP.values())))

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        project = prog.project
        target_abs_path = os.path.abspath(self.filepath)

        if not os.path.isdir(target_abs_path):
            logger.error(f"Error: Path '{self.filepath}' is not a directory.")
            sys.exit(1)

        discovered_files: List[str] = []
        
        # Determine effective min_lines and max_lines
        effective_min_lines = self.min_lines
        effective_max_lines = self.max_lines
        if self.size_heuristic == "small":
            effective_min_lines = 0
            effective_max_lines = 1000
        elif self.size_heuristic == "medium":
            effective_min_lines = 300
            effective_max_lines = 3000
        elif self.size_heuristic == "large":
            effective_min_lines = 3000
            effective_max_lines = None # No upper limit for large

        # Compile exclude pattern if provided
        exclude_regex = None
        if self.exclude_pattern:
            try:
                exclude_regex = re.compile(self.exclude_pattern)
            except re.error as e:
                logger.error(f"Invalid regex for --exclude-pattern: {e}")
                sys.exit(1)

        # Prepare language filters
        enabled_languages_set = set(self.languages_enabled)
        disabled_languages_set = set(self.languages_disabled)
        
        # If --all is provided, enable all possible languages first
        if self.all:
            enabled_languages_set.update(self.possible_languages)

        # Remove explicitly disabled languages
        enabled_languages_set = enabled_languages_set - disabled_languages_set

        for root, dirs, files in os.walk(target_abs_path):
            # Filter out directories starting with '_' to prevent recursion into them
            dirs[:] = [d for d in dirs if not d.startswith('_')]

            for filename in files:
                abs_filepath = os.path.join(root, filename)
                relative_filepath = os.path.relpath(abs_filepath, start=prog.project_root)

                # Skip hidden files/directories (e.g., .git, .ghostcode, files starting with .)
                if any(part.startswith('.') for part in relative_filepath.split(os.sep)):
                    logger.debug(f"Skipping hidden file: {relative_filepath}")
                    continue
                
                # 1. Apply language filter
                file_language = language_from_extension(abs_filepath)
                if file_language not in enabled_languages_set:
                    logger.debug(f"Skipping '{relative_filepath}': Language '{file_language}' not enabled.")
                    continue

                # 2. Apply line count filter
                try:
                    with open(abs_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                except Exception as e:
                    logger.warning(f"Could not read file '{relative_filepath}' for line count: {e}. Skipping.")
                    continue

                if effective_min_lines is not None and line_count < effective_min_lines:
                    logger.debug(f"Skipping '{relative_filepath}': {line_count} lines, less than min_lines ({effective_min_lines}).")
                    continue
                if effective_max_lines is not None and line_count > effective_max_lines:
                    logger.debug(f"Skipping '{relative_filepath}': {line_count} lines, more than max_lines ({effective_max_lines}).")
                    continue

                # 3. Apply exclude pattern filter
                if exclude_regex and exclude_regex.search(filename):
                    logger.debug(f"Skipping '{relative_filepath}': Matches exclude pattern '{self.exclude_pattern}'.")
                    continue

                discovered_files.append(relative_filepath)
                result.print(f"Discovered and added '{relative_filepath}' to context.")

        # Add all discovered files to the project context
        added_count = 0
        for fp in discovered_files:
            # Use add_or_alter to handle existing files gracefully
            # For discover, we don't set RAG by default, but it could be an option later
            if not project.context_files.has_filepath(fp):
                project.context_files.add_or_alter(fp)
                added_count += 1
            else:
                result.print(f"'{fp}' already in context. Skipping.")

        project.save_to_root(prog.project_root)
        result.print(f"Discovery complete. Added {added_count} new files to context.")
        return result

class LogCommand(BaseModel, CommandInterface):
    """Manages and displays interaction history."""

    interaction_identifier: Optional[str] = Field(
        default=None,
        description="Optional unique ID or tag of a specific interaction to display in detail.",
    )

    all_branches: bool = Field(
        default = False,
        description = "Do not filter any interactions based on branches."
    )
    
    def _overview_show_interaction(
        self, interaction: types.InteractionHistory, num_turns: int
    ) -> str:
        turns_str = f"turns: {num_turns}\n"
        tag_str = f"tag: {interaction.tag}\n" if interaction.tag else ""
        branch_str = f"branch: {interaction.branch_name}\n" if interaction.branch_name is not None else ""
        git_str = (
            f"commits affected: {", ".join(commits)}\n"
            if (commits := interaction.get_affected_git_commits()) != []
            else ""
        )
        title_str = f"title: {interaction.title}\n" if interaction.title else ""
        if (timestamps := interaction.timestamps()) is not None:
            time_str = (
                f"date started: {timestamps[0]}\nlast interaction: {timestamps[1]}\n"
            )
        else:
            time_str = ""

        if interaction.empty():
            first_msg = ""
        else:
            match interaction.contents[0]:
                case types.UserInteractionHistoryItem() as user_item:
                    first_msg = user_item.prompt.strip()
                case _:
                    first_msg = ""

            if first_msg:
                limit = 120
                if len(first_msg) >= limit:
                    first_msg = first_msg[:limit].strip() + "..."

        return f"""interaction {interaction.unique_id}
        {tag_str}{branch_str}{git_str}{turns_str}{time_str}{title_str}
{first_msg}
"""

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        project = prog.project

        if self.interaction_identifier:
            # Detailed view
            target_id = self.interaction_identifier

            # 1. Try to find by unique_id (should be unique)
            found_by_id = [i for i in project.interactions if i.unique_id == target_id]
            if len(found_by_id) == 1:
                interaction = found_by_id[0]
                result.print(
                    f"--- Interaction Details (ID: {interaction.unique_id}, Tag: {interaction.tag}) ---"
                )
                result.print(interaction.show())
                return result
            elif len(found_by_id) > 1:
                # This should ideally not happen for unique_id, but handle defensively
                result.print(
                    f"CRITICAL ERROR: Multiple interactions found with the same unique ID '{target_id}'. This indicates data corruption."
                )
                return result

            # 2. If not found by unique_id, try to find by tag
            found_by_tag = [i for i in project.interactions if i.tag == target_id]
            if len(found_by_tag) == 1:
                interaction = found_by_tag[0]
                result.print(
                    f"--- Interaction Details (ID: {interaction.unique_id}, Tag: {interaction.tag}) ---"
                )
                result.print(interaction.show())
                return result
            elif len(found_by_tag) > 1:
                result.print(
                    f"Multiple interactions found with tag '{target_id}'. Please use the full unique ID for detailed view:"
                )
                for interaction in found_by_tag:
                    result.print(
                        f"  - ID: {interaction.unique_id}, Tag: {interaction.tag}, Title: '{interaction.title}', Turns: {len(interaction.contents)}"
                    )
                return result
            else:
                result.print(f"No interaction found with tag or ID '{target_id}'.")
                return result
        else:
            # Overview mode
            if not project.interactions:
                result.print("No past interactions found.")
            else:
                # FIXME: actually sort by date
                # note: sorting by date is slightly more complicated, e.g.: what date? date started or date last interacted? for now this will do
                branch_gr = git.get_current_branch(prog.project_root)
                current_branch = branch_gr.value if branch_gr is not None else ""
                num_filtered = 0
                for interaction in reversed(project.interactions):
                    # check branches, unless all is requested
                    if not self.all_branches:
                        if interaction.branch_name != current_branch:
                            num_filtered += 1
                            continue
                        
                    num_turns = len(interaction.contents)
                    result.print(
                        self._overview_show_interaction(interaction, num_turns)
                    )

            if num_filtered > 0:
                result.print(f"{num_filtered} interaction histories have been filtered.")
                if prog.user_config.newbie:
                    result.print(f"Hint: By default, only interactions that were started on your current git branch are shown. Use --all-branches to disable this behaviour, or checkout a different branch.")
        return result


class InteractCommand(BaseModel, CommandInterface):
    """Launches an interactive session with the Coder LLM."""

    interaction_history: Optional[types.InteractionHistory] = Field(default=None)

    # wether we will perform actions
    # disabling this will make the backend do talking instead of generating code parts etc
    actions: bool = True

    initial_prompt: Optional[str] = Field(
        default=None,
        description="An optional initial prompt to start the interactive session. If provided, it bypasses the first user input.",
    )

    skip_to: Optional[types.AIAgent] = Field(
        default=None,
        description="If this is set to either coder or worker, interaction will skip prompt routing and query the specified AI directly.",
    )

    interaction_identifier: Optional[str] = Field(
        default=None,
        description="Unique ID or tag of a past interaction to load and continue.",
    )

    initial_interaction_history_length: int = Field(
        default = 0,
        description = "Number of messages in initial interaction. For new interactions this is always zero. It may be nonzero if an existing interaction is loaded. Used primarily to check wether there was any real change at all and if the current interaction needs to be saved."
    )

    # whether to force releasing of interaction lock
    force_lock: bool = False
    
    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        # since interact queries the backend, we verify the keys
        verify_result = VerifyCommand().run(prog)
        if verify_result.data["error"]:
            result.print(verify_result.text)
            result.print("Aborting interaction.")
            return result

        actions_str = (
            " Talk only, no actions will be performed in this interaction."
            if not (self.actions)
            else ""
        )
        result.print("Starting interactive session with 👻." + actions_str)
        prog.print("API keys checked. All good.")
        # Load existing interaction history if an identifier is provided
        if self.interaction_identifier:
            if prog.project is None:
                logger.error("Project is null, cannot load interaction history.")
                sys.exit(1)

            loaded_history = prog.project.get_interaction_history(
                unique_id=self.interaction_identifier,
                tag=self.interaction_identifier
            )

            if loaded_history is None:
                prog.print(f"Error: No interaction found with ID or tag '{self.interaction_identifier}'.")
                sys.exit(1)
            else:
                self.interaction_history = loaded_history
                self.initial_interaction_history_length = len(self.interaction_history.contents)
                # need to add the preamble injection
                ghostbox_history = self.interaction_history.to_chat_messages()
                if ghostbox_history:
                    content = ghostbox_history[0].content                    
                    if isinstance(content, str):
                        ghostbox_history[0].content = self._prefix_preamble_string(content)
                    else:
                        # technically the content can be a list or a dict, but we never use this
                        logger.warning(f"Non-string content field in first ghostbox history message. This means we didn't add the preamble injection string.")
                prog.coder_box.set_history(ghostbox_history)
                prog.print(f"Continuing interaction '{self.interaction_history.title}' (ID: {self.interaction_history.unique_id}).")

        # start the actual loop in another method
        return self._interact_loop(result, prog)

        # This code is unreachable but in the future error handling/control flow of this method might become more complicated and we may need it
        return result
    
    def _make_preamble(self, prog: Program) -> str:
        """Plaintext context that is inserted before the user prompt - though only once."""
        return prompts.make_prompt(
            prog,
            prompt_config=prompts.PromptConfig.minimal(
                project_metadata=True,
                style_file = True,
                context_files="full",
                # could add shell here?
            ),
        )

    
    def _prefix_preamble_string(self, prompt: str) -> str:
        """
        Prefixes the user's prompt with the magic `{_{_preamble_injection_}_}` placeholder (without the outer underscores).
        This method *only* adds the placeholder string. The actual content for the
        preamble is dynamically set and updated via `prog.coder_box.set_vars()`
        before each LLM call. This ensures the preamble is always current without
        being hardcoded into the interaction history or prompt template.
        """
        # note the song-and-dance with string concatenation below is so that we can use ghostcode on itself without an unwanted expansion of the preamble magic string
        return "{{" + "preamble_injection" + "}}" + f"# User Prompt\n\n{prompt}"
    
    def _make_user_prompt(self, prog: Program, prompt: str) -> str:
        """Prepare a user prompt to be sent to the backend.

        This method intelligently constructs the prompt to be sent to the LLM backend.
        For the *first* message in a new interaction, it includes the `{_{_preamble_injection_}_}` (without the outer underscores)
        placeholder by calling `_prefix_preamble_string`. For subsequent messages in an
        ongoing interaction, it sends only the raw user `prompt`.

        The actual content of the preamble (project metadata, context files, etc.) is
        generated by `_make_preamble` and then dynamically injected into the LLM context
        by `prog.coder_box.set_vars()` before each LLM request. This ensures that the
        LLM always receives the most up-to-date project context without incurring
        repeated token costs for the full preamble on every turn.
        """
        if self.interaction_history is None:
            logger.warning(
                f"Tried to construct preamble with null interaction history in interaction."
            )
            return ""

        # Only inject the preamble placeholder for the very first message of an interaction.
        # Subsequent messages rely on `prog.coder_box.set_vars` to keep the preamble updated.
        if self.interaction_history.empty():
            return self._prefix_preamble_string(prompt)
        return prompt    
        
    def _dump_interaction(self, prog: Program) -> None:
        """Dumps interaction history to .ghostcode/current_interaction.txt and .ghostcode/current_interaction.json"""
        if (
            prog.project is not None
            and prog.project_root is not None
            and self.interaction_history is not None
        ):
            with ProgressPrinter(
                message=" Saving interaction ", print_function=prog.print
            ):
                current_interaction_history_filepath = os.path.join(
                    prog.project_root,
                    ".ghostcode",
                    prog.project._CURRENT_INTERACTION_HISTORY_FILE,
                )
                current_interaction_history_plaintext_filepath = os.path.join(
                    prog.project_root,
                    ".ghostcode",
                    prog.project._CURRENT_INTERACTION_PLAINTEXT_FILE,
                )
                logger.info(
                    f"Dumping current interaction history to {current_interaction_history_filepath} and {current_interaction_history_plaintext_filepath}"
                )
                with open(current_interaction_history_filepath, "w") as hf:
                    json.dump(self.interaction_history.model_dump(), hf, indent=4)

                with open(current_interaction_history_plaintext_filepath, "w") as pf:
                    pf.write(self.interaction_history.show())
        else:
            logger.warning(
                f"Null project_root while trying to dump current itneraction history. Create a project root or no dump for you!"
            )

    def _save_interaction(self, prog: Program) -> None:
        if self.interaction_history is None:
            logger.warning(
                f"Tried to save null interaction history during interaction."
            )
            return

        if self.interaction_history.empty():
            # nothing to do
            return

        if len(self.interaction_history.contents) == self.initial_interaction_history_length:
            # history may have been loaded and wasn't change -> do nothing
            return
        
        logger.info(f"Finishing interaction.")
        new_title = worker.worker_generate_title(prog, self.interaction_history)
        self.interaction_history.title = (
            new_title if new_title else self.interaction_history.title
        )

    def _make_llm_response_profile(self) -> types.LLMResponseProfile:
        if not (self.actions):
            return types.LLMResponseProfile.text_only()

        # default is return whatever is the default
        return types.LLMResponseProfile()

    def _make_initial_action(self, **kwargs: Any) -> types.Action:
        """Create the initial action to place on the action queue.
        Arguments are passed directly through to query constructors, like ActionQueryCoder, ActionQueryWorker, or ActionRouteRequest.
        """
        if self.skip_to == types.AIAgent.CODER:
            return types.ActionQueryCoder(**kwargs)

        if self.skip_to == types.AIAgent.WORKER:
            return types.ActionQueryWorker(**kwargs)

        # routing
        return types.ActionRouteRequest(**kwargs)

    def _process_user_input(self, prog: types.Program, user_input: str) -> None:
        """Helper method to encapsulate the logic for sending user input to the LLM."""
        if prog.project is None:
            raise RuntimeError(
                f"Project seems to be null during interaction. This shouldn't happen, but just in case, you may want to do `ghostcode init` in your project's directory."
            )

        preamble = self._make_preamble(prog)
        prompt_to_send = self._make_user_prompt(prog, user_input)
        prog.coder_box.set_vars({"preamble_injection": preamble})

        if self.interaction_history is not None:
            self.interaction_history.contents.append(
                types.UserInteractionHistoryItem(
                    preamble=preamble,
                    prompt=user_input,
                    context=prog.project.context_files,
                )
            )
        else:
            logger.warning(f"No interaction history. Interaction discarded.")

        logger.info(f"Preparing action queue.")
        prog.discard_actions()
        prog.queue_action(
            self._make_initial_action(
                prompt=prompt_to_send,
                interaction_history_id=(
                    self.interaction_history.unique_id
                    if self.interaction_history is not None
                    else "000-000-000-000"
                ),
                hidden=False,
                llm_response_profile=self._make_llm_response_profile(),
            )
        )
        worker.run_action_queue(prog)

    def _interact_loop(
        self, intermediate_result: CommandOutput, prog: Program
    ) -> CommandOutput:
        if prog.project is None:
            logger.error("Encountered null project in interact loop. Aborting.")
            intermediate_result.print(
                "Failed to initialize project. Please do\n\n```\nghostcode init\n```\n\nto create a project in the current working directory, then retry interact."
            )
            return intermediate_result

        prog.print(intermediate_result.text)
        if self.interaction_history is None:
            self.interaction_history = prog.project.new_interaction_history()

        # lock guard
        if (lock_id := prog.lock_read()) is not None:
            if self.force_lock:
                logger.warning(f"Failed to acquire lock because of interaction {lock_id}, but lock will be forced.")
                prog.lock_release()
            else:
                logger.error(f"Failed to acquire interaction lock due to ongoing interaction {lock_id} .")
                intermediate_result.print(f"Failed to acquire lock. Aborting.\nAnother ghostcode session (interaction {lock_id}) is currently in progress. Please finish that interaction, or\nrestart ghostcode with `ghostcode interaction --force` to force it closed. This may lead to data loss. You have been warned.")
                return intermediate_result
            
        # Initial prompt handling
        if self.initial_prompt is not None:
            current_user_input = self.initial_prompt
            self.initial_prompt = None  # Consume the initial prompt
        else:
            current_user_input = ""

        if current_user_input:
            # If an initial prompt was provided, process it immediately
            self._process_user_input(prog, current_user_input)
            # After processing, the loop will continue to ask for more input
            current_user_input = ""  # Clear for subsequent inputs

        # Main interactive loop
        prog.print(
            "Multiline mode enabled. Type your prompt over multiple lines.\nType a single '\\' and hit enter to submit.\nType /quit or CTRL+D to quit."
        )

        while True:
            try:
                if current_user_input == "":
                    # don't print this if user is building multi-line input
                    prog.print(prog._get_cli_prompt(), end="")
                line = input()
            except EOFError:
                break  # User pressed CTRL+D, exit interaction

            # Handle slash commands
            match slash_commands.try_command(prog, self.interaction_history, line):
                case slash_commands.SlashCommandResult.OK:
                    continue  # Command handled, go to next loop iteration (ask for input)
                case slash_commands.SlashCommandResult.HALT:
                    break  # Command halted, exit interaction
                case slash_commands.SlashCommandResult.COMMAND_NOT_FOUND:
                    prog.print(f"Unrecognized command: {line}")
                    continue
                case slash_commands.SlashCommandResult.BAD_ARGUMENTS:
                    prog.print(
                        f"Invalid arguments. Try /help COMMAND for more information."
                    )
                    continue
                case slash_commands.SlashCommandResult.ACTIONS_OFF:
                    if self.actions:
                        self.actions = False
                        prog.print("Enabled talk mode. Coder backend will generate text only, no file edits.")
                    else:
                        prog.print("Talk mode already enabled, use /interact to switch to interactive mode.")
                case slash_commands.SlashCommandResult.ACTIONS_ON:
                    if not(self.actions):
                        self.actions = True
                        prog.print("Interact mode enabled. Coder backend will generate code and produce file edits.")
                    else:
                        prog.print("Interact mode already enabled. Use /talk to disable code generation and file edits.")
                case _:
                    pass  # Not a slash command, accumulate input

            # Accumulate user input
            if line != "\\":
                current_user_input += "\n" + line

                if prog.user_config.newbie and current_user_input.endswith("\n\n"):
                    # user may be frantically trying to submit
                    prog.print("(Hint: Enter a single backslash (\\) and hit enter to submit your prompt. Disable this message with `ghostcode config set newbie False`)")

                continue  # Keep accumulating

            # If we reach here, it means user typed '\\' to submit
            if not current_user_input.strip():
                prog.print(
                    "Empty prompt. Please provide some input or a slash command."
                )
                continue  # Ask for input again

            try:
                with prog.interaction_lock(
                        interaction_history_id=self.interaction_history.unique_id,
                        should_lock=self.actions # Only lock if actions are enabled
                ):            
                    self._process_user_input(prog, current_user_input)
                    current_user_input = ""  # Clear buffer for next turn
                    self._dump_interaction(prog)  # Save state after each turn
            except types.InteractionLockError as e:
                logger.error(f"Failed to acquire lock: {e}")                
                prog.print(f"Cannot proceed because another ghostcode session is in progress (interaction {prog.lock_read()}).\nPlease finish the ongoing interaction, or force it to close by running ghostcode \nwith `ghostcode interaction --force` or doing /force right here in the terminal. Data may be lost. You have been warned.")

        # End of interaction
        prog.debug_dump()                
        self._save_interaction(prog)

        return CommandOutput(text="Finished interaction.")


class VerifyCommand(BaseModel):
    """Command to verify program integrity and configuration.
    This is currently only used internally to check for API keys."""

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        result.data = {"error": False}
        if prog.project is None:
            logger.warning(
                f"Encountered null project in program during verify command."
            )
            result.print(
                f"Failed verification due to missing project. Please do `ghostcode init` in a directory of your choice to initialize a project, then retry the verification."
            )
            result.data["error"] = True
            return result

        missing_api_keys = prog._has_api_keys()
        if missing_api_keys:
            result.data["error"] = True
            user_config_path = os.path.join(
                appdirs.user_config_dir("ghostcode"),
                types.UserConfig._GHOSTCODE_CONFIG_FILE,
            )
            result.print("\n--- WARNING: Missing API Keys for Cloud LLMs ---")
            result.print(
                f"To use the configured cloud LLMs, please provide the necessary API keys in your user configuration file:"
            )
            result.print(f"  {user_config_path}")
            result.print(
                "You can edit this file directly or use 'ghostcode config set' for user-specific API keys."
            )
            result.print("--------------------------------------------------")
            for backend, _ in missing_api_keys.items():
                if backend == LLMBackend.google:
                    result.print(
                        f"  - Google AI Studio API key is missing for backend '{prog.project.config.coder_backend}' or '{prog.project.config.worker_backend}'."
                    )
                    result.print(
                        "    Get your key at: https://aistudio.google.com/app/apikey"
                    )
                elif backend == LLMBackend.openai:
                    result.print(
                        f"  - OpenAI API key is missing for backend '{prog.project.config.coder_backend}' or '{prog.project.config.worker_backend}'."
                    )
                    result.print(
                        "    Get your key at: https://platform.openai.com/account/api-keys"
                    )
            result.print("--------------------------------------------------\n")
        return result


def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="ghostcode",
        description="A command line tool to help programmers code using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help formatting
    )

    # Add --user-config argument
    parser.add_argument(
        "-c",
        "--user-config",
        type=str,
        default=None,
        help=f"Path to a custom user configuration file (YAML format). "
        f"Defaults to {os.path.join(appdirs.user_config_dir('ghostcode'), types.UserConfig._GHOSTCODE_CONFIG_FILE)}.",
    )

    # -- coder-backend argument
    parser.add_argument(
        "-C", "--coder-backend",
        type=str,
        choices=[backend.name for _, backend in enumerate(ghostbox.LLMBackend)],
        default="",
        help="Set the choice of backend for the coder LLM. This will override both project and user configuration options for the coder backend.",
    )    

    parser.add_argument(
        "-W", "--worker-backend",
        type=str,
        choices=[backend.name for _, backend in enumerate(ghostbox.LLMBackend)],
        default="",
        help="Set the choice of backend for the worker LLM. This will override both project and user configuration options for the coder backend.",
    )    
    
    # Add --logging argument
    parser.add_argument(
        "--logging",
        type=str,
        choices=["off", "stderr", "file"],
        default="off",
        help="Configure secondary logging output: 'off' (no additional output), 'stderr' (output to stderr in addition to file), or 'file' (output to a secondary file specified by --secondary-log-filepath).",
    )
    parser.add_argument(
        "--secondary-log-filepath",
        type=str,
        default=None,
        help="Path for secondary file logging when --logging is set to 'file'.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging, showing verbose internal messages."
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new ghostcode project in the current directory or a specified path.",
    )
    init_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path to initialize the project. Defaults to current directory.",
    )
    init_parser.set_defaults(func=lambda args: InitCommand(path=args.path))

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Manage ghostcode project configuration."
    )
    config_subparsers = config_parser.add_subparsers(
        dest="subcommand", required=True, help="Config subcommands"
    )

    config_ls_parser = config_subparsers.add_parser(
        "ls", help="List all configuration options."
    )
    config_ls_parser.set_defaults(func=lambda args: ConfigCommand(subcommand="ls"))

    config_get_parser = config_subparsers.add_parser(
        "get", help="Get the value of a specific configuration option."
    )
    config_get_parser.add_argument(
        "key",
        help="The configuration key to retrieve (e.g., 'coder_llm_config.model', 'config.coder_backend').",
    )
    config_get_parser.set_defaults(
        func=lambda args: ConfigCommand(subcommand="get", key=args.key)
    )

    config_set_parser = config_subparsers.add_parser(
        "set", help="Set the value of a configuration option."
    )
    config_set_parser.add_argument(
        "key",
        help="The configuration key to set (e.g., 'coder_llm_config.model', 'config.coder_backend').",
    )
    config_set_parser.add_argument(
        "value",
        help="The new value for the configuration key. Use JSON format for complex types (e.g., 'true', '123', '\"string\"', '[1,2]', '{\"key\":\"value\"}'). For backend types, use the string name (e.g., 'google', 'generic').",
    )
    config_set_parser.set_defaults(
        func=lambda args: ConfigCommand(
            subcommand="set", key=args.key, value=args.value
        )
    )

    # Context command
    context_parser = subparsers.add_parser(
        "context", help="Manage files included in the project context."
    )
    context_subparsers = context_parser.add_subparsers(
        dest="subcommand", required=True, help="Context subcommands"
    )

    context_ls_parser = context_subparsers.add_parser(
        "ls", aliases=["list"], help="List all files in the project context."
    )
    context_ls_parser.set_defaults(func=lambda args: ContextCommand(subcommand="ls"))
    
    context_clean_parser = context_subparsers.add_parser(
        "clean", aliases=[], help="Remove bogus or non-existing files from context."
    )
    context_clean_parser.set_defaults(func=lambda args: ContextCommand(subcommand="clean"))

    context_wipe_parser = context_subparsers.add_parser(
        "wipe", help="Remove all files from the project context."
    )
    context_wipe_parser.set_defaults(func=lambda args: ContextCommand(subcommand="wipe"))

    context_lock_parser = context_subparsers.add_parser(
        "lock", help="Lock file(s) in the project context, preventing their removal."
    )
    context_lock_parser.add_argument(
        "filepaths",
        nargs="+",
        help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
    )
    context_lock_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="lock", filepaths=args.filepaths)
    )

    context_unlock_parser = context_subparsers.add_parser(
        "unlock", help="Unlock file(s) in the project context, allowing their removal."
    )
    context_unlock_parser.add_argument(
        "filepaths",
        nargs="+",
        help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
    )
    context_unlock_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="unlock", filepaths=args.filepaths)
    )

    context_add_parser = context_subparsers.add_parser(
        "add", help="Add file(s) to the project context. Supports wildcards."
    )
    context_add_parser.add_argument(
        "filepaths",
        nargs="+",
        help="One or more file paths (can include wildcards like '*.py', 'src/**.js').",
    )
    context_add_parser.add_argument(
        "--rag",
        action="store_true",
        help="Enable Retrieval Augmented Generation for these files.",
    )
    context_add_parser.set_defaults(
        func=lambda args: ContextCommand(
            subcommand="add", filepaths=args.filepaths, rag=args.rag
        )
    )

    context_rm_parser = context_subparsers.add_parser(
        "rm",
        aliases=["remove"],
        help="Remove file(s) from the project context. Supports wildcards.",
    )
    context_rm_parser.add_argument(
        "filepaths", nargs="+", help="One or more file paths (can include wildcards)."
    )
    context_rm_parser.set_defaults(
        func=lambda args: ContextCommand(subcommand="rm", filepaths=args.filepaths)
    )


    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Intelligently discovers files associated with the project and adds them to the context.",
    )
    discover_parser.add_argument(
        "filepath",
        help="The filepath that points to a directory in which files to add to the context will be recursively discovered.",
    )

    # Dynamically add language flags
    for lang in sorted(DiscoverCommand.possible_languages):
        discover_parser.add_argument(
            f"--{lang}",
            action="append_const", # Use append_const to collect multiple languages
            const=lang,
            dest="languages_enabled", # All --lang flags append to this list
            help=f"Include files primarily written in {lang.capitalize()}.",
        )
        discover_parser.add_argument(
            f"--no-{lang}",
            action="append_const",
            const=lang,
            dest="languages_disabled", # All --no-lang flags append to this list
            help=f"Exclude files primarily written in {lang.capitalize()}.",
        )

    discover_parser.add_argument(
        "--exclude-pattern",
        type=str,
        default="",
        help="A regex that can be provided. File names that match against it are excluded from the context. The exclude pattern is applied at the very end of the discovery process.",
    )

    discover_parser.add_argument(
        "--all",
        action="store_true",
        help="If provided, will add all (non-hidden) files to the context that are found in subdirectories of the given path. Providing --all is like providing all of the language parameters.",
    )

    discover_parser.add_argument(
        "--min-lines",
        type=int,
        default=None,
        help="Minimum amount of lines that a file must have in order to be added to the context.",
    )
    discover_parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of lines a file can have in order to be added to the context.",
    )

    size_heuristic_group = discover_parser.add_mutually_exclusive_group()
    size_heuristic_group.add_argument(
        "--small",
        action="store_const",
        const="small",
        dest="size_heuristic",
        help="If provided, sets min_lines=0 and max_lines=1000. Mutually exclusive with --medium and --large.",
    )
    size_heuristic_group.add_argument(
        "--medium",
        action="store_const",
        const="medium",
        dest="size_heuristic",
        help="If provided, sets min_lines=300 and max_lines=3000. Mutually exclusive with --small and --large.",
    )
    size_heuristic_group.add_argument(
        "--large",
        action="store_const",
        const="large",
        dest="size_heuristic",
        help="If provided, sets min_lines=3000 and no max_lines. Mutually exclusive with --small and --medium.",
    )

    discover_parser.set_defaults(
        func=lambda args: DiscoverCommand(
            filepath=args.filepath,
            languages_enabled=args.languages_enabled if args.languages_enabled else [],
            languages_disabled=args.languages_disabled if args.languages_disabled else [],
            min_lines=args.min_lines,
            max_lines=args.max_lines,
            size_heuristic=args.size_heuristic,
            exclude_pattern=args.exclude_pattern,
            all=args.all,
        )
    )
    
    # Interact command
    interact_parser = subparsers.add_parser(
        "interact", help="Launches an interactive session with the Coder LLM."
    )
    interact_parser.add_argument(
        "--actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --no-actions is provided, ghostcode won't try to do edits, file creation, git commits or anything else that changes state. The ghostcoder LLM will be forced to generate only text, which may or may not contain inline code examples.",
    )
    interact_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt for the interactive session. If provided, it bypasses the first user input.",
    )
    interact_parser.add_argument(
        "--skip-to-coder",
        action="store_true",
        help="Skip prompt routing and directly query the Coder LLM.",
    )
    interact_parser.add_argument(
        "--skip-to-worker",
        action="store_true",
        help="Skip prompt routing and directly query the Worker LLM.",
    )
    interact_parser.add_argument(
        "-i",
        "--interaction",
        type=str,
        default=None,
        help="Optional unique ID or tag of a past interaction to continue.",
    )
    interact_parser.add_argument(
        "--force",
        action="store_true",
        help="Force release of interaction lock if one exists.",
    )
    interact_parser.set_defaults(
        func=lambda args: _create_interact_command(args, actions=args.actions, force_lock=args.force)
    )

    # Talk command
    talk_parser = subparsers.add_parser(
        "talk",
        help="Launches an interactive session with the Coder LLM, but without performing any actions. This is shorthand for ghostcode interact --no-actions",
    )
    talk_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt for the interactive session. If provided, it bypasses the first user input.",
    )
    talk_parser.add_argument(
        "--skip-to-coder",
        action="store_true",
        help="Skip prompt routing and directly query the Coder LLM.",
    )
    talk_parser.add_argument(
        "--skip-to-worker",
        action="store_true",
        help="Skip prompt routing and directly query the Worker LLM.",
    )
    talk_parser.add_argument(
        "-i",
        "--interaction",
        type=str,
        default=None,
        help="Optional unique ID or tag of a past interaction to continue.",
    )
    talk_parser.add_argument(
        "--force",
        action="store_true",
        help="Force release of interaction lock if one exists.",
    )
    talk_parser.set_defaults(
        func=lambda args: _create_interact_command(args, actions=False, force_lock=args.force)
    )
    
    # Log command
    log_parser = subparsers.add_parser("log", help="Display past interaction history.")
    log_parser.add_argument(
        "--interaction",
        type=str,
        help="Display a specific interaction in detail by its unique ID or tag.",
    )
    log_parser.add_argument(
        "--all-branches",
        action="store_true",
        help="Do not filter any interactions based on branches."
    )
    log_parser.set_defaults(
        func=lambda args: LogCommand(interaction_identifier=args.interaction, all_branches=args.all_branches)
    )

    args = parser.parse_args()    
    # --- Determine project_root early for logging configuration ---

    current_project_root: Optional[str] = None
    if args.command == "init":
        # For init, the project_root is the path being initialized
        current_project_root = os.path.abspath(args.path if args.path else os.getcwd())
    else:
        # For other commands, find the existing project root
        current_project_root = types.Project.find_project_root()

    # Configure logging based on argument and determined project_root
    _configure_logging(
        args.logging,
        current_project_root,
        is_init=args.command == "init",
        secondary_log_filepath=args.secondary_log_filepath,
        is_debug=args.debug,
    )

    # Now that logging is configured, get the main logger instance
    global logger  # Declare logger as global to assign to it
    logger = logging.getLogger("ghostcode.main")

    # Load UserConfig
    user_config_path = args.user_config
    user_config: types.UserConfig
    try:
        user_config = types.UserConfig.load(user_config_path)
    except FileNotFoundError:
        logger.info(f"User configuration file not found. Creating a default one.")
        user_config = types.UserConfig()
        user_config.save(user_config_path)  # Save to the default or specified path
    except Exception as e:
        logger.error(
            f"Failed to load user configuration: {e}. Using default settings.",
            exc_info=True,
        )
        user_config = types.UserConfig()

    # Instantiate the command object
    command_obj: CommandInterface = args.func(args)

    # Special handling for 'init' command as it doesn't require an existing project
    if isinstance(command_obj, InitCommand):
        # The project_root for init is already determined as current_project_root
        # and logging is configured. Just run the command.
        out = command_obj.run(
            Program(
                project_root=current_project_root,
                project=None,
                worker_box=None,
                coder_box=None,
                user_config=user_config,
            )
        )  # Pass user_config
        print(out.text)
        sys.exit(0)  # Exit after init

    # For all other commands, a project must exist
    if not current_project_root:  # This would have been set by find_project_root
        logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
        sys.exit(1)

    try:
        project = types.Project.from_root(current_project_root)
    except (
        FileNotFoundError
    ):  # Should ideally be caught by find_project_root, but good to be safe
        logger.error(
            f"Project directory '.ghostcode' not found at {current_project_root}. This should not happen after root detection."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"Failed to load ghostcode project from {current_project_root}: {e}",
            exc_info=True,
        )
        sys.exit(1)

    # Initialize Ghostbox instances using project.config for endpoints and backends
    # the quiet options have to be passed here or we will get a couple of stray messages until the project box configs are read
    quiet_options = {"quiet": True, "stdout": False, "stderr": False}

    # Now directly use the string values from project.config
    # Pass API keys from user_config to Ghostbox instances
    # we catch missing API key and other backend initialization errors here, and inform the user later in commands that actually require backends
    try:
        worker_backend = project.config.worker_backend if args.worker_backend == "" else args.worker_backend
        worker_box = Ghostbox(
            endpoint=project.config.worker_endpoint,
            backend= worker_backend,
            character_folder=os.path.join(
                current_project_root, ".ghostcode", project._WORKER_CHARACTER_FOLDER
            ),
            api_key=user_config.api_key,  # General API key
            google_api_key=user_config.google_api_key,
            openai_api_key=user_config.openai_api_key,
            deepseek_api_key=user_config.deepseek_api_key,
            model = user_config.get_model(types.AIAgent.CODER, worker_backend),            
            **quiet_options,
        )
    except Exception as e:
        logger.error(f"Setting worker to dummy. Reason: {e}")
        worker_box = ghostbox.from_dummy(**quiet_options)

    try:
        coder_backend = project.config.coder_backend if args.coder_backend == "" else args.coder_backend
        coder_box = Ghostbox(
            endpoint=project.config.coder_endpoint,
            backend=coder_backend,
            character_folder=os.path.join(
                current_project_root, ".ghostcode", project._CODER_CHARACTER_FOLDER
            ),
            api_key=user_config.api_key,  # General API key
            google_api_key=user_config.google_api_key,
            openai_api_key=user_config.openai_api_key,
            deepseek_api_key=user_config.deepseek_api_key,
            model = user_config.get_model(types.AIAgent.CODER, coder_backend),
            **quiet_options,
        )
    except Exception as e:
        logger.error(f"Setting coder backend to dummy. Reason: {e}")
        coder_box = ghostbox.from_dummy(**quiet_options)

    # Create the Program instance
    prog_instance = Program(
        project_root=current_project_root,
        project=project,
        worker_box=worker_box,
        coder_box=coder_box,
        user_config=user_config,  # Pass the loaded user_config
    )

    # Run the command
    out = command_obj.run(prog_instance)
    print(out.text)
    if prog_instance.project_root is not None:
        project.save_to_root(prog_instance.project_root)


def _create_interact_command(
    args: argparse.Namespace, actions: bool, force_lock: bool = False
) -> InteractCommand:
    """Helper function to create InteractCommand, handling skip_to logic and mutual exclusivity."""
    skip_to_agent: Optional[types.AIAgent] = None

    if args.skip_to_coder and args.skip_to_worker:
        msg = "Cannot use both --skip-to-coder and --skip-to-worker simultaneously."
        logger.error(msg)
        print(msg)
        sys.exit(1)
    elif args.skip_to_coder:
        skip_to_agent = types.AIAgent.CODER
    elif args.skip_to_worker:
        skip_to_agent = types.AIAgent.WORKER

    return InteractCommand(
        actions=actions,
        initial_prompt=args.prompt,
        skip_to=skip_to_agent,
        interaction_identifier=args.interaction,
        force_lock=force_lock,
    )


if __name__ == "__main__":
    _main()

