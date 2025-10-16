from typing import *
import json
import traceback
from ghostcode import types, worker
from ghostcode.progress_printer import ProgressPrinter

# don't want the types. prefix for these
from ghostcode.types import Program
import ghostbox
import os
from ghostbox import Ghostbox  # type: ignore
from ghostbox.definitions import BrokenBackend  # type: ignore
from ghostbox.commands import showTime  # type: ignore
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

# logger will be configured after argument parsing
logger: logging.Logger  # Declare logger globally, will be assigned later


class ExceptionListHandler(logging.Handler):
    """Stores exceptions in a global list."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.exceptions = Queue()

    def emit(self, record):
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
EXCEPTION_HANDLER = ExceptionListHandler()


def _configure_logging(
    log_mode: str,
    project_root: Optional[str],
    is_init: bool = False,
    secondary_log_filepath: Optional[str] = None,
):
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
        def filter(self, record):
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
    def timing_log(self, message, *args, **kwargs):
        if self.isEnabledFor(TIMING_LEVEL_NUM):
            self._log(TIMING_LEVEL_NUM, message, args, **kwargs)

    logging.Logger.timing = timing_log  # type: ignore

    # Set the root logger level to INFO by default
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

    subcommand: Literal["add", "rm", "remove", "ls"]
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
                for cf in project.context_files.data:
                    rag_status = "(RAG enabled)" if cf.rag else ""
                    result.print(f"  - {cf.filepath} {rag_status}")
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
                        project.context_files.data.append(
                            types.ContextFile(filepath=fp, rag=self.rag)
                        )
                        result.print(f"Added '{fp}' to context (RAG={self.rag}).")
                    else:
                        # If already exists, update RAG status if different
                        existing_cf = current_context_files[fp]
                        if existing_cf.rag != self.rag:
                            existing_cf.rag = self.rag
                            result.print(
                                f"Updated RAG status for '{fp}' to {self.rag}."
                            )
                        else:
                            result.print(
                                f"'{fp}' is already in context with RAG={self.rag}."
                            )
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
        return result


class LogCommand(BaseModel, CommandInterface):
    """Manages and displays interaction history."""

    interaction_identifier: Optional[str] = Field(
        default=None,
        description="Optional unique ID or tag of a specific interaction to display in detail.",
    )

    def _overview_show_interaction(
        self, interaction: types.InteractionHistory, num_turns: int
    ) -> str:
        turns_str = f"turns: {num_turns}\n"
        tag_str = f"tag: {interaction.tag}\n" if interaction.tag else ""
        git_str = (
            f"git history: {", ".join(commits)}\n"
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
        {tag_str}{git_str}{turns_str}{time_str}{title_str}
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
                for interaction in reversed(project.interactions):
                    num_turns = len(interaction.contents)
                    result.print(
                        self._overview_show_interaction(interaction, num_turns)
                    )
        return result


class InteractCommand(BaseModel, CommandInterface):
    """Launches an interactive session with the Coder LLM."""

    interaction_history: types.InteractionHistory = Field(
        default_factory=types.InteractionHistory
    )

    # wether we will perform actions
    # disabling this will make the backend do talking instead of generating code parts etc
    actions: bool = True

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


        actions_str = " Talk only, no actions will be performed in this interaction." if not(self.actions) else ""
        result.print("Starting interactive session with 👻." + actions_str)
        result.print("API keys checked. All good.")

        # start the actual loop in another method
        return self._interact_loop(result, prog)

        # This code is unreachable but in the future error handling/control flow of this method might become more complicated and we may need it
        return result

    def _make_preamble(self, prog: Program) -> str:
        """Plaintext context that is inserted before the user prompt - though only once."""
        if prog.project is None:
            logger.warning(
                f"Encountered null project in program while creating preamble."
            )
            return ""
        return f"""{prog.project.context_files.show()}

# User Prompt

"""

    def _make_user_prompt(self, prog: Program, prompt: str) -> str:
        """Prepare a user prompt to be sent to the backend."""
        # atm we only add a var hook for injections with ghostbox
        # i.e. watch out the braces aren't for python format strings

        # also what complicates this is that we only want the preamble to be in the context once, so as to not explode our token limit
        # this is why we check the history.
        # ghostbox will then use the var injection to keep the preamble at the start of the context up-to-date
        if self.interaction_history.empty():
            return "{{project_metadata}\n\n}{{preamble_injection}}" + prompt
        return prompt

    def _query_coder(self, prog: Program, prompt: str) -> Optional[types.CoderResponse]:
        try:
            with ProgressPrinter(message=" Querying 👻 ", print_function=prog.print):
                if self.actions:
                    # this is the default, offer the coder the full menu of parts to generate
                    response = prog.coder_box.new(
                        types.CoderResponse,
                        prompt,
                    )
                elif not (self.actions):
                    # we only want a text response
                    text_response = prog.coder_box.new(types.TextResponsePart, prompt)
                    response = types.CoderResponse(contents=[text_response])
            logger.timing(f"ghostcoder performance statistics:\n{showTime(prog.coder_box._plumbing, [])}")  # type: ignore
        except Exception as e:
            prog.print(
                f"Problems encountered during 👻 request. Reason: {e}\nRetry the request or consult the logs for more information."
            )
            logger.error(
                f"error on ghostbox.new. See the full traceback:\n{traceback.format_exc()}"
            )
            logger.error(
                f"\nAnd here is the last result:\n\n{prog.coder_box.get_last_result()}"
            )
            return None
        return response

    def _interact_loop(
        self, intermediate_result: CommandOutput, prog: Program
    ) -> CommandOutput:
        if prog.project is None:
            logger.error("Encountered null project in interact loop. Aborting.")
            intermediate_result.print(
                "Failed to initialize project. Please do\n\n```\nghostcode init\n```\n\nto create a project in the current working directory, then retry interact."
            )
            return intermediate_result

        # since the interaction is blocking we can't wait with printing until we return the result
        # so we print it here and return an empty result at the end
        prog.print(intermediate_result.text)
        interacting = True
        print(
            "Multiline mode enabled. Type your prompt over multiple lines.\nType a single '\\' and hit enter to submit.\nType /quit or CTRL+D to quit."
        )
        user_input = ""
        while interacting:
            try:
                # for complex reasons we do printing of cli prompt ourselves instead of leaving it to input()
                prog.print(prog._get_cli_prompt(), end="")
                line = input()
            except EOFError:
                break

            if line == "/quit":
                break
            elif line == "/lastrequest":
                print(json.dumps(prog.coder_box.get_last_request(), indent=4))
                continue
            elif line == "/traceback":
                global EXCEPTION_HANDLER
                if (tr_str := EXCEPTION_HANDLER.try_get_last_traceback()) is not None:
                    print(
                        f"Displaying most recent exception below. Repeated calls of /traceback will display earlier exceptions.\n\n```\n{tr_str}```\n"
                    )
                else:
                    print("No recent tracebacks.")
                continue
            elif line == "/show":
                print(self.interaction_history.show())
                continue
            elif line == "/save":
                with open("out.txt", "w") as f:
                    f.write(self.interaction_history.show())
                continue
            elif line != "\\":
                user_input += "\n" + line
                continue

            # ok, process input

            preamble = prog.project.context_files.show()
            if prog.project.project_metadata is None:
                logger.warning("Empty metadata while setting ghostbox vars.")
                metadata_str = ""
            else:
                metadata_str = prog.project.project_metadata.show()

            prompt = self._make_user_prompt(prog, user_input)
            prog.coder_box.set_vars(
                {"project_metadata": metadata_str, "preamble_injection": preamble}
            )

            self.interaction_history.contents.append(
                types.UserInteractionHistoryItem(
                    preamble=preamble,
                    prompt=user_input,
                    context=prog.project.context_files,
                )
            )

            response = self._query_coder(prog, prompt)
            if not (response):
                continue

            try:
                self.interaction_history.contents.append(
                    types.CoderInteractionHistoryItem(
                        context=prog.project.context_files,
                        backend=prog.project.config.coder_backend,
                        model=prog.project.coder_llm_config.get("model", "N/A"),
                        response=response,
                    )
                )
            except Exception as e:
                # this is not critical, log and continue
                logger.error(f"Could not save response to history. Reason: {e}")

            # handle the response
            prog.print(response.show_cli())
            logger.info(f"Preparing action queue.")
            prog.action_queue = worker.actions_from_response_parts(
                prog, response.contents
            )
            worker.run_action_queue(prog)

            user_input = ""
            # this is a failsafe and for user convenience in case we crash
            if prog.project_root is not None:
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

        # end of interaction
        logger.info(f"Finishing interaction.")
        if not (self.interaction_history.empty()):
            prog.project.interactions.append(self.interaction_history)

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


def main():
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

    # Interact command (placeholder)
    interact_parser = subparsers.add_parser(
        "interact", help="Launches an interactive session with the Coder LLM."
    )
    interact_parser.add_argument(
        "--actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --no-actions is provided, ghostcode won't try to do edits, file creation, git commits or anything else that changes state. The ghostcoder LLM will be forced to generate only text, which may or may not contain inline code examples.",
    )
    interact_parser.set_defaults(func=lambda args: InteractCommand(actions=args.actions))

    # Talk command
    talk_parser = subparsers.add_parser(
        "talk", help="Launches an interactive session with the Coder LLM, but without performing any actions."
    )
    talk_parser.set_defaults(func=lambda args: InteractCommand(actions=False))

    # Log command
    log_parser = subparsers.add_parser("log", help="Display past interaction history.")
    log_parser.add_argument(
        "--interaction",
        type=str,
        help="Display a specific interaction in detail by its unique ID or tag.",
    )
    log_parser.set_defaults(
        func=lambda args: LogCommand(interaction_identifier=args.interaction)
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
        worker_box = Ghostbox(
            endpoint=project.config.worker_endpoint,
            backend=project.config.worker_backend,  # Use string directly
            character_folder=os.path.join(
                current_project_root, ".ghostcode", project._WORKER_CHARACTER_FOLDER
            ),
            api_key=user_config.api_key,  # General API key
            google_api_key=user_config.google_api_key,
            openai_api_key=user_config.openai_api_key,
            **quiet_options,
        )
    except Exception as e:
        logger.error(f"Setting worker to dummy. Reason: {e}")
        worker_box = ghostbox.from_dummy(**quiet_options)

    try:
        coder_box = Ghostbox(
            endpoint=project.config.coder_endpoint,
            backend=project.config.coder_backend,  # Use string directly
            character_folder=os.path.join(
                current_project_root, ".ghostcode", project._CODER_CHARACTER_FOLDER
            ),
            api_key=user_config.api_key,  # General API key
            google_api_key=user_config.google_api_key,
            openai_api_key=user_config.openai_api_key,
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
    project.save_to_root(prog_instance.project_root)


if __name__ == "__main__":
    main()
