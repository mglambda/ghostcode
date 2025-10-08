from typing import *
import json
import traceback
from ghostcode import types
import ghostbox
import os
from ghostbox import Ghostbox
from ghostbox.definitions import BrokenBackend
from dataclasses import dataclass, field
import argparse
import logging
import os
import glob
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import sys
import json # For parsing config values
import yaml # For ProjectConfig
import appdirs # Added for UserConfig path
from ghostbox.definitions import LLMBackend # Added for API key checks

# logger will be configured after argument parsing
logger: logging.Logger # Declare logger globally, will be assigned later

def _configure_logging(log_mode: str, project_root: Optional[str]):
    """
    Configures the root logger based on the specified mode and project root.
    """
    # Clear existing handlers to prevent duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set the root logger level to INFO by default
    logging.root.setLevel(logging.INFO)
    
    if log_mode == "off":
        logging.root.setLevel(logging.CRITICAL + 1) # Effectively disable all logging
        return

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_mode == "stderr":
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        print("Logging to stderr.", file=sys.stderr)
    elif log_mode == "file":
        log_filepath = None
        if project_root:
            ghostcode_dir = os.path.join(project_root, ".ghostcode")
            if os.path.isdir(ghostcode_dir):
                log_filepath = os.path.join(ghostcode_dir, "log.txt")
        
        if log_filepath:
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
                handler = logging.FileHandler(log_filepath)
                handler.setFormatter(formatter)
                logging.root.addHandler(handler)
                # Also add a stderr handler for critical errors, so they are always visible
                stderr_handler = logging.StreamHandler(sys.stderr)
                stderr_handler.setFormatter(formatter)
                stderr_handler.setLevel(logging.ERROR) # Only show ERROR and CRITICAL to stderr
                logging.root.addHandler(stderr_handler)
                print(f"Logging to file: {log_filepath}", file=sys.stderr) # Inform user where logs are going
            except Exception as e:
                # Fallback to stderr if file logging fails
                print(f"WARNING: Failed to set up file logging to {log_filepath}: {e}. Falling back to stderr.", file=sys.stderr)
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(formatter)
                logging.root.addHandler(handler)
        else:
            # Fallback to stderr if project_root or .ghostcode dir is not found
            print("WARNING: Project root or .ghostcode directory not found for file logging. Falling back to stderr.", file=sys.stderr)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            logging.root.addHandler(handler)


@dataclass
class Program:
    """Holds program state for the main function.
    This instance is passed to command run methods."""
    project_root: Optional[str]
    project: Optional[types.Project]
    worker_box: Ghostbox
    coder_box: Ghostbox

    user_config: types.UserConfig = field(
        default_factory = types.UserConfig,
        )
    
    def _get_cli_prompt(self) -> str:
        """Returns the CLI prompt used in the interact command and any other REPL like interactions with the LLMs."""
        # some ghostbox internal magic to get the token count
        coder_tokens = self.coder_box._plumbing._get_last_result_tokens()
        worker_tokens = self.worker_box._plumbing._get_last_result_tokens()
        return f" 👻{coder_tokens} 🔧{worker_tokens} >"
    def _has_api_keys(self) -> Dict[LLMBackend, bool]:
        """
        Compares the chosen backends to the user config and checks for required API keys.
        
        Returns:
            Dict[LLMBackend, bool]: A dictionary where keys are LLMBackend enum members
                                    and values are True if the API key is present, False otherwise.
                                    Only includes backends that are actually used and require keys.
        """
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
                    missing_keys[LLMBackend.openai] = False # Assume OpenAI key is preferred for OpenAI endpoints
                # else:
                #     missing_keys[LLMBackend.openai] = True
            elif "googleapis.com" in self.project.config.coder_endpoint:
                if not self.user_config.google_api_key and not self.user_config.api_key:
                    missing_keys[LLMBackend.google] = False # Assume Google key is preferred for Google endpoints
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


# --- Command Interface and Implementations ---

class CommandOutput(BaseModel):
    """Helper class to package output from commands.
    Mostly will just contain text, but this exists so we can add JSON etc. in the future."""

    text: str = ""
    data: Optional[Dict[str, Any] ] = None

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
    path: Optional[str] = Field(default=None, description="Optional path to initialize the project. Defaults to current directory.")

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        target_root = os.path.abspath(self.path if self.path else os.getcwd())
        try:
            types.Project.init(target_root)
            result.text += f"Ghostcode project initialized successfully in {target_root}"
        except Exception as e:
            logger.error(f"Failed to initialize project in {target_root}: {e}")
            sys.exit(1)

        return result

class ConfigCommand(BaseModel, CommandInterface):
    """Manages ghostcode project configuration."""
    subcommand: Literal["set", "get", "ls"]
    key: Optional[str] = Field(default=None, description="The configuration key to retrieve or set (e.g., 'coder_llm_config.model', 'config.coder_backend').")
    value: Optional[str] = Field(default=None, description="The new value for the configuration key.")

    def _get_config_value(self, project: types.Project, key: str) -> Any:
        """Helper to get config value from nested dictionaries/models."""
        parts = key.split('.', 1)
        if len(parts) == 1: # Top-level key (e.g., 'name' in metadata)
            if hasattr(project.project_metadata, key):
                return getattr(project.project_metadata, key)
            logger.warning(f"Config key '{key}' not found at top level of project metadata.")
            return None

        # Nested key (e.g., 'coder_llm_config.model', 'config.coder_backend')
        parent_key, sub_key = parts
        if parent_key == "coder_llm_config":
            return project.coder_llm_config.get(sub_key)
        elif parent_key == "worker_llm_config":
            return project.worker_llm_config.get(sub_key)
        elif parent_key == "project_metadata" and project.project_metadata:
            return getattr(project.project_metadata, sub_key, None)
        elif parent_key == "config": # Handle ProjectConfig
            if hasattr(project.config, sub_key):
                # Now that backend fields are strings, just return the value directly
                return getattr(project.config, sub_key)
            logger.warning(f"Sub-key '{sub_key}' not found in project.config.")
            return None

        logger.warning(f"Config key '{key}' not found or invalid parent key '{parent_key}'.")
        return None

    def _set_config_value(self, project: types.Project, key: str, value_str: str) -> None:
        """Helper to set config value in nested dictionaries/models."""
        parsed_value = self._parse_value(value_str)
        parts = key.split('.', 1)

        if len(parts) == 1: # Top-level key in metadata
            if hasattr(project.project_metadata, key):
                setattr(project.project_metadata, key, parsed_value)
                logger.debug(f"Set project_metadata.{key} to {parsed_value}")
            else:
                logger.warning(f"Config key '{key}' not found at top level of project metadata for setting.")
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
                logger.warning(f"Sub-key '{sub_key}' not found in project_metadata for setting.")
        elif parent_key == "config": # Handle ProjectConfig
            if hasattr(project.config, sub_key):
                # No special conversion needed here, as ProjectConfig fields are now strings
                setattr(project.config, sub_key, parsed_value)
                logger.debug(f"Set project.config.{sub_key} to {parsed_value}")
            else:
                logger.warning(f"Sub-key '{sub_key}' not found in project.config for setting.")
        else:
            logger.warning(f"Config key '{key}' not recognized for setting.")

    def _parse_value(self, value_str: str) -> Any:
        """Attempt to parse string to int, float, bool, or keep as string."""
        try:
            return json.loads(value_str) # Try parsing as JSON (handles bool, int, float, list, dict, null)
        except json.JSONDecodeError:
            return value_str # If not JSON, treat as string

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
                    logger.warning(f"Config key '{self.key}' not found in project or user configuration.")
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
            initial_project_metadata_dump = project.project_metadata.model_dump_json() if project.project_metadata else "{}"

            self._set_config_value(project, self.key, self.value)
            
            # Check if any project config was actually changed
            project_config_changed = (
                project.config.model_dump_json() != initial_project_config_dump or
                json.dumps(project.coder_llm_config) != initial_coder_llm_config_dump or
                json.dumps(project.worker_llm_config) != initial_worker_llm_config_dump or
                (project.project_metadata and project.project_metadata.model_dump_json() != initial_project_metadata_dump)
            )

            if project_config_changed:
                project.save_to_root(prog.project_root)
                result.print(f"Project config key '{self.key}' set to '{self.value}' and saved.")
            elif hasattr(prog.user_config, self.key):
                # If not a project config key, try user config
                setattr(prog.user_config, self.key, self._parse_value(self.value))
                prog.user_config.save()
                result.print(f"User config key '{self.key}' set to '{self.value}' and saved.")
            else:
                logger.warning(f"Config key '{self.key}' not found in project or user configuration for setting.")
        return result

class ContextCommand(BaseModel, CommandInterface):
    """Manages files included in the project context."""
    subcommand: Literal["add", "rm", "remove", "ls"]
    filepaths: List[str] = Field(default_factory=list, description="File paths to add or remove, can include wildcards.")
    rag: bool = Field(default=False, description="Enable Retrieval Augmented Generation for added files.")

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
                        relative_to_project_root = os.path.relpath(matched_path, start=prog.project_root)
                        resolved_paths.add(relative_to_project_root)
                    else:
                        logger.warning(f"Skipping '{matched_path}': Not a file or does not exist.")

            if self.subcommand == "add":
                for fp in resolved_paths:
                    if fp not in current_context_files:
                        project.context_files.data.append(types.ContextFile(filepath=fp, rag=self.rag))
                        result.print(f"Added '{fp}' to context (RAG={self.rag}).")
                    else:
                        # If already exists, update RAG status if different
                        existing_cf = current_context_files[fp]
                        if existing_cf.rag != self.rag:
                            existing_cf.rag = self.rag
                            result.print(f"Updated RAG status for '{fp}' to {self.rag}.")
                        else:
                            result.print(f"'{fp}' is already in context with RAG={self.rag}.")
            elif self.subcommand in ["rm", "remove"]:
                initial_count = len(project.context_files.data)
                project.context_files.data = [
                    cf for cf in project.context_files.data if cf.filepath not in resolved_paths
                ]
                removed_count = initial_count - len(project.context_files.data)
                if removed_count > 0:
                    result.print(f"Removed {removed_count} file(s) from context.")
                else:
                    result.print("No matching files found in context to remove.")

            project.save_to_root(prog.project_root)
            result.print("Context files updated and saved.")
        return result

class InteractCommand(BaseModel, CommandInterface):
    """Launches an interactive session with the Coder LLM."""

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

        result.print("Starting interactive session with Coder LLM (placeholder).")
        result.print("API keys checked. If warnings were shown, please address them.")

        # start the actual loop in another method 
        return self._interact_loop(result, prog)

    # This code is unreachable but in the future error handling/control flow of this method might become more complicated and we may need it
        return result

    def _print(self, w: str, end: str = "\n") -> None:
        """Synonymous with print. Just aliases here as future-proofing."""
        print(w, end=end)


    def _make_user_prompt(self, prog: Program, prompt: str) -> str:
        return f"""{prog.project.context_files.show()}

# User Prompt

{prompt}"""        
    def _interact_loop(self, intermediate_result: CommandOutput, prog: Program) -> CommandOutput:
        # since the interaction is blocking we can't wait with printing until we return the result
        # so we print it here and return an empty result at the end
        self._print(intermediate_result.text)
        interacting = True
        print("Multiline mode enabled. Type your prompt over multiple lines.\nType a single '\\' and hit enter to submit.\nType /quit or CTRL+D to quit.")
        prompt = ""
        current_history = []
        while interacting:
            try:
                line = input(prog._get_cli_prompt())
            except EOFError:
                break
            
            if line == "/quit":
                break
            elif line == "/lastrequest":
                print(json.dumps(prog.coder_box.get_last_request(), indent=4))
                continue
            elif line == "/show":
                for item in current_history:
                    print(item.show())
                continue
            elif line == "/save":
                with open("out.txt", "w") as f:
                    for item in current_history:
                        f.write(item.show())
                continue
            elif line != "\\":
                prompt += "\n" + line
                continue

            # ok, process prompt
            prog.coder_box.set_vars({
                "project_metadata": "", # placeholder - should be filled in with textual representation of actual metadata
                #"context_files": prog.project.context_files.show()
            })

            debug_options = {
                "stderr":True,
                "stdout":True,
                "quiet":False,
                "verbose":True
            }
            try:
                response = prog.coder_box.new(types.CoderResponse,
                                              self._make_user_prompt(prog, prompt),
                                              #options=debug_options,
                                              )
                current_history.append(response)
                for part in response.contents:
                    print(part.show_cli())
            except:
                print(f"error on ghostbox.new. See the full traceback:\n{traceback.format_exc()}")
                print(f"\nAnd here is the last result:\n\n{prog.coder_box.get_last_result()}") 

            prompt = ""            
        

        return CommandOutput(text="Finished interaction.")

# --- Main CLI Logic ---

class VerifyCommand(BaseModel):
    """Command to verify program integrity and configuration.
    This is currently only used internally to check for API keys."""

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        result.data = {"error": False}
        
        missing_api_keys = prog._has_api_keys()
        if missing_api_keys:
            result.data["error"] = True
            user_config_path = os.path.join(appdirs.user_config_dir('ghostcode'), types.UserConfig._GHOSTCODE_CONFIG_FILE)
            result.print("\n--- WARNING: Missing API Keys for Cloud LLMs ---")
            result.print(f"To use the configured cloud LLMs, please provide the necessary API keys in your user configuration file:")
            result.print(f"  {user_config_path}")
            result.print("You can edit this file directly or use 'ghostcode config set' for user-specific API keys.")
            result.print("--------------------------------------------------")
            for backend, _ in missing_api_keys.items():
                if backend == LLMBackend.google:
                    result.print(f"  - Google AI Studio API key is missing for backend '{prog.project.config.coder_backend}' or '{prog.project.config.worker_backend}'.")
                    result.print("    Get your key at: https://aistudio.google.com/app/apikey")
                elif backend == LLMBackend.openai:
                    result.print(f"  - OpenAI API key is missing for backend '{prog.project.config.coder_backend}' or '{prog.project.config.worker_backend}'.")
                    result.print("    Get your key at: https://platform.openai.com/account/api-keys")
            result.print("--------------------------------------------------\n")
        return result
    
    
def main():
    parser = argparse.ArgumentParser(
        prog="ghostcode",
        description="A command line tool to help programmers code using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter # For better help formatting
    )

    # Add --user-config argument
    parser.add_argument("-c", "--user-config", type=str, default=None,
                        help=f"Path to a custom user configuration file (YAML format). "
                             f"Defaults to {os.path.join(appdirs.user_config_dir('ghostcode'), types.UserConfig._GHOSTCODE_CONFIG_FILE)}.")

    # Add --logging argument
    parser.add_argument("--logging", type=str, choices=["off", "file", "stderr"], default="file",
                        help="Configure logging output: 'off', 'file' (to .ghostcode/log.txt), or 'stderr'.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new ghostcode project in the current directory or a specified path.")
    init_parser.add_argument("path", nargs="?", default=None, help="Optional path to initialize the project. Defaults to current directory.")
    init_parser.set_defaults(func=lambda args: InitCommand(path=args.path))

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage ghostcode project configuration.")
    config_subparsers = config_parser.add_subparsers(dest="subcommand", required=True, help="Config subcommands")

    config_ls_parser = config_subparsers.add_parser("ls", help="List all configuration options.")
    config_ls_parser.set_defaults(func=lambda args: ConfigCommand(subcommand="ls"))

    config_get_parser = config_subparsers.add_parser("get", help="Get the value of a specific configuration option.")
    config_get_parser.add_argument("key", help="The configuration key to retrieve (e.g., 'coder_llm_config.model', 'config.coder_backend').")
    config_get_parser.set_defaults(func=lambda args: ConfigCommand(subcommand="get", key=args.key))

    config_set_parser = config_subparsers.add_parser("set", help="Set the value of a configuration option.")
    config_set_parser.add_argument("key", help="The configuration key to set (e.g., 'coder_llm_config.model', 'config.coder_backend').")
    config_set_parser.add_argument("value", help="The new value for the configuration key. Use JSON format for complex types (e.g., 'true', '123', '\"string\"', '[1,2]', '{\"key\":\"value\"}'). For backend types, use the string name (e.g., 'google', 'generic').")
    config_set_parser.set_defaults(func=lambda args: ConfigCommand(subcommand="set", key=args.key, value=args.value))

    # Context command
    context_parser = subparsers.add_parser("context", help="Manage files included in the project context.")
    context_subparsers = context_parser.add_subparsers(dest="subcommand", required=True, help="Context subcommands")

    context_ls_parser = context_subparsers.add_parser("ls", aliases=["list"], help="List all files in the project context.")
    context_ls_parser.set_defaults(func=lambda args: ContextCommand(subcommand="ls"))

    context_add_parser = context_subparsers.add_parser("add", help="Add file(s) to the project context. Supports wildcards.")
    context_add_parser.add_argument("filepaths", nargs="+", help="One or more file paths (can include wildcards like '*.py', 'src/**.js').")
    context_add_parser.add_argument("--rag", action="store_true", help="Enable Retrieval Augmented Generation for these files.")
    context_add_parser.set_defaults(func=lambda args: ContextCommand(subcommand="add", filepaths=args.filepaths, rag=args.rag))

    context_rm_parser = context_subparsers.add_parser("rm", aliases=["remove"], help="Remove file(s) from the project context. Supports wildcards.")
    context_rm_parser.add_argument("filepaths", nargs="+", help="One or more file paths (can include wildcards).")
    context_rm_parser.set_defaults(func=lambda args: ContextCommand(subcommand="rm", filepaths=args.filepaths))

    # Interact command (placeholder)
    interact_parser = subparsers.add_parser("interact", help="Launches an interactive session with the Coder LLM.")
    interact_parser.set_defaults(func=lambda args: InteractCommand())


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
    _configure_logging(args.logging, current_project_root)

    # Now that logging is configured, get the main logger instance
    global logger # Declare logger as global to assign to it
    logger = logging.getLogger('ghostcode.main')

    # Load UserConfig
    user_config_path = args.user_config
    user_config: types.UserConfig
    try:
        user_config = types.UserConfig.load(user_config_path)
    except FileNotFoundError:
        logger.info(f"User configuration file not found. Creating a default one.")
        user_config = types.UserConfig()
        user_config.save(user_config_path) # Save to the default or specified path
    except Exception as e:
        logger.error(f"Failed to load user configuration: {e}. Using default settings.", exc_info=True)
        user_config = types.UserConfig()


    # Instantiate the command object
    command_obj: CommandInterface = args.func(args)

    # Special handling for 'init' command as it doesn't require an existing project
    if isinstance(command_obj, InitCommand):
        # The project_root for init is already determined as current_project_root
        # and logging is configured. Just run the command.
        out = command_obj.run(Program(project_root=current_project_root, project=None, worker_box=None, coder_box=None, user_config=user_config)) # Pass user_config
        print(out.text)
        sys.exit(0) # Exit after init

    # For all other commands, a project must exist
    if not current_project_root: # This would have been set by find_project_root
        logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
        sys.exit(1)

    try:
        project = types.Project.from_root(current_project_root)
    except FileNotFoundError: # Should ideally be caught by find_project_root, but good to be safe
        logger.error(f"Project directory '.ghostcode' not found at {current_project_root}. This should not happen after root detection.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load ghostcode project from {current_project_root}: {e}", exc_info=True)
        sys.exit(1)

    # Initialize Ghostbox instances using project.config for endpoints and backends
    # the quiet options have to be passed here or we will get a couple of stray messages until the project box configs are read
    quiet_options = {
        "quiet": True,
        "stdout": False,
        "stderr": False
    }
    
    # Now directly use the string values from project.config
    # Pass API keys from user_config to Ghostbox instances
    # we catch missing API key and other backend initialization errors here, and inform the user later in commands that actually require backends
    try:
        worker_box = Ghostbox(endpoint=project.config.worker_endpoint,
                              backend=project.config.worker_backend, # Use string directly
                              character_folder=os.path.join(current_project_root, ".ghostcode", project._WORKER_CHARACTER_FOLDER),
                              api_key=user_config.api_key, # General API key
                              google_api_key=user_config.google_api_key,
                              openai_api_key=user_config.openai_api_key,
                              **quiet_options)
    except Exception as e:
        logger.error(f"Setting worker to dummy. Reason: {e}")
        worker_box = ghostbox.from_dummy(**quiet_options)

    try:
        coder_box = Ghostbox(endpoint=project.config.coder_endpoint,
                             backend=project.config.coder_backend, # Use string directly
                             character_folder=os.path.join(current_project_root, ".ghostcode", project._CODER_CHARACTER_FOLDER),
                             api_key=user_config.api_key, # General API key
                             google_api_key=user_config.google_api_key,
                             openai_api_key=user_config.openai_api_key,
                             **quiet_options)
    except Exception as e:
        logger.error(f"Setting coder backend to dummy. Reason: {e}")
        coder_box = ghostbox.from_dummy(**quiet_options)

    # Create the Program instance
    prog_instance = Program(
        project_root=current_project_root,
        project=project,
        worker_box=worker_box,
        coder_box=coder_box,
        user_config=user_config # Pass the loaded user_config
    )

    # Run the command
    out = command_obj.run(prog_instance)
    print(out.text)


if __name__ == "__main__":
    main()



