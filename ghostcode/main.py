from typing import *
from ghostcode import types
import ghostbox
from ghostbox import Ghostbox
from dataclasses import dataclass
import argparse
import logging
import os
import glob
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import sys
import json # For parsing config values

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('ghostcode.main')


@dataclass
class Program:
    """Holds program state for the main function.
    This instance is passed to command run methods."""
    project_root: Optional[str]
    project: Optional[types.Project]
    worker_box: Ghostbox
    coder_box: Ghostbox


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
    key: Optional[str] = Field(default=None, description="The configuration key to retrieve or set (e.g., 'coder_llm_config.model').")
    value: Optional[str] = Field(default=None, description="The new value for the configuration key.")

    def _get_config_value(self, project: types.Project, key: str) -> Any:
        """Helper to get config value from nested dictionaries/models."""
        parts = key.split('.', 1)
        if len(parts) == 1: # Top-level key (e.g., 'name' in metadata)
            if hasattr(project.project_metadata, key):
                return getattr(project.project_metadata, key)
            logger.warning(f"Config key '{key}' not found at top level of project metadata.")
            return None

        # Nested key (e.g., 'coder_llm_config.model')
        parent_key, sub_key = parts
        if parent_key == "coder_llm_config":
            return project.coder_llm_config.get(sub_key)
        elif parent_key == "worker_llm_config":
            return project.worker_llm_config.get(sub_key)
        elif parent_key == "project_metadata" and project.project_metadata:
            return getattr(project.project_metadata, sub_key, None)
        
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
            result.print("Current Coder LLM Config:")
            for k, v in project.coder_llm_config.items():
                result.print(f"  {k}: {v}")
            result.print("\nCurrent Worker LLM Config:")
            for k, v in project.worker_llm_config.items():
                result.print(f"  {k}: {v}")
            result.print("\nProject Metadata:")
            if project.project_metadata:
                for k, v in project.project_metadata.model_dump().items():
                    result.print(f"  {k}: {v}")
        elif self.subcommand == "get":
            if not self.key:
                logger.error("Key required for 'config get'.")
                sys.exit(1)
            val = self._get_config_value(project, self.key)
            if val is not None:
                result.print(f"{self.key}: {val}")
            else:
                logger.warning(f"Config key '{self.key}' not found.")
        elif self.subcommand == "set":
            if not self.key or self.value is None:
                logger.error("Key and value required for 'config set'.")
                sys.exit(1)
            self._set_config_value(project, self.key, self.value)
            project.save_to_root(prog.project_root)
            result.print(f"Config key '{self.key}' set to '{self.value}' and saved.")
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

# --- Main CLI Logic ---

def main():
    parser = argparse.ArgumentParser(
        prog="ghostcode",
        description="A command line tool to help programmers code using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter # For better help formatting
    )

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
    config_get_parser.add_argument("key", help="The configuration key to retrieve (e.g., 'coder_llm_config.model').")
    config_get_parser.set_defaults(func=lambda args: ConfigCommand(subcommand="get", key=args.key))

    config_set_parser = config_subparsers.add_parser("set", help="Set the value of a configuration option.")
    config_set_parser.add_argument("key", help="The configuration key to set (e.g., 'coder_llm_config.model').")
    config_set_parser.add_argument("value", help="The new value for the configuration key. Use JSON format for complex types (e.g., 'true', '123', '\"string\"', '[1,2]', '{\"key\":\"value\"}').")
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


    args = parser.parse_args()

    # Instantiate the command object
    command_obj: CommandInterface = args.func(args)

    # Special handling for 'init' command as it doesn't require an existing project
    if isinstance(command_obj, InitCommand):
        out = command_obj.run(Program(project_root=None, project=None, worker_box=None, coder_box=None)) # Dummy boxes for init
        print(out.text)
        sys.exit(0) # Exit after init

    # For all other commands, a project must exist
    project_root = types.Project.find_project_root()
    if not project_root:
        logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
        sys.exit(1)

    try:
        project = types.Project.from_root(project_root)
    except FileNotFoundError: # Should ideally be caught by find_project_root, but good to be safe
        logger.error(f"Project directory '.ghostcode' not found at {project_root}. This should not happen after root detection.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load ghostcode project from {project_root}: {e}", exc_info=True)
        sys.exit(1)

    # Initialize Ghostbox instances (dummy for now, as AI interaction is not yet implemented)
    # These would eventually be configured using project.worker_llm_config and project.coder_llm_config
    worker_box = ghostbox.from_generic(endpoint="http://localhost:8080", quiet=True, stdout=False, stderr=False)
    coder_box = ghostbox.from_generic(quiet=True, stdout=False, stderr=False)

    # Create the Program instance
    prog_instance = Program(
        project_root=project_root,
        project=project,
        worker_box=worker_box,
        coder_box=coder_box
    )

    # Run the command
    out = command_obj.run(prog_instance)
    print(out.text)
    

if __name__ == "__main__":
    main()



