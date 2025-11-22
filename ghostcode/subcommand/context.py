# ghostcode.subcommand.context
from typing import *
from pydantic import Field
import sys
import os
import glob
from ..types import CommandOutput
from .. import types
from ..program import CommandInterface, Program
from ..utility import show_model_nt
import logging
logger = logging.getLogger("ghostcode.subcommand.context")


class ContextCommand(CommandInterface):
    """Manages files included in the project context."""

    subcommand: Literal[
        "add", "rm", "remove", "ls", "clean", "lock", "unlock", "wipe", "list-summaries", "ignore", "force","default","summary"]

    filepaths: List[str] = Field(
        default_factory=list,
        description="File paths to add or remove, can include wildcards.",
    )
    rag: bool = Field(
        default=False,
        description="Enable Retrieval Augmented Generation for added files.",
    )

    @staticmethod
    def _maybe_visibility(subcommand: str) -> Optional[types.ContextFileVisibility]:
        try:
            return types.ContextFileVisibility[subcommand]
        except KeyError:
            return None
        
    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        project = prog.project
        current_context_files = {cf.filepath: cf for cf in project.context_files.data}
        visibility = self._maybe_visibility(self.subcommand)
        
        if self.subcommand == "ls":
            if not project.context_files.data:
                result.print("No context files tracked.")
            else:
                result.print("Tracked Context Files (relative to project root):")
                result.print(project.context_files.show_overview())
        elif self.subcommand == "list-summaries":
            if not project.context_files.data:
                result.print("No context files with summaries tracked.")
            else:
                result.print("Context File Summaries:")
                for cf in project.context_files.data:
                    if cf.config.summary:
                        result.print(f"# {cf.filepath}")
                        result.print(show_model_nt(cf.config.summary))
        elif self.subcommand in ["add", "rm", "remove"] or (visibility is not None):
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

            if (self.subcommand == "add") or (visibility is not None):
                visibility_str = "" if visibility is None else f" with visibility: {visibility}"                                    
                for fp in resolved_paths:
                    if fp not in current_context_files:
                        project.context_files.add_or_alter(
                            fp,
                            types.ContextFileConfig(
                                source=types.ContextFileSourceCLI(),
                                coder_visibility=visibility if visibility is not None else types.ContextFileVisibility.default
                            )
                        )                    
                        result.print(f"Added '{fp}' to context{visibility_str}.")
                    else:
                        # file already exists
                        project.context_files.add_or_alter(
                            fp,
                            types.ContextFileConfig(
                                source=types.ContextFileSourceCLI(),
                                coder_visibility=visibility if visibility is not None else types.ContextFileVisibility.default
                            )
                        )
                        result.print(f"Set visibility of {fp} to {visibility}")
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
                            result.print(
                                f"File '{fp}' is already {action_word}. Skipping."
                            )
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
                result.print(f"{n} file(s) removed from context.")
        return result
