# ghostcode.subcommand.discover
from typing import *
from pydantic import Field
import os
import sys
import re
import json
from ..types import CommandOutput
from .. import types
from ..utility import (
    EXTENSION_TO_LANGUAGE_MAP,
    language_from_extension
)
from ..program import CommandInterface, Program
import logging
logger = logging.getLogger("ghostcode.subcommand.discover")

class DiscoverCommand(CommandInterface):
    """Intelligently discovers files associated with the project and adds them to the context."""

    filepath: str = Field(
        description="The filepath that points to a directory in which files to add to the context will be recursively discovered."
    )

    languages_enabled: List[str] = Field(
        default_factory=list,
        description="Programming languages (e.g. 'python', 'javascript') for which files in the respective language will be added to the context.",
    )

    languages_disabled: List[str] = Field(
        default_factory=list,
        description="Providing e.g. --no-python --no-javascript will exclude languages from the discovery.",
    )
    min_lines: Optional[int] = Field(
        default=None,
        description="Minimum amount of lines that a file must have in order to be added to the context.",
    )

    max_lines: Optional[int] = Field(
        default=None,
        description="Maximum number of lines a file can have in order to be added to the context.",
    )

    size_heuristic: Optional[Literal["small", "medium", "large"]] = Field(
        default=None,
        description="If provided with --small, --medium, or --large, will automatically set the min_lines and max_lines parameters. Small means 0 - 1000 lines. Medium is 300 - 3000 lines. Large is 3000+. The small, medium, and large parameters are mutually exclusive.",
    )

    exclude_pattern: str = Field(
        default="",
        description="A regex that can be provided. File names that match against it are excluded from the context. The exclude pattern is applied at the very end of the discovery process.",
    )

    all: bool = Field(
        default=False,
        description="If provided, will add all (non-hidden) files to the context that are found in subdirectories of the given path. Providing --all is like providing all of the language parameters.",
    )

    # this containts e.g. "python", "javascript, "cpp". It should be used to create argparse parameters, like --python, --javascript, --cpp and so on
    possible_languages: ClassVar[List[str]] = list(
        set(list(EXTENSION_TO_LANGUAGE_MAP.values()))
    )

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
            effective_max_lines = None  # No upper limit for large

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
            dirs[:] = [d for d in dirs if not d.startswith("_")]

            for filename in files:
                abs_filepath = os.path.join(root, filename)
                relative_filepath = os.path.relpath(
                    abs_filepath, start=prog.project_root
                )

                # Skip hidden files/directories (e.g., .git, .ghostcode, files starting with .)
                if any(
                    part.startswith(".") for part in relative_filepath.split(os.sep)
                ):
                    logger.debug(f"Skipping hidden file: {relative_filepath}")
                    continue

                # 1. Apply language filter
                file_language = language_from_extension(abs_filepath)
                if file_language not in enabled_languages_set:
                    logger.debug(
                        f"Skipping '{relative_filepath}': Language '{file_language}' not enabled."
                    )
                    continue

                # 2. Apply line count filter
                try:
                    with open(
                        abs_filepath, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        lines = f.readlines()
                        line_count = len(lines)
                except Exception as e:
                    logger.warning(
                        f"Could not read file '{relative_filepath}' for line count: {e}. Skipping."
                    )
                    continue

                if effective_min_lines is not None and line_count < effective_min_lines:
                    logger.debug(
                        f"Skipping '{relative_filepath}': {line_count} lines, less than min_lines ({effective_min_lines})."
                    )
                    continue
                if effective_max_lines is not None and line_count > effective_max_lines:
                    logger.debug(
                        f"Skipping '{relative_filepath}': {line_count} lines, more than max_lines ({effective_max_lines})."
                    )
                    continue

                # 3. Apply exclude pattern filter
                if exclude_regex and exclude_regex.search(filename):
                    logger.debug(
                        f"Skipping '{relative_filepath}': Matches exclude pattern '{self.exclude_pattern}'."
                    )
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

