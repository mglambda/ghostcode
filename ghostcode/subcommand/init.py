# ghostcode.subcommand.init
from typing import *
from pydantic import Field
import os
import sys
from ..types import CommandOutput
from .. import types
from ..program import CommandInterface, Program
import logging
logger = logging.getLogger("ghostcode.subcommand.init")



class InitCommand(CommandInterface):
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

