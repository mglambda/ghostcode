# ghostcode.subcommand.verify
from ghostbox.definitions import LLMBackend
import os
import sys
import appdirs
from ..types import CommandOutput
from .. import types
from ..program import CommandInterface, Program
import logging
logger = logging.getLogger("ghostcode.subcommand.verify")


class VerifyCommand(CommandInterface):
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
