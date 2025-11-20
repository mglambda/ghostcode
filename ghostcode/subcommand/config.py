# ghostcode.subcommand.config
from typing import *
from pydantic import Field
import os
import sys
import json
from ..types import CommandOutput
from .. import types
from ..program import CommandInterface, Program
import logging
logger = logging.getLogger("ghostcode.subcommand.config")


class ConfigCommand(CommandInterface):
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
