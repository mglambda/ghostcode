# ghostcode/types.py
from typing import *
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import os
import uuid
import subprocess
import traceback
import json
import yaml
import logging
import ghostbox.definitions  # type: ignore
from ghostbox.definitions import LLMBackend  # type: ignore
from ghostcode.ansi_colors import Color256, colored
from ghostbox import Ghostbox
from ghostcode.utility import (
    language_from_extension,
    timestamp_now_iso8601,
    show_model,
    PydanticEnumDumper,
    make_mnemonic,
)
from ghostcode import shell
import appdirs  # Added for platform-specific config directory

# --- Logging Setup ---
# Configure a basic logger for the ghostcode project
logger = logging.getLogger("ghostcode.types")

# --- Default Configurations ---
# Default configuration for the coder LLM (e.g., for planning, complex reasoning)
DEFAULT_CODER_LLM_CONFIG = {
    "model": "models/gemini-2.5-flash",
    "temperature": 0.2,
    "max_length": -1,
    "top_p": 0.9,
    "log_time": True,
    "verbose": False,
    "chat_ai": "GhostCoder",
    "stdout": False,
    "stderr": False,
    "quiet": True,
    "stream": False,
}

# Default configuration for the worker LLM (e.g., for generating code snippets, answering questions)
DEFAULT_WORKER_LLM_CONFIG = {
    "model": "llama3",  # Placeholder, user should configure their local model
    "temperature": 0.7,
    "max_length": -1,
    "top_p": 0.95,
    "log_time": True,
    "verbose": False,
    "stdout": False,
    "stderr": False,
    "quiet": True,
    "stream": False,
    "chat_ai": "GhostWorker",
}

# Default project metadata
DEFAULT_PROJECT_METADATA = {
    "name": "New Ghostcode Project",
    "description": "A new project initialized with ghostcode. Edit this description to provide an overview of your project.",
}


class ClearanceRequirement(Enum):
    """Defines the level of user clearance required for an action.
    Higher values indicate a higher requirement for user interaction/permission.
    """

    AUTOMATIC = 10  # Actions that are generally safe, non-destructive, or purely informational (e.g., logging, internal state changes). Can proceed without explicit user confirmation.
    INFORM = 20  # Actions that are safe but the user should be made aware of, typically non-critical file creations (e.g., temporary files, new log files) or minor, reversible changes. User is informed, but no explicit confirmation is strictly required unless configured otherwise.
    CONFIRM = 30  # Actions that modify existing user-created files, create new permanent files, or execute commands that might have noticeable side effects (e.g., `pip install`). Requires explicit 'yes/no' user confirmation by default.
    DANGEROUS = 40  # Actions with significant potential for data loss, system instability, or execution of arbitrary, untrusted code (e.g., `rm -rf`, running downloaded scripts). Requires strong user confirmation (e.g., typing the full word 'confirm').
    FORBIDDEN = 50  # Actions that the AI is explicitly forbidden from performing under any circumstances. Attempting these actions should result in an immediate halt and error.


class UserConfirmation(Enum):
    """Possible results of a user confirmation dialog."""

    YES = 1
    NO = 2
    ALL = 3
    CANCEL = 4
    SHOW_MORE = 5

    @staticmethod
    def is_confirmation(value: "UserConfirmation") -> bool:
        return value in [UserConfirmation.YES, UserConfirmation.ALL]


class UserConfig(BaseModel):
    """Stores user specific data, like names, emails, and api keys.
    Usually stored in a .ghostcodeconfig file, in a platform specific location. The user settings are used across multiple ghostcode projects, and so aren't stored in the .ghostcode project folder.
    """

    name: str = ""
    email: str = ""
    api_key: str = Field(
        default="",
        description="The most general API key. This is used for a LLM backend service that requires an API key if none of the specific API keys are set.",
    )

    openai_api_key: str = Field(
        default="",
        description="Used with Chat-GPT and the OpenAI LLM backend. Get your API key at https://openai.com\nIf this key is set and you use openai as your backend, it will have precedence over the generic api_key.",
    )

    google_api_key: str = Field(
        default="",
        description="Use with Gemini and the Google AI Studio API. Get your API key at https://aistudio.google.com\nIf this key is set and you use google as your backend, it will have precedence over the generic api_key.",
    )

    _GHOSTCODE_CONFIG_FILE: ClassVar[str] = ".ghostcodeconfig"

    def save(self, user_config_path: Optional[str] = None) -> None:
        """
        Saves the UserConfig instance to a YAML file.

        Args:
            user_config_path (Optional[str]): The full path to save the config file.
                                              If None, defaults to a platform-specific user config directory.
        """
        if user_config_path is None:
            config_dir = appdirs.user_config_dir("ghostcode")
            os.makedirs(config_dir, exist_ok=True)
            save_path = os.path.join(config_dir, self._GHOSTCODE_CONFIG_FILE)
        else:
            save_path = os.path.abspath(user_config_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logger.info(f"Saving user configuration to {save_path}")

        # Add helpful comments to the YAML output
        config_data = self.model_dump()
        yaml_content = f"""# Ghostcode User Configuration
# This file stores your personal settings, including API keys,
# which are used across all your ghostcode projects.
#
# API Keys:
#   - 'api_key': General API key for LLM services if no specific key is provided.
#   - 'openai_api_key': Specific API key for OpenAI (e.g., GPT-4o).
#     Get it from https://platform.openai.com/account/api-keys
#   - 'google_api_key': Specific API key for Google AI Studio (e.g., Gemini).
#     Get it from https://aistudio.google.com/app/apikey
#
# To fill in an API key, replace the empty string "" with your actual key.
# Example:
# openai_api_key: "sk-YOUR_OPENAI_API_KEY_HERE"
#
"""
        yaml_content += yaml.dump(config_data, indent=2, sort_keys=False)

        try:
            with open(save_path, "w") as f:
                f.write(yaml_content)
            logger.debug(f"User configuration saved successfully to {save_path}")
        except Exception as e:
            logger.error(
                f"Failed to save user configuration to {save_path}: {e}", exc_info=True
            )
            raise

    @staticmethod
    def load(user_config_path: Optional[str] = None) -> "UserConfig":
        """
        Loads a UserConfig instance from a YAML file.

        Args:
            user_config_path (Optional[str]): The full path to load the config file from.
                                              If None, defaults to a platform-specific user config directory.

        Returns:
            UserConfig: The loaded UserConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist at the specified or default path.
            yaml.YAMLError: If there's an error parsing the YAML content.
        """
        if user_config_path is None:
            config_dir = appdirs.user_config_dir("ghostcode")
            load_path = os.path.join(config_dir, UserConfig._GHOSTCODE_CONFIG_FILE)
        else:
            load_path = os.path.abspath(user_config_path)

        logger.info(f"Attempting to load user configuration from {load_path}")

        if not os.path.isfile(load_path):
            logger.debug(f"User configuration file not found at {load_path}")
            raise FileNotFoundError(f"User configuration file not found at {load_path}")

        try:
            with open(load_path, "r") as f:
                config_data = yaml.safe_load(f)
            if config_data is None:
                logger.warning(
                    f"User configuration file {load_path} is empty. Returning default UserConfig."
                )
                return UserConfig()

            user_config = UserConfig(**config_data)
            logger.debug(f"User configuration loaded successfully from {load_path}")
            return user_config
        except yaml.YAMLError as e:
            logger.error(f"Error decoding YAML from {load_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {load_path}: {e}", exc_info=True
            )
            raise


class ProjectConfig(BaseModel):
    """Contains project wide configuration obptions.
    This is stored in project_root/config.yaml"""

    # Changed these to str for easier human readability/editing in YAML
    coder_backend: str = Field(
        default=ghostbox.definitions.LLMBackend.google.name,
        description="Backend for the Coder LLM (e.g., 'google', 'openai', 'generic').",
    )
    worker_backend: str = Field(
        default=ghostbox.definitions.LLMBackend.generic.name,
        description="Backend for the Worker LLM (e.g., 'llamacpp', 'generic').",
    )
    coder_endpoint: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Endpoint for the Coder LLM.",
    )
    worker_endpoint: str = Field(
        default="http://localhost:8080", description="Endpoint for the Worker LLM."
    )

    coder_clearance_level: ClearanceRequirement = Field(
        default=ClearanceRequirement.INFORM,
        description="The clearance level that the coder LLM has by default. If the coder's clearance level meets or exceeds the clearance requirement for a given action, it is permitted to perform that action without user confirmation.",
    )

    worker_clearance_level: ClearanceRequirement = Field(
        default=ClearanceRequirement.INFORM,
        description="The clearance level that the worker LLM has by default. If the worker's clearance level meets or exceeds the clearance requirement for a given action, it is permitted to perform that action without user confirmation.",
    )


# --- Type Definitions ---
class ContextFile(BaseModel):
    """Abstract representation of a filepath along with metadata. Context files are sent to the cloud LLM for prompts."""

    filepath: str = Field(
        description="The filepath to the context file. This is always relative to the directory that the .ghostcode directory is in."
    )

    rag: bool = Field(
        default=False,
        description="Whether to enable retrieval augmented generation for this file. Enabling RAG means the local LLM will retrieve only parts of the file if it deems it necessary with the given prompt. This is usually done for large text files or documentation, and helps to avoid huge token counts for the cloud LLM.",
    )


class ContextFiles(BaseModel):
    """Encapsulates the files (both code and otherwise) that are tracked by the project. Tracked files are sent to the cloud LLM with the prompt.
    To keep it simple and human-readable, filepaths are stored in .ghostcode/context_files, one per line, all relative to the directory that the .ghostcode directory is in. This file may change frequently.
    A separate .ghostcode/context_files_options stores metadata about the individual files.
    """

    data: List[ContextFile] = Field(default_factory=list)

    def to_plaintext(self) -> str:
        """Serializes the list of context file paths to a plaintext string, one path per line."""
        return "\n".join([cf.filepath for cf in self.data])

    @staticmethod
    def from_plaintext(content: str) -> "ContextFiles":
        """Deserializes a plaintext string into a ContextFiles object.
        Each line is treated as a filepath."""
        filepaths = [line.strip() for line in content.splitlines() if line.strip()]
        # For now, RAG is always False when loaded from plaintext, as there's no metadata in this format.
        # A future enhancement might involve a separate file for options or a more complex format.
        context_files = [ContextFile(filepath=fp, rag=False) for fp in filepaths]
        return ContextFiles(data=context_files)

    def show(self) -> str:
        """Renders file contents of the context files to a string, in a format that is suitable for an LLM."""
        w = ""
        for context_file in self.data:
            try:
                with open(context_file.filepath, "r") as f:
                    contents = f.read()
            except Exception as e:
                logging.error(
                    f"Could not read file {context_file.filepath}. Skipping. Reason: {e}"
                )
                continue

            w += f"""# {context_file.filepath}

```{language_from_extension(context_file.filepath)}
{contents}
```

"""
        return w

    def show_cli(self) -> str:
        """Shows the list of filepaths in a short, command line interface friendly manner."""
        return "(" + " ".join([item.filepath for item in self.data]) + ")"


class ProjectMetadata(BaseModel):
    name: str
    description: str

    def show(self) -> str:
        """Returns a human-readable string representation of the project metadata."""
        return f"## Project Metadata\n{show_model(self)}\n"


class Showable(Protocol):
    """Interface for anything that can be turned into strings.
    By convention, the CLI method turns objects into strings in a way that is slightly less verbose and more friendly to humans, though it is not required that show and show_cli have different return values at all.
    """

    def show_cli(self) -> str:
        pass

    def show(self) -> str:
        pass


### ResponseParts ###
# ResponseParts represent structured data that can be sent back by an LLM backend.
# These types are generally shared between coder and worker backends, though they both use their own subsets for during any particular request, and may use a subset of these that is appropriate to the particular request and context in which it happens.
# See also the various Action types further below.


class CodeResponsePart(BaseModel):
    """Represents a segment of programming code generated by the LLM backend.
    This can be either a full file replacement or a partial modification within an existing file.

    Crucially, if the 'type' is 'partial', the 'original_code' field MUST be accurately provided.
    This 'original_code' block is used by GhostWorker to precisely locate the section of code
    to be replaced within the specified 'filepath'. Without an exact or very close match
    for 'original_code', the partial replacement operation will fail.
    """

    type: Literal["full", "partial"] = Field(
        default="full",
        description="Indicates whether this code response part is a full or partial code block.\nFull code blocks are intended to replace entire files, and do not require an 'original_code' block.\nA partial response replaces only a specific segment of a file, and therefore *must* include the 'original_code' block that is to be replaced, along with the 'new_code' replacement. The 'original_code' should include a few lines of context before and after the actual change to aid precise location.",
    )

    filepath: Optional[str] = Field(
        default=None,
        description="The path to the file where this code generation belongs or should be saved.\nThis path is always relative to the project root. If None, the code is a free-standing snippet not intended for file modification.",
    )

    language: str = Field(
        default="",
        description="The programming language of the code (e.g., 'python', 'javascript', 'markdown').",
    )

    original_code: Optional[str] = Field(
        default=None,
        description="The exact original code block that is to be replaced by 'new_code'.\nThis field is MANDATORY and must be accurate if 'type' is 'partial'. It should include the surrounding lines of code (e.g., 2-3 lines before and after the actual change) to provide sufficient context for GhostWorker to locate the block reliably within the file.",
    )

    new_code: str = Field(
        default="",
        description="The actual new or modified code as plaintext that will replace 'original_code' (for partial) or the entire file (for full).",
    )

    context_anchor: Optional[str] = Field(
        default=None,
        description="A descriptive string identifying the containing syntactic construct (e.g., 'class MyClass:', 'def my_function(arg):', '# Main loop').\nThis helps GhostWorker narrow down the search area for 'original_code' in complex or large files, especially when multiple identical code snippets might exist.",
    )
    notes: List[str] = Field(
        default_factory=lambda: [],
        description="Concise notes, phrased as pull request comments, explaining the rationale, design decisions, assumptions, or potential implications of the generated code. This helps with code review and understanding the intent. Can be left empty if the change is self-explanatory. Remember that commit messages use present imperative tense.",
    )

    title: str = Field(
        default="",
        description="A short, descriptive title for the code block or the change it represents.",
    )

    def show_cli(self) -> str:
        """Return a (short) representation that is suitable for the CLI interface."""
        filepath_str = f"({self.filepath})" if self.filepath else ""
        notes_str = (
            "\n" + "\n".join([f" - {note}" for note in self.notes])
            if self.notes
            else ""
        )
        return f"## {self.title} {filepath_str}{notes_str}"

    def show(self) -> str:
        """Returns a comprehensive string representation of the part."""
        filepath_str = f" ({self.filepath})" if self.filepath else ""
        notes_str = (
            "\n" + "\n".join([f" - {note}" for note in self.notes])
            if self.notes
            else ""
        )
        anchor_str = f"\nacnhor: {self.context_anchor}\n" if self.context_anchor else ""
        if self.original_code is not None:
            original_code_str = f"""

### Original Code
            
```{self.language}
{self.original_code}
```
            
"""
        else:
            original_code_str = ""

        return f"""## {self.title}{filepath_str}{anchor_str}{notes_str}{original_code_str}
        {"### New Code" if original_code_str else ""}
        
```{self.language}
        {self.new_code}
```
"""


class TextResponsePart(BaseModel):
    """A text chunk that is part of a response by the LLM backend.
    Use this for general discussions, explanations, open questions, brainstorming, warnings, or any information not directly tied to a code change.
    Examples include: overall architectural thoughts, design decisions, potential issues, alternative approaches, or responses to user questions that don't require code.
    """

    text: str = Field(default="", description="The text payload.")

    filepath: Optional[str] = Field(
        default=None,
        description="An optional filepath. Use this to explicitly mark the generated text as refering to one of the files in the context.",
    )

    def show_cli(self) -> str:
        """A (short) CLI representation for the part."""
        filepath_str = f"## {self.filepath}\n" if self.filepath else ""
        return f"{filepath_str}{self.text}"

    def show(self) -> str:
        """Comprehensive longform string representation for the part."""
        filepath_str = f"## {self.filepath}\n\n" if self.filepath else ""
        return f"""{filepath_str}{self.text}
"""


class ShellCommandPart(BaseModel):
    """Represents a shell command that will be run.
    The output of the command will either be used for further processing by the worker, sent to the coder backend, shown to the user, or stored in a file.
    By default, commands are run relative to the project root directory."""

    command: str = Field(
        description="The full shell command that will be run in a virtual terminal."
    )

    reason: str = Field(
        description="A short, informative statement that describes the reason and intent behind this shell command. This will be shown to the user and stored in the logs."
    )

    def show_cli(self) -> str:
        return f"""## Shell command
{self.reason}

```bash
 $> {self.command}
```
        """

    def show(self) -> str:
        return show_model(self)


class FilesLoadPart(BaseModel):
    """Represents a request to load files into context so that their contents become available to the coder LLM."""

    filepaths: List[str] = Field(
        description="One or more  filepaths relative to the project root which will be loaded into context."
    )

    def show_cli(self) -> str:
        return f"""## Load Files into Context
{" -> ".join(self.filepaths)}
"""

    def show(self) -> str:
        return show_model(self)


class FilesUnloadPart(BaseModel):
    """Represents a request to unload one or more files from the current context.
    Unloading files reduces the amount of tokens sent to the coder LLM backend, which will improve performance and may save on token costs.
    Files should only be unloaded if they really are unnecessary to the current task."""

    filepaths: List[str] = Field(
        description="One or more filepaths relative to the project root that should be unloaded from the current context."
    )

    def show_cli(self) -> str:
        return f"""## Unload files from context
{" <- ".join(self.filepaths)}
"""

    def show(self) -> str:
        return show_model(self)


# this one has all the parts
type LLMResponsePart = CodeResponsePart | TextResponsePart | ShellCommandPart | FilesLoadPart | FilesUnloadPart | WorkerCoderRequestPart | EndInteractionPart

# watch out, a kCoderResponsePart is a subset of all available ResponseParts!
type CoderResponsePart = CodeResponsePart | TextResponsePart


class CoderResponse(BaseModel):
    """A response returned from the code generation backend.
    Coder responses are conceptual in nature and contain code and text. THey are not intended to be tool-calls etc.
    """

    contents: List[CoderResponsePart] = Field(
        default_factory=lambda: [],
        description="A list of response parts returned by the backend coder LLM. This may contain code and/or text.",
    )

    def show_cli(self) -> str:
        return "\n".join([part.show_cli() for part in self.contents])

    def show(self) -> str:
        return "\n".join([part.show() for part in self.contents])


class WorkerCoderRequestPart(BaseModel):
    """Represents a request that should be send to the coder LLM backend.
    This allows the worker to communicate with the coder for e.g. clarification, refactoring, adjustment to the user prompt, or anything else that requires the coders capabilities.
    """

    text: str = Field(
        description="The plaintext prompt that will be send verbatim to the coder LLM."
    )

    reason: str = Field(
        description="A short, informative message describing the reason for the request. This message will be displayed to the user and stored in the logs."
    )

    def show_cli(self):
        return f"""## Request from ghostworker to ghostcoder

```
        {show_model(self)}
```

"""

    def show(self) -> str:
        return show_model(self)


class EndInteractionPart(BaseModel):
    """Represents a signal to end the current interaction."""

    reason: str = Field(description="The reason for ending the interaction.")

    def show_cli(self) -> str:
        return show_model(self)

    def show(self) -> str:
        return show_model(self)


# watch out, WorkerResponsePart is a subset of all available kResponsePart types, and it's not the same as a CoderResponsePartresponse
type WorkerResponsePart = ShellCommandPart | FilesLoadPart | FilesUnloadPart | WorkerCoderRequestPart | EndInteractionPart


class WorkerResponse(BaseModel):
    """A response returned from the worker backend.
    Worker responses serve to either enable or improve prompts to the coder backend, or implement responses from the coder backend. They may include tool calls like file reads, shell commands, or http requests, as well as follow-up prompts to the coder backend for clarification.
    """

    contents: List[WorkerResponsePart] = Field(
        description="The various response parts returned by the worker backend."
    )

    def show_cli(self) -> str:
        return "\n\n".join([part.show_cli() for part in self.contents])

    def show(self) -> str:
        return "\n\n".join([part.show() for part in self.contents])


class CoderInteractionHistoryItem(BaseModel):
    """A LLM coder response that was generated during a user interaction."""

    timestamp: str = Field(default_factory=timestamp_now_iso8601)

    context: ContextFiles = Field(
        description="The context that was current at the time the interaction was made."
    )

    backend: str = Field(
        description="The LLM backend that was used at the time the interaction was made. See ghostbox.definitions.LLMBackends."
    )

    model: str = Field(
        description="The particular LLM model that was used by the backend at the time the interaction was made."
    )

    response: CoderResponse = Field(
        description="The full response returned by the LLM. as aprt of this interaction."
    )


    git_commit_hash: Optional[str] = Field(
        default = None,
        description = "The commit that was checked out at the time this interaction took place."
    )
    def show(self) -> str:
        """Returns a human readable string representation of the item."""
        return f"""[{self.timestamp}] {self.context.show_cli()}
  {self.model}@{self.backend} >
{        self.response.show()}
"""


class UserInteractionHistoryItem(BaseModel):
    """Part of an AI/User interaction representing a user prompt."""

    timestamp: str = Field(default_factory=timestamp_now_iso8601)

    git_commit_hash: Optional[str] = Field(
        default=None,
        description="The commit that was at the head position at the time this interaction item was produced. None if no git repository is present or the hash could otherwise not be determined.",
    )

    context: ContextFiles = Field(
        description="The context that was current at the time the interaction was made."
    )

    preamble: str = Field(
        default="",
        description="The plaintext of the context that was added automatically before the user prompt. This is usually long and not present in every interaction",
    )

    prompt: str = Field(
        description="The actual user prompt. This includes only the plain text created directly by the user at the time of the interaction."
    )

    def show(self) -> str:
        return f"""[{self.timestamp}] {self.context.show_cli()}
  user >
        {self.prompt}
"""


type InteractionHistoryItem = UserInteractionHistoryItem | CoderInteractionHistoryItem

class HasTimestamp(Protocol):
    timestamp: str

class HasUniqueID(Protocol):
    unique_id: str
    tag: Optional[str]


class HasGitCommitHash(Protocol):
    git_commit_hash: str


class InteractionHistory(BaseModel):
    """An interaction history represents 0 or more conversational turns between user and an LLM backend.
    This is usually saved to .ghostcode/interaction_history.json, where4 all interaction histories of a project are tracked.
    """

    unique_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Randomly generated unique identified that can be used to retrieve the interaction history.",
    )

    tag: Optional[str] = Field(
        default_factory=lambda: make_mnemonic(3),
        description="An easy to remember, somewhat unique string that can be used to retrieve the conversation history. If any ambiguity is encountered, the unique_id is used as a fallback. Can also be set by users.",
    )

    title: str = Field(
        default = "",
        description = "A short phrase summarizing what the interaction was about. An interaction's title may change over time."
    )
    
    contents: List[InteractionHistoryItem] = Field(
        default_factory=lambda: [],
        description="List of past interactions in this ghostcode project.",
    )

    def empty(self) -> bool:
        """Returns true if the history is empty."""
        return self.contents == []

    def timestamps(self) -> Optional[Tuple[str, str]]:
        """Returns a pair of (earliest interaction, most recent itneraction), or None if no interactions present."""
        if self.empty():
            return None


        # rely on the fact that iso8601 timestamps equate chronological with lexicographical ordering
        timestamps = sorted([item.timestamp for item in self.contents])
        return (timestamps[0], timestamps[-1])
    
    def get_affected_git_commits(self) -> List[str]:
        """Returns a (non-duplicate containing) list of git commits that served as basis for the interaction turns."""
        return list(set([hash
                         for item in self.contents
                         if (hash := item.git_commit_hash) is not None]))
        
    def show(self) -> str:
        """Returns a human readable text representation of the history."""
        return "\n".join([item.show() for item in self.contents])


### Actions ###
# These are put on the action queue and sequentially resolved.
# Actions share some structure with ResponseParts but are also different.
# In general, the distinction between Actions and ResponseParts exists to alleviate the context burden on the LLM backend, both for memory constraints and performance degradation with large contexts. In particular:
# 1. Actions are more fine grained and low level. They contain things that are hard for LLMs (e.g. line numbers/character positions for file edits).
# 2. Actions have clearance and user confirmation logic associated with them. LLM ResponseParts do not (we can't keep an LLM from generating a part).
# 3. Actions can fail.
# 4. A ResponsePart may map to 0 or more actions.


class ActionDoNothing(BaseModel):
    """Action that represents a noop.
    Used mostly as an identity under composition in the 'Action' type."""

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.AUTOMATIC


class ActionHaltExecution(BaseModel):
    """Represents a signal to halt the execution of the action queue. This is put on the queue whenever execution has run into too many errors or has encoutnered a critical problem that can not be overcome."""

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.AUTOMATIC

    reason: str = Field(description="The reason for the halt signal in plain english.")

    announce: bool = Field(
        default=False,
        description="Wether to announce the reason for the halting to the user. During regular execution, this is usually unnecessary.",
    )


class ActionHandleCodeResponsePart(BaseModel):
    """This action represents a code response that was received by the coder LLM and needs to be handled.
    This is a thin wrapper around a CodeResponsePart. It may be handled in a variety of ways, the exact manner of which is usually determined by the worker LLM.
    """

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.AUTOMATIC

    content: CodeResponsePart = Field(
        description="The original code response part that was received from the coder LLM."
    )


class ActionFileCreate(BaseModel):
    """Represents the creation of a new file."""

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.CONFIRM
    filepath: str = Field(description="The path to the file that should be created.")

    content: str = Field(
        description="The contents that will be written to the newly created file."
    )


class ActionFileEdit(BaseModel):
    """Representation of a single replace operation in a file.
    This type is intended to be the end result of a worker file edit and the last step before the actual text is changed on disk.
    """

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.CONFIRM

    filepath: str = Field(description="The file that should be edited.")

    replace_pos_begin: int = Field(
        description="The character position in the file that marks the beginning of the text that should be replaced. Assume python-style indices."
    )

    replace_pos_end: int = Field(
        description="The character position in the file that marks the end of the text that should be replaced. Assume python-style indices."
    )

    insert_text: str = Field(
        description="The text that will be inserted as a replacement for the text between replace_pos_begin and replace_pos_end."
    )


class ActionShellCommand(BaseModel):
    """Action for executing shell commands.
    This is one security nightmare, but I guess we are doing this!
    This type shares most of its structure with the ShellCommandResponsePart and so contains it in full.
    """

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.DANGEROUS

    content: ShellCommandPart = Field(
        description="The various parameters required to invoke subprocess.Popen. Stored in a Part object for convenience and reuse."
    )

    def run(
        self,
        tty: shell.VirtualTerminal,
        clearance: Optional[ClearanceRequirement] = None,
    ) -> None:
        """Executes the shell command asynchronously."""

        # Although this parameter is optional and not really used, it exists here and is kchecked to force people to
        # state it explicitly at the call site
        if clearance is None:
            raise RuntimeError(
                "Bad call of ActionShellCommand.run: You **must** provide the clearance parameter explicitly."
            )

        if clearance.value < self.clearance_required.value:
            raise PermissionError(
                f"Insufficient clearance of {clearance.name} for ActionShellCommand.run. Clearance required: {self.clearance_required.name}"
            )

        logger.info(f"Running shell command: {" ".join(self.content.command)}")
        logger.debug("Shell command dump:\n{self.content.show()}")


# These are a bunch of small classes for a ContextAlteration type
# I wish we had less verbose ways to do sum types but oh well


class ContextAlterationLoadFile(BaseModel):
    filepath: str


class ContextAlterationUnloadFile(BaseModel):
    filepath: str


class ContextAlterationFlagFile(BaseModel):
    # this is a placeholder and not used yet.
    flag: bool


type ContextAlteration = ContextAlterationLoadFile | ContextAlterationUnloadFile | ContextAlterationFlagFile


class ActionAlterContext(BaseModel):
    """Action to either load, unload, or otherwise mark files in the ghostcode context. Files that are not in context will not be sent to the coder LLM backend."""

    clearance_required: ClassVar[ClearanceRequirement] = ClearanceRequirement.INFORM

    context_alteration: ContextAlteration


class HasClearance(Protocol):
    """Interface for anything that requires clearance."""

    clearance_required: ClassVar[ClearanceRequirement]


type Action = ActionHandleCodeResponsePart | ActionFileCreate | ActionFileEdit | ActionDoNothing | ActionHaltExecution | ActionShellCommand | ActionAlterContext


def action_show_short(action: Action) -> str:
    """Gives a short representation of an action.
    By default, this function returns the actions type name with the conventional Action prefix removed. Some actions may give additional information, e.g. ActionFileEdit becomes "FileEdit(types.py)".
    This short representation is used at various points in the CLI interface, such as when asking the user for permission to perform an action.
    """

    canonical_name = action.__class__.__name__
    if canonical_name.startswith("Action"):
        name = canonical_name[len("Action") :]
    else:
        name = canonical_name

    # customize for some actions
    # this is the kind of thing you could endlessly overcomplicate with an ABC interface when a simple match case will do just fine
    match action:
        case ActionFileEdit() as file_edit_action:
            return f"{name}({file_edit_action.filepath})"
        case ActionFileCreate() as file_create_action:
            return f"{name}({file_create_action.filepath})"
        case _:
            return name


def action_show_user_message(
    action: Action, delimiter_color: Optional[Color256] = None
) -> str:
    """Display a user message for the action, which is printed when the action is popped off the action queue."""
    if delimiter_color is None:
        delimiter = " >>= "
    else:
        delimiter = f" {colored(">>=", delimiter_color)} "

    match action:
        case _:
            result = action_show_short(action)
    return result + delimiter


class ActionResultOk(BaseModel):
    """Represents a successfully executed action."""

    # maybe add timing information?
    success_message: Optional[str] = Field(
        default=None,
        description="Optional message that describes the successfully completed action or its result.",
    )


class ActionResultFailure(BaseModel):
    """Represents an action that failed to execute for some reason."""

    original_action: Action = Field(
        description="The original action that failed to execute."
    )

    error_messages: List[str] = Field(
        default_factory=lambda: [],
        description="Any error messages, logs, failure rports, or other technical indicators of failure that are associated with this action.",
    )

    failure_reason: str = Field(
        description="The reason or problem that kept the action from executing either partially or completely, in plain english."
    )


class ActionResultMoreActions(BaseModel):
    """Result that represents that more actions need to be executed, along with action objects representing the required actions. Crucially, actions in the MoreActions result are pushed to the front of the action queue."""

    actions: List[Action] = Field(
        default_factory=lambda: [],
        description="The queue of actions that will need to be executed.",
    )


type ActionResult = ActionResultOk | ActionResultFailure | ActionResultMoreActions


class Project(BaseModel):
    """Basic context for a ghostcode project.
    By convention, all Fields of this type are found in the .ghostcode directory at the project root, with their Field names also being the filename.
    """

    config: ProjectConfig = Field(
        default_factory=ProjectConfig, description="Project wide configuration options."
    )

    worker_llm_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options sent to ghostbox for the worker LLM. This is a json file stored in .ghostcode/worker_llm_config.json.",
    )

    coder_llm_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options sent to ghostbox for the coder LLM. This is a json file stored in .ghostcode/coder_llm_config.json.",
    )

    context_files: ContextFiles = Field(default_factory=ContextFiles)

    directory_file: str = Field(
        default="",
        description="A directory file is an automatically generated markdown file intended to keep LLMs from introducing changes that break the project. It usually contains a project overview and important information about the project, such as module dependencies, architectural limitations, technology choices, and much more. Stored in .ghostcode/directory_file.md.",
    )
    project_metadata: Optional[ProjectMetadata] = Field(
        default_factory=lambda: ProjectMetadata(**DEFAULT_PROJECT_METADATA)
    )

    interactions: List[InteractionHistory] = Field(
        default_factory=lambda: [],
        description="List of interactions that have occurred in the past.",
    )

    # --- File names within the .ghostcode directory ---
    _GHOSTCODE_DIR: ClassVar[str] = ".ghostcode"
    _WORKER_CHARACTER_FOLDER: ClassVar[str] = "worker"
    _WORKER_CONFIG_FILE: ClassVar[str] = "worker/config.json"
    _CODER_CHARACTER_FOLDER: ClassVar[str] = "coder"
    _CODER_CONFIG_FILE: ClassVar[str] = "coder/config.json"
    _CONTEXT_FILES_FILE: ClassVar[str] = "context_files"  # Plaintext list of filepaths
    _DIRECTORY_FILE: ClassVar[str] = "directory_file.md"  # Plaintext markdown
    _PROJECT_METADATA_FILE: ClassVar[str] = "project_metadata.yaml"  # YAML format
    _PROJECT_CONFIG_FILE: ClassVar[str] = "config.yaml"  # YAML format
    _INTERACTION_HISTORY_FILE: ClassVar[str] = "interaction_history.json"
    _CURRENT_INTERACTION_HISTORY_FILE: ClassVar[str] = "current.json"
    _CURRENT_INTERACTION_PLAINTEXT_FILE: ClassVar[str] = "current.txt"
    _LOG_FILE: ClassVar[str] = "log.txt"

    _WORKER_SYSTEM_MSG: ClassVar[str] = (
        "You are GhostWorker, a helpful AI assistant focused on executing specific, local programming tasks. Your primary role is to interact with the file system, run shell commands, and perform other environment-specific operations as instructed by GhostCoder. Be concise, precise, and report results clearly. Do not engage in high-level planning or code generation unless explicitly asked to generate a small snippet for a tool."
    )
    _CODER_SYSTEM_MSG: ClassVar[str] = (
        "You are GhostCoder, a highly intelligent and experienced AI programmer. Your role is to understand complex programming tasks, devise high-level plans, generate code, and review existing code. You will delegate specific environment interactions (like reading/writing files or running commands) to GhostWorker. Focus on architectural decisions, code quality, and problem-solving. When delegating, provide clear and unambiguous instructions to GhostWorker."
    )

    @staticmethod
    def _get_ghostcode_path(root: str) -> str:
        """Helper to get the full path to the .ghostcode directory."""
        return os.path.join(root, Project._GHOSTCODE_DIR)

    @staticmethod
    def find_project_root(start_path: str = ".") -> Optional[str]:
        """
        Traverses up the directory tree from start_path to find the .ghostcode directory.

        Args:
            start_path (str): The directory to start searching from.

        Returns:
            Optional[str]: The absolute path to the project root (parent of .ghostcode), or None if not found.
        """
        current_path = os.path.abspath(start_path)
        while True:
            ghostcode_path = os.path.join(current_path, Project._GHOSTCODE_DIR)
            if os.path.isdir(ghostcode_path):
                logger.debug(
                    f"Found .ghostcode directory at {ghostcode_path}. Project root is {current_path}"
                )
                return current_path

            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:  # Reached filesystem root
                logger.debug(
                    f"No .ghostcode directory found up to filesystem root from {start_path}"
                )
                return None
            current_path = parent_path

    @staticmethod
    def init(root: str) -> None:
        """
        Sets up a .ghostcode directory in the given root directory with all the necessary default files.

        Args:
            root (str): The root directory of the project where .ghostcode should be created.
        """
        ghostcode_dir = Project._get_ghostcode_path(root)
        logger.info(f"Initializing .ghostcode directory at: {ghostcode_dir}")

        try:
            os.makedirs(ghostcode_dir, exist_ok=True)
            logger.info(f"Ensured directory {ghostcode_dir} exists.")

            # Create .ghostcode/worker and .ghostcode/coder directories
            worker_char_dir = os.path.join(ghostcode_dir, "worker")
            coder_char_dir = os.path.join(ghostcode_dir, "coder")
            os.makedirs(worker_char_dir, exist_ok=True)
            os.makedirs(coder_char_dir, exist_ok=True)
            logger.info(f"Created worker character directory at {worker_char_dir}")
            logger.info(f"Created coder character directory at {coder_char_dir}")

            # Create system_msg files within character directories
            worker_system_msg_path = os.path.join(worker_char_dir, "system_msg")
            with open(worker_system_msg_path, "w") as f:
                f.write(Project._WORKER_SYSTEM_MSG)
            logger.info(f"Created worker system_msg at {worker_system_msg_path}")

            coder_system_msg_path = os.path.join(coder_char_dir, "system_msg")
            with open(coder_system_msg_path, "w") as f:
                f.write(Project._CODER_SYSTEM_MSG)
            logger.info(f"Created coder system_msg at {coder_system_msg_path}")

            # 1. Create worker/config.json
            worker_config_path = os.path.join(
                ghostcode_dir, Project._WORKER_CONFIG_FILE
            )
            with open(worker_config_path, "w") as f:
                json.dump(DEFAULT_WORKER_LLM_CONFIG, f, indent=4)
            logger.info(f"Created default worker LLM config at {worker_config_path}")

            # 2. Create coder/config.json
            coder_config_path = os.path.join(ghostcode_dir, Project._CODER_CONFIG_FILE)
            with open(coder_config_path, "w") as f:
                json.dump(DEFAULT_CODER_LLM_CONFIG, f, indent=4)
            logger.info(f"Created default coder LLM config at {coder_config_path}")

            # 3. Create context_files (plaintext, initially empty)
            context_files_path = os.path.join(
                ghostcode_dir, Project._CONTEXT_FILES_FILE
            )
            with open(context_files_path, "w") as f:
                f.write("")  # Empty file
            logger.info(f"Created empty context files list at {context_files_path}")

            # 4. Create directory_file.md (plaintext, initially empty)
            directory_file_path = os.path.join(ghostcode_dir, Project._DIRECTORY_FILE)
            with open(directory_file_path, "w") as f:
                f.write("")  # Empty file
            logger.info(f"Created empty directory file at {directory_file_path}")

            # 5. Create project_metadata.yaml (YAML, with default metadata)
            project_metadata_path = os.path.join(
                ghostcode_dir, Project._PROJECT_METADATA_FILE
            )
            with open(project_metadata_path, "w") as f:
                yaml.dump(
                    DEFAULT_PROJECT_METADATA,
                    f,
                    indent=2,
                    sort_keys=False,
                    default_flow_style=False,
                    Dumper=PydanticEnumDumper,
                )
            logger.info(f"Created default project metadata at {project_metadata_path}")

            # 6. Create config.yaml (YAML, with default ProjectConfig)
            project_config_path = os.path.join(
                ghostcode_dir, Project._PROJECT_CONFIG_FILE
            )
            default_project_config = ProjectConfig()
            with open(project_config_path, "w") as f:
                yaml.dump(
                    default_project_config.model_dump(mode="json"),
                    f,
                    Dumper=PydanticEnumDumper,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False,
                )
            logger.info(f"Created default project config at {project_config_path}")

            # 7. Create interaction_history.json (JSON, initially empty)
            interaction_history_path = os.path.join(
                ghostcode_dir, Project._INTERACTION_HISTORY_FILE
            )
            with open(interaction_history_path, "w") as f:
                json.dump([], f, indent=4)
            logger.info(
                f"Created empty interaction history at {interaction_history_path}"
            )

            logger.info(f"Ghostcode project initialized successfully in {root}.")

        except OSError as e:
            logger.error(
                f"Failed to create .ghostcode directory or files in {root}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during initialization in {root}: {e}",
                exc_info=True,
            )
            raise

    @staticmethod
    def from_root(root: str) -> "Project":
        """
        Looks for a .ghostcode folder in the given root directory, loads all the necessary files,
        constructs the respective types, and then returns a new Project instance.

        Args:
            root (str): The root directory of the project.

        Returns:
            Project: A new Project instance populated with data from the .ghostcode directory.

        Raises:
            FileNotFoundError: If the .ghostcode directory does not exist.
        """
        ghostcode_dir = Project._get_ghostcode_path(root)
        logger.info(f"Loading ghostcode project from: {ghostcode_dir}")

        if not os.path.isdir(ghostcode_dir):
            logger.error(
                f"'.ghostcode' directory not found at {ghostcode_dir}. Please initialize the project first using `ghostcode init '{root}'`."
            )
            raise FileNotFoundError(
                f"'.ghostcode' directory not found at {ghostcode_dir}"
            )

        worker_llm_config = {}
        coder_llm_config = {}
        context_files = ContextFiles()
        directory_file_content = ""
        project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
        project_config = ProjectConfig()  # Initialize with defaults
        interactions = []

        # 1. Load worker_llm_config.json (now from worker/config.json)
        worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
        try:
            with open(worker_config_path, "r") as f:
                worker_llm_config = json.load(f)
            logger.debug(f"Loaded worker LLM config from {worker_config_path}")
        except FileNotFoundError:
            logger.warning(
                f"Worker LLM config file not found at {worker_config_path}. Using default configuration."
            )
            worker_llm_config = DEFAULT_WORKER_LLM_CONFIG
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding JSON from {worker_config_path}: {e}. Using default configuration.",
                exc_info=True,
            )
            worker_llm_config = DEFAULT_WORKER_LLM_CONFIG
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {worker_config_path}: {e}. Using default configuration.",
                exc_info=True,
            )
            worker_llm_config = DEFAULT_WORKER_LLM_CONFIG

        # 2. Load coder_llm_config.json (now from coder/config.json)
        coder_config_path = os.path.join(ghostcode_dir, Project._CODER_CONFIG_FILE)
        try:
            with open(coder_config_path, "r") as f:
                coder_llm_config = json.load(f)
            logger.debug(f"Loaded coder LLM config from {coder_config_path}")
        except FileNotFoundError:
            logger.warning(
                f"Coder LLM config file not found at {coder_config_path}. Using default configuration."
            )
            coder_llm_config = DEFAULT_CODER_LLM_CONFIG
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding JSON from {coder_config_path}: {e}. Using default configuration.",
                exc_info=True,
            )
            coder_llm_config = DEFAULT_CODER_LLM_CONFIG
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {coder_config_path}: {e}. Using default configuration.",
                exc_info=True,
            )
            coder_llm_config = DEFAULT_CODER_LLM_CONFIG

        # 3. Load context_files
        context_files_path = os.path.join(ghostcode_dir, Project._CONTEXT_FILES_FILE)
        try:
            with open(context_files_path, "r") as f:
                content = f.read()
                context_files = ContextFiles.from_plaintext(content)
            logger.debug(f"Loaded context files from {context_files_path}")
        except FileNotFoundError:
            logger.warning(
                f"Context files list not found at {context_files_path}. Using empty list."
            )
            context_files = ContextFiles(data=[])
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {context_files_path}: {e}. Using empty list.",
                exc_info=True,
            )
            context_files = ContextFiles(data=[])

        # 4. Load directory_file.md
        directory_file_path = os.path.join(ghostcode_dir, Project._DIRECTORY_FILE)
        try:
            with open(directory_file_path, "r") as f:
                directory_file_content = f.read()
            logger.debug(f"Loaded directory file from {directory_file_path}")
        except FileNotFoundError:
            logger.warning(
                f"Directory file not found at {directory_file_path}. Using empty content."
            )
            directory_file_content = ""
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {directory_file_path}: {e}. Using empty content.",
                exc_info=True,
            )
            directory_file_content = ""

        # 5. Load project_metadata.yaml
        project_metadata_path = os.path.join(
            ghostcode_dir, Project._PROJECT_METADATA_FILE
        )
        try:
            with open(project_metadata_path, "r") as f:
                metadata_dict = yaml.safe_load(f)
                if metadata_dict:
                    project_metadata = ProjectMetadata(**metadata_dict)
                else:
                    logger.warning(
                        f"Project metadata file {project_metadata_path} is empty. Using default metadata."
                    )
                    project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
            logger.debug(f"Loaded project metadata from {project_metadata_path}")
        except FileNotFoundError:
            logger.warning(
                f"Project metadata file not found at {project_metadata_path}. Using default metadata."
            )
            project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
        except yaml.YAMLError as e:
            logger.error(
                f"Error decoding YAML from {project_metadata_path}: {e}. Using default metadata.",
                exc_info=True,
            )
            project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {project_metadata_path}: {e}. Using default metadata.",
                exc_info=True,
            )
            project_metadata = ProjectMetadata(**DEFAULT_PROJECT_METADATA)

        # 6. Load config.yaml
        project_config_path = os.path.join(ghostcode_dir, Project._PROJECT_CONFIG_FILE)
        try:
            with open(project_config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                if config_dict:
                    # Pydantic will handle validation and conversion for ProjectConfig
                    project_config = ProjectConfig(**config_dict)
                else:
                    logger.warning(
                        f"Project config file {project_config_path} is empty. Using default ProjectConfig."
                    )
                    project_config = ProjectConfig()
            logger.debug(f"Loaded project config from {project_config_path}")
        except FileNotFoundError:
            logger.warning(
                f"Project config file not found at {project_config_path}. Using default ProjectConfig."
            )
            project_config = ProjectConfig()
        except yaml.YAMLError as e:
            logger.error(
                f"Error decoding YAML from {project_config_path}: {e}. Using default ProjectConfig.",
                exc_info=True,
            )
            project_config = ProjectConfig()
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {project_config_path}: {e}. Using default ProjectConfig.",
                exc_info=True,
            )
            project_config = ProjectConfig()

        # 7. Load interaction_history.json
        interaction_history_path = os.path.join(
            ghostcode_dir, Project._INTERACTION_HISTORY_FILE
        )
        try:
            with open(interaction_history_path, "r") as f:
                history_data = json.load(f)
                if history_data:
                    interactions = [
                        InteractionHistory(**history_item)
                        for history_item in history_data
                    ]
                else:
                    logger.warning(
                        f"Interaction history file {interaction_history_path} is empty. Using default InteractionHistory."
                    )
                    interactions = []
            logger.debug(f"Loaded interaction history from {interaction_history_path}")
        except FileNotFoundError:
            logger.warning(
                f"Interaction history file not found at {interaction_history_path}. Using empty history."
            )
            interactions = []
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding JSON from {interaction_history_path}: {e}. Using empty history.",
                exc_info=True,
            )
            interactions = []
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading {interaction_history_path}: {e}. Using empty history.",
                exc_info=True,
            )
            interactions = []

        logger.info(f"Ghostcode project loaded successfully from {root}.")
        return Project(
            config=project_config,  # Pass the loaded config
            worker_llm_config=worker_llm_config,
            coder_llm_config=coder_llm_config,
            context_files=context_files,
            directory_file=directory_file_content,
            project_metadata=project_metadata,
            interactions=interactions,
        )

    def save_to_root(self, root: str) -> None:
        """
        Serializes all the contained types and saves them to the .ghostcode folder
        within the given root directory.

        Args:
            root (str): The root directory of the project.
        """
        ghostcode_dir = Project._get_ghostcode_path(root)
        logger.info(f"Saving ghostcode project to: {ghostcode_dir}")

        if not os.path.isdir(ghostcode_dir):
            logger.warning(
                f"'.ghostcode' directory not found at {ghostcode_dir}. Attempting to create it."
            )
            try:
                os.makedirs(ghostcode_dir)
            except OSError as e:
                logger.error(
                    f"Failed to create .ghostcode directory at {ghostcode_dir}: {e}"
                )
                raise

        # Ensure worker and coder character directories exist before saving their configs
        worker_char_dir = os.path.join(ghostcode_dir, "worker")
        coder_char_dir = os.path.join(ghostcode_dir, "coder")
        os.makedirs(worker_char_dir, exist_ok=True)
        os.makedirs(coder_char_dir, exist_ok=True)

        # 1. Save worker_llm_config.json (now to worker/config.json)
        worker_config_path = os.path.join(ghostcode_dir, Project._WORKER_CONFIG_FILE)
        try:
            with open(worker_config_path, "w") as f:
                json.dump(self.worker_llm_config, f, indent=4)
            logger.debug(f"Saved worker LLM config to {worker_config_path}")
        except Exception as e:
            logger.error(
                f"Failed to save worker LLM config to {worker_config_path}: {e}",
                exc_info=True,
            )
            raise

        # 2. Save coder_llm_config.json (now to coder/config.json)
        coder_config_path = os.path.join(ghostcode_dir, Project._CODER_CONFIG_FILE)
        try:
            with open(coder_config_path, "w") as f:
                json.dump(self.coder_llm_config, f, indent=4)
            logger.debug(f"Saved coder LLM config to {coder_config_path}")
        except Exception as e:
            logger.error(
                f"Failed to save coder LLM config to {coder_config_path}: {e}",
                exc_info=True,
            )
            raise

        # 3. Save context_files
        context_files_path = os.path.join(ghostcode_dir, Project._CONTEXT_FILES_FILE)
        try:
            with open(context_files_path, "w") as f:
                f.write(self.context_files.to_plaintext())
            logger.debug(f"Saved context files to {context_files_path}")
        except Exception as e:
            logger.error(
                f"Failed to save context files to {context_files_path}: {e}",
                exc_info=True,
            )
            raise

        # 4. Save directory_file.md
        directory_file_path = os.path.join(ghostcode_dir, Project._DIRECTORY_FILE)
        try:
            with open(directory_file_path, "w") as f:
                f.write(self.directory_file)
            logger.debug(f"Saved directory file to {directory_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to save directory file to {directory_file_path}: {e}",
                exc_info=True,
            )
            raise

        # 5. Save project_metadata.yaml
        project_metadata_path = os.path.join(
            ghostcode_dir, Project._PROJECT_METADATA_FILE
        )
        if self.project_metadata:
            try:
                with open(project_metadata_path, "w") as f:
                    yaml.dump(
                        self.project_metadata.model_dump(mode="json"),
                        f,
                        Dumper=PydanticEnumDumper,
                        indent=2,
                        sort_keys=False,
                        default_flow_style=False,
                    )
                logger.debug(f"Saved project metadata to {project_metadata_path}")
            except Exception as e:
                logger.error(
                    f"Failed to save project metadata to {project_metadata_path}: {e}",
                    exc_info=True,
                )
                raise
        else:
            logger.warning(f"No project metadata to save for {root}.")

        # 6. Save config.yaml
        project_config_path = os.path.join(ghostcode_dir, Project._PROJECT_CONFIG_FILE)
        try:
            with open(project_config_path, "w") as f:
                yaml.dump(
                    self.config.model_dump(mode="json"),
                    f,
                    Dumper=PydanticEnumDumper,
                    indent=2,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.debug(f"Saved project config to {project_config_path}")
        except Exception as e:
            logger.error(
                f"Failed to save project config to {project_config_path}: {e}",
                exc_info=True,
            )
            raise

        # 7. Save interaction_history.json
        interaction_history_path = os.path.join(
            ghostcode_dir, Project._INTERACTION_HISTORY_FILE
        )
        try:
            with open(interaction_history_path, "w") as f:
                json.dump(
                    [history.model_dump() for history in self.interactions], f, indent=4
                )
            logger.debug(f"Saved interaction history to {interaction_history_path}")
        except Exception as e:
            logger.error(
                f"Failed to save interaction history to {interaction_history_path}: {e}",
                exc_info=True,
            )
            raise

        logger.info(f"Ghostcode project saved successfully to {root}.")


class CosmeticProgramState(Enum):
    """A vague indicator of program state. This is used to e.g. color certain outputs in the interface. Do not use this to query program state programmatically."""

    IDLE = 0
    WORKING = 10
    PROBLEM = 20
    CRITICAL_PROBLEM = 30

    def to_color(self) -> Color256:
        match self:
            case CosmeticProgramState.IDLE:
                return Color256.GREEN
            case CosmeticProgramState.WORKING:
                return Color256.ORANGE
            case CosmeticProgramState.PROBLEM:
                return Color256.DARK_MAGENTA
            case CosmeticProgramState.CRITICAL_PROBLEM:
                return Color256.RED
            case _:
                return Color256.GRAY_MEDIUM


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

    # used to color the interface and cannot be relied on except for cosmetics
    cosmetic_state: CosmeticProgramState = field(
        default_factory=lambda: CosmeticProgramState.IDLE
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
        self.action_queue.append(action)

    def push_front_action(self, action: Action) -> None:
        """Pushes an action to the front of the queue. An action at the front will be executed before the remaining ones."""
        logger.info(
            f"Pushing action {action_show_short(action)} to the front of the action queue."
        )
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

        self.print(f"{agent_name} wants to perform {action_show_short(action)}.")
        abridge = 80  # type: Optional[int]
        try:
            while True:
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
                    case _:
                        # cover "?", empty input, and anything else, as showing more nformation is generally a safe option.
                        print(
                            show_model(
                                action,
                                heading=action.__class__.__name__,
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
            print("")
        print(text, end=end, flush=flush)
        self.last_print_tag = tag

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

    def show_log(self, tail: Optional[int] = None) -> Optional[str]:
        """Returns the current event log if available.
        If tail is given, show only the tail latest number of log lines.
        """
        if self.project is None:
            logger.warn("Null project while trying to read logs.")
            return None
        try:
            with open(self.get_data(self.project._LOG_FILE), "r") as f:
                log = f.read()

            if tail is None:
                return log

            return "\n".join([line for line in log.splitlines()][(-1) * tail :])

        except Exception as e:
            logger.warning(f"Failed to read log file. Reason: {e}")
            return None
