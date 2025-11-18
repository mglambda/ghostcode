# ghostcode.nag_sources
from typing import *
from abc import ABC, abstractmethod, abstractproperty
import os
import subprocess
import hashlib
import requests
from enum import StrEnum
import json
from pydantic import BaseModel, Field, TypeAdapter, model_validator
if TYPE_CHECKING:
    from .program import Program
from . import emacs
import logging

logger = logging.getLogger("ghostcode.nag_sources")

class NagCheckResult(BaseModel):
    """Returned by a NagSource check method to both retrieve the source content and an indicator whether it is generally ok or not.""" 
    source_content: str = Field(
        description = "A string representing the original source content that may be ok or not. This can be e.g. a file, a log excerpt, output of a shell command, or an HTTP query."
    )
    
    has_problem: bool = Field(
        description = "If true, the user should probably be nagged about the result."
    )

    error_while_checking: str = Field(
        default = "",
        description = "A string describing an error that occured during the checking process itself. This is distinguished from actual errors or problems with the source, which should not be described here. This string may contain e.g. File not found for a file source, or network connection error for an HTTP source."
    )

    hash: Optional[str] = Field(
        default = None,
        description = "A hash of the content. Providing this let's users of a check result quickly see if content has changed when a problem persists across multiple checks."
    )

    @model_validator(mode='before')
    @classmethod
    def _compute_hash_if_missing(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('hash') is None and values.get('source_content') is not None:
            import hashlib
            content_bytes = values['source_content'].encode('utf-8')
            values['hash'] = hashlib.sha256(content_bytes).hexdigest()
        return values
        
def ok(source_content: Optional[str] = None, hash: Optional[str] = None) -> NagCheckResult:
    """Return a default NagCheckResult that signals everything is ok."""
    return NagCheckResult(source_content = source_content if source_content is not None else "", has_problem = False, hash = hash)

def problem(source_content: Optional[str] = None, error_while_checking: str = "", hash: Optional[str] = None) -> NagCheckResult:
    """Returns a NagCheckResult that has a problem and may need to be nagged about."""
    return NagCheckResult(source_content=source_content if source_content is not None else "", has_problem=True, error_while_checking=error_while_checking, hash=hash)


class EmacsBufferProblemClassification(BaseModel):
    has_problem: bool = Field(description="True if the Emacs buffer content or metadata indicates a problem (e.g., syntax errors, warnings, uncommitted changes, or specific keywords indicating an issue), False otherwise.")
    reason: str = Field(description="A brief reason for the classification.")



class NagSourceBase(BaseModel):
    """A data source for the nag command.
    A source is something that can be monitored, allowing the LLM to potentially nag about something that's wrong."""

    type: str = Field(
        description = "Discriminator for union type."
    )
    
    display_name: str = Field(
        description = "Name of the source. This is what will be displayed to the user in the clie when refering to the source."
    )
    
    nag_interval_seconds: int = Field(
        description = "The interval between checks of the source."
    )


    @abstractmethod
    def identity(self) -> str:
        """Returns some kind of identifier, the nature of which depends on the particular source.
        This is so we can compare sources. For instance, this method may reutrn a filename for a file, or a URL for a webpage, or a PID for a unix shell process etc."""
        pass
    
    @abstractmethod
    def check(self, prog: 'Program', previous_hash: Optional[str] = None) -> NagCheckResult:
        """Returns the source content and whether the source is generally ok or not, potentially using a provided LLM request client.
        This does not deliver an accurate report, it merely classifies the source as being ok or not, allowing for further processing."""
        pass



class NagSourceFile(NagSourceBase):
    """Represents a file that should be periodically read to be monitored and potentially nagged about."""
    type: Literal["NagSourceFile"] = "NagSourceFile"
    filepath: str    
    nag_interval_seconds: int = Field(
        default = 3,
        description = "Number of seconds between file reads. 1 minute is the default because we don't expect files to change super frequently."
    )
    last_modified_timestamp: float = Field(
        default = 0.0,
        description = "Internal timestamp of the last time the file was modified, used to avoid re-reading unchanged files."
    )

    def identity(self) -> str:
        return self.filepath

    def check(self, prog: 'Program', previous_hash: Optional[str] = None) -> NagCheckResult:
        """Reads the file content if it has been updated since the last time it was checked, and returns an LLMs classification."""
        # the filepaths we use here are different from context files and are **not** relative to the root
        abs_filepath = self.filepath

        logger.debug(f"Checking NagSourceFile: {self.filepath}")
        if not os.path.exists(abs_filepath):
            logger.debug(f"File '{self.filepath}' not found.")
            return problem(source_content="", error_while_checking=f"File '{self.filepath}' not found.")
        
        try:
            current_mtime = os.path.getmtime(abs_filepath)
            logger.debug(f"Current mtime for '{self.filepath}': {current_mtime}")
        except OSError as e:
            logger.warning(f"Could not get modification time for '{self.filepath}': {e}")
            return problem(source_content="", error_while_checking=f"Could not get modification time for '{self.filepath}': {e}")

        if self.last_modified_timestamp != 0.0 and current_mtime == self.last_modified_timestamp:
            logger.debug(f"File '{self.filepath}' unchanged since last check.")
            # File hasn't changed since last check, no need to re-read or re-classify
            return ok(source_content=None, hash=previous_hash)

        # File has changed or it's the first check, proceed to read and classify
        self.last_modified_timestamp = current_mtime
        logger.debug(f"File '{self.filepath}' modified or first check. Reading content.")
        
        try:
            with open(abs_filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            logger.debug(f"Successfully read content from '{self.filepath}'.")
        except IOError as e:
            logger.warning(f"Could not read file '{self.filepath}': {e}")
            return problem(source_content="", error_while_checking=f"Could not read file '{self.filepath}': {e}")
        
        # Compute hash for current content
        current_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if previous_hash is not None and current_hash == previous_hash:
            logger.debug(f"NagSourceFile: Content hash for '{self.filepath}' matches previous hash. Returning ok.")
            return ok(source_content=content, hash=previous_hash)

        class FileProblemClassification(BaseModel):
            has_problem: bool

        try:
            result = prog.worker_box.new(
                FileProblemClassification,
                f"These are the contents of file `{self.filepath}.\nPlease inspect the file contents and identify if there is a potential problem or not.\n\n```\n{content}\n```"
            )
        except Exception as e:
            return problem(source_content=content, error_while_checking=f"An exception was encountered while trying to classify the file: {e}.")
        return NagCheckResult(has_problem=result.has_problem, source_content=content, hash=current_hash)
    
class NagSourceHTTPRequest(NagSourceBase):
    type: Literal["NagSourceHTTPRequest"] = "NagSourceHTTPRequest"
    url: str = Field(
        description = "The URL to send the http request to."
    )

    nag_interval_seconds: int = 60
    
    def identity(self) -> str:
        return self.url

    def check(self, prog: 'Program', previous_hash: Optional[str] = None) -> NagCheckResult:
        logger.debug(f"Checking NagSourceHTTPRequest: {self.url}")
        try:
            logger.debug(f"Sending GET request to {self.url}")
            response = requests.get(self.url, timeout=10) # 10 second timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            content = response.text
            logger.debug(f"Received HTTP response (status: {response.status_code}, length: {len(content)}).")

            # Compute hash for current content
            current_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if previous_hash is not None and current_hash == previous_hash:
                logger.debug(f"NagSourceHTTPRequest: Content hash for '{self.url}' matches previous hash. Returning ok.")
                return ok(source_content=content, hash=previous_hash)

            class HTTPResponseClassification(BaseModel):
                has_problem: bool = Field(description="True if the HTTP response content indicates a problem (e.g., error messages, unexpected status codes, or specific content indicating a failure), False otherwise.")
                reason: str = Field(description="A brief reason for the classification.")

            try:
                classification_result = prog.worker_box.new(
                    HTTPResponseClassification,
                    f"The following HTTP GET request was made to URL: `{self.url}`\n\nIts response content is:\n```\n{content}\n```\nPlease inspect the content and determine if it indicates any problems (e.g., error messages, unexpected status codes, or specific content indicating a failure). Respond with `has_problem: true` if there's an issue, `false` otherwise, and a brief `reason`."
                )
                logger.debug(f"LLM classified HTTP response: has_problem={classification_result.has_problem}, reason='{classification_result.reason}'")
                return NagCheckResult(
                    source_content=content,
                    has_problem=classification_result.has_problem,
                    error_while_checking="", # No error during checking, LLM classified output
                    hash=current_hash
                )
            except Exception as e:
                # Error during LLM classification
                logger.error(f"LLM classification failed for HTTP response from {self.url}: {e}")
                return problem(
                    source_content=content,
                    error_while_checking=f"HTTP request succeeded, but LLM classification failed: {e}",
                    hash=current_hash
                )

        except requests.exceptions.Timeout:
            logger.warning(f"HTTP request to '{self.url}' timed out.")
            return problem(source_content="", error_while_checking=f"HTTP request to '{self.url}' timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Failed to connect to '{self.url}': {e}")
            return problem(source_content="", error_while_checking=f"Failed to connect to '{self.url}': {e}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error for '{self.url}': {e}")
            return problem(source_content="", error_while_checking=f"HTTP error for '{self.url}': {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"An unexpected request error occurred for '{self.url}': {e}")
            return problem(source_content="", error_while_checking=f"An unexpected request error occurred for '{self.url}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during HTTP check for '{self.url}': {e}")
            return problem(source_content="", error_while_checking=f"An unexpected error occurred during HTTP check for '{self.url}': {e}")

class NagSourceSubprocess(NagSourceBase):
    type: Literal["NagSourceSubprocess"] = "NagSourceSubprocess"
    command: str
    nag_interval_seconds: int = 30
    def identity(self) -> str:
        return self.command

    def check(self, prog: 'Program', previous_hash: Optional[str] = None) -> NagCheckResult:
        logger.debug(f"Checking NagSourceSubprocess: {self.command}")
        try:
            # Execute the command
            logger.debug(f"Executing command: {self.command}")
            process = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                check=False, # We handle non-zero exit codes ourselves
                cwd=prog.project_root if prog.project_root else None # Run in project root
            )

            combined_output = process.stdout + process.stderr
            logger.debug(f"Command '{self.command}' finished with exit code {process.returncode}. Output length: {len(combined_output)}")
            
            # Compute hash for current content
            current_hash = hashlib.sha256(combined_output.encode('utf-8')).hexdigest()
            if previous_hash is not None and current_hash == previous_hash:
                logger.debug(f"NagSourceSubprocess: Output hash for '{self.command}' matches previous hash. Returning ok.")
                return ok(source_content=combined_output, hash=previous_hash)

            if process.returncode != 0:
                logger.debug(f"Command returned non-zero exit code: {process.returncode}. Classifying as problem.")
                # Command failed (non-zero exit code)
                # compilers, type checkers and test suites universally use this to signify a probelm
                # so it just made our job very easy - and also this is therefore not considered an error_while_checking
                return problem(
                    source_content=combined_output,
                    hash=current_hash
                )
            else:
                # Command succeeded (zero exit code), but check its output for problems
                class CommandOutputClassification(BaseModel):
                    has_problem: bool = Field(description="True if the command output indicates a problem (e.g., warnings, errors, unexpected behavior), False otherwise.")
                    reason: str = Field(description="A brief reason for the classification.")

                try:
                    classification_result = prog.worker_box.new(
                        CommandOutputClassification,
                        f"The following command executed successfully (exit code 0):\n`{self.command}`\n\nIts output is:\n```\n{combined_output}\n```\nPlease inspect the output and determine if it indicates any problems (e.g., warnings, errors, unexpected behavior). Respond with `has_problem: true` if there's an issue, `false` otherwise, and a brief `reason`."
                    )
                    logger.debug(f"LLM classified command output: has_problem={classification_result.has_problem}, reason='{classification_result.reason}'")
                    return NagCheckResult(
                        source_content=combined_output,
                        has_problem=classification_result.has_problem,
                        error_while_checking="", # No error during checking, LLM classified output
                        hash=current_hash
                    )
                except Exception as e:
                    # Error during LLM classification
                    logger.error(f"LLM classification failed for command '{self.command}': {e}")
                    return problem(
                        source_content=combined_output,
                        error_while_checking=f"Command succeeded, but LLM classification failed: {e}",
                        hash=current_hash
                    )

        except FileNotFoundError:
            logger.warning(f"Command '{self.command.split()[0]}' not found.")
            return problem(
                source_content="",
                error_while_checking=f"Command '{self.command.split()[0]}' not found. Is it in PATH?"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred while running command '{self.command}': {e}")
            return problem(
                source_content="",
                error_while_checking=f"An unexpected error occurred while running command '{self.command}': {e}"
            )

class NagSourceEmacsBuffer(NagSourceBase):
    """Represents an emacs buffer used as a source to check for problems.
    Requires emacs to be run in server-mode."""
    type: Literal["NagSourceEmacsBuffer"] = "NagSourceEmacsBuffer"
    buffer_name: str = Field(
        description="The name of the Emacs buffer to monitor."
    )
    nag_interval_seconds: int = 5 # Default interval, can be adjusted

    def identity(self) -> str:
        return f"emacs-buffer:{self.buffer_name}"

    def check(self, prog: 'Program', previous_hash: Optional[str] = None) -> NagCheckResult:
        logger.debug(f"Checking NagSourceEmacsBuffer: {self.buffer_name}")
        try:
            buffer_info = emacs.get_single_buffer(self.buffer_name)

            if buffer_info is None:
                return problem(
                    source_content="",
                    error_while_checking=f"Emacs buffer '{self.buffer_name}' not found or Emacs server not running."
                )

            buffer_content = buffer_info.content
            # Compute hash for current content
            current_hash = hashlib.sha256(buffer_content.encode('utf-8')).hexdigest()
            if previous_hash is not None and current_hash == previous_hash:
                logger.debug(f"NagSourceEmacsBuffer: Content hash for '{self.buffer_name}' matches previous hash. Returning ok.")
                return ok(source_content=buffer_content, hash=previous_hash)

            # We can still provide metadata as additional context if helpful, but the primary focus is content.
            buffer_metadata_json = buffer_info.model_dump_json(indent=2)

            try:
                classification_result = prog.worker_box.new(
                    EmacsBufferProblemClassification,
                    f"The following is the content of Emacs buffer '{self.buffer_name}':\n```\n{buffer_content}\n```\n\nAdditionally, here is its metadata:\n```json\n{buffer_metadata_json}\n```\n\nPlease inspect the buffer content and metadata and determine if it indicates any problems (e.g., syntax errors, warnings, uncommitted changes, or specific keywords indicating an issue). Respond with `has_problem: true` if there's an issue, `false` otherwise, and a brief `reason`."
                )
                logger.debug(f"LLM classified Emacs buffer: has_problem={classification_result.has_problem}, reason='{classification_result.reason}'")
                return NagCheckResult(
                    source_content=buffer_content,
                    has_problem=classification_result.has_problem,
                    error_while_checking="",
                    hash=current_hash
                )
            except Exception as e:
                logger.error(f"LLM classification failed for Emacs buffer '{self.buffer_name}': {e}")
                return problem(
                    source_content=buffer_content,
                    error_while_checking=f"Emacs buffer content retrieved, but LLM classification failed: {e}",
                    hash=current_hash
                )

        except Exception as e:
            logger.error(f"An unexpected error occurred during Emacs buffer check for '{self.buffer_name}': {e}")
            return problem(
                source_content="",
                error_while_checking=f"An unexpected error occurred during Emacs buffer check for '{self.buffer_name}': {e}"
            )

class NagSourceEmacsActiveBuffer(NagSourceBase):
    """Represents the region around point of the currently active emacs buffer.
    This can be checked as a source to identify obvious spelling or code errors.
    Requires emacs to be run in server-mode."""
    
    type: Literal["NagSourceEmacsActiveBuffer"] = "NagSourceEmacsActiveBuffer"
    # this should always be the same and is a magic value
    display_name: str = Field(
        default = "**Active emacs buffer region**"
    )
    nag_interval_seconds: int = 5 # Default interval, can be adjusted

    # New attribute for region size
    region_size: int = Field(
        default = 40, # Default to 40 lines (20 before, 20 after)
        description = "The number of lines to include around point when inspecting the currently active buffer. A value of e.g. 40 will inspect 20 lines above and 20 lines below point. Set to -1 to include the entire buffer (not recommended)."
    )

    def identity(self) -> str:
        # fixed value for this type
        return "__GHOSTCODE_EMACS_ACTIVE_BUFFER_REGION__"

    def check(self, prog: 'Program', previous_hash: Optional[str] = None) -> NagCheckResult:
        logger.debug(f"Checking NagSourceEmacsActiveBuffer with region_size: {self.region_size}")
        
        if not prog.user_config.emacs_integration:
            return problem(
                source_content="",
                error_while_checking="Emacs integration is disabled in user config."
            )

        before_lines = None
        after_lines = None
        if self.region_size != -1:
            half_size = self.region_size // 2
            before_lines = half_size
            after_lines = half_size

        try:
            active_buffer_info = emacs.get_active_buffer(
                before_lines=before_lines,
                after_lines=after_lines
            )

            if active_buffer_info is None:
                return problem(
                    source_content="",
                    error_while_checking="Could not retrieve active Emacs buffer content. Is Emacs server running?"
                )

            buffer_content = active_buffer_info.content
            # Compute hash for current content
            current_hash = hashlib.sha256(buffer_content.encode('utf-8')).hexdigest()
            if previous_hash is not None and current_hash == previous_hash:
                logger.debug(f"NagSourceEmacsActiveBuffer: Content hash for active buffer matches previous hash. Returning ok.")
                return ok(source_content=buffer_content, hash=previous_hash)

            # Provide metadata as additional context if helpful
            buffer_metadata_json = active_buffer_info.model_dump_json(indent=2)

            # Use the same classification model as NagSourceEmacsBuffer
            try:
                classification_result = prog.worker_box.new(
                    EmacsBufferProblemClassification, # Reusing this Pydantic model
                    f"The following is the content of the active Emacs buffer region (buffer: '{active_buffer_info.buffer_name}', file: '{active_buffer_info.file_name}'):\n```\n{buffer_content}\n```\n\nAdditionally, here is its metadata:\n```json\n{buffer_metadata_json}\n```\n\n"
                    #+ "Please inspect the buffer content and metadata and determine if it indicates any problems (e.g., syntax errors, warnings, uncommitted changes, or specific keywords indicating an issue). Respond with `has_problem: true` if there's an issue, `false` otherwise, and a brief `reason`."
                    + "Please consider the buffer's major mode and content. If it appears to be programming language code, indicate a problem if there are obvious code mistakes, syntax errors, or anything else that looks erroneous or confused. This is a heuristic, Please only classify this as a problem if you are certain there is a problem. Better be safe than sorry. If you detect an issue, respond with has_problem: true, and false otherwise. If you state a reason, it should be extremely brief. If you cannot identify the content as programming relatex, simply classify it as problem-free."
                )
                prog.print(f"LLM classified active Emacs buffer region: has_problem={classification_result.has_problem}, reason='{classification_result.reason}'")
                return NagCheckResult(
                    source_content=buffer_content,
                    has_problem=classification_result.has_problem,
                    error_while_checking="",
                    hash=current_hash
                )
            except Exception as e:
                logger.error(f"LLM classification failed for active Emacs buffer region: {e}")
                return problem(
                    source_content=buffer_content,
                    error_while_checking=f"Active Emacs buffer content retrieved, but LLM classification failed: {e}",
                    hash=current_hash
                )

        except Exception as e:
            logger.error(f"An unexpected error occurred during active Emacs buffer check: {e}")
            return problem(
                source_content="",
                error_while_checking=f"An unexpected error occurred during active Emacs buffer check: {e}"
            )

        
NagSource = Annotated[
    NagSourceFile | NagSourceHTTPRequest | NagSourceSubprocess | NagSourceEmacsBuffer | NagSourceEmacsActiveBuffer,
    Field(discriminator="type")
]

NagSourceAdapter: TypeAdapter[NagSource] = TypeAdapter(NagSource)

