# ghostcode.nag_sources
from typing import *

from abc import ABC, abstractmethod, abstractproperty
import os
from enum import StrEnum
import json
from pydantic import BaseModel, Field, TypeAdapter
if TYPE_CHECKING:
    from .program import Program

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
def ok(source_content: Optional[str] = None) -> NagCheckResult:
    """Return a default NagCheckResult that signals everything is ok."""
    return NagCheckResult(source_content = source_content if source_content is not None else "", has_problem = False)

def problem(source_content: Optional[str] = None) -> NagCheckResult:
    """Returns a NagCheckResult that has a problem and may need to be nagged about."""
    return NagCheckResult(source_content=source_content if source_content is not None else "", has_problem=True)



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
    def check(self, prog: 'Program') -> NagCheckResult:
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
    _last_modified_timestamp: float = Field(
        default = 0.0,
        description = "Internal timestamp of the last time the file was modified, used to avoid re-reading unchanged files."
    )

    def identity(self) -> str:
        return self.filepath

    def check(self, prog: 'Program') -> NagCheckResult:
        """Reads the file content if it has been updated since the last time it was checked, and returns an LLMs classification."""
        # the filepaths we use here are different from context files and are **not** relative to the root
        abs_filepath = self.filepath

        if not os.path.exists(abs_filepath):
            return problem(source_content=f"File '{self.filepath}' not found.")
        
        try:
            current_mtime = os.path.getmtime(abs_filepath)
        except OSError as e:
            return problem(source_content=f"Could not get modification time for '{self.filepath}': {e}")

        if self._last_modified_timestamp != 0.0 and current_mtime == self._last_modified_timestamp:
            # File hasn't changed since last check, no need to re-read or re-classify
            return ok(source_content=f"File '{self.filepath}' unchanged.")

        # File has changed or it's the first check, proceed to read and classify
        self._last_modified_timestamp = current_mtime
        
        try:
            with open(abs_filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except IOError as e:
            return problem(source_content=f"Could not read file '{self.filepath}': {e}")
        
        class FileProblemClassification(BaseModel):
            has_problem: bool

        try:
            result = prog.worker_box.new(
                FileProblemClassification,
                f"These are the contents of file `{self.filepath}.\nPlease inspect the file contents and identify if there is a potential problem or not.\n\n```\n{content}\n```"
            )
        except Exception as e:
            return problem(source_content=content + f"\n\n---\nAdditionally, an exception was encountered while trying to classify the file: {e}.")
        return NagCheckResult(has_problem=result.has_problem, source_content=content)
    
class NagSourceHTTPRequest(NagSourceBase):
    type: Literal["NagSourceHTTPRequest"] = "NagSourceHTTPRequest"
    url: str = Field(
        description = "The URL to send the http request to."
    )

    nag_interval_seconds: int = 60
    
    def identity(self) -> str:
        return self.url

    def check(self, prog: 'Program') -> NagCheckResult:
        # stub
        return ok()

class NagSourceSubprocess(NagSourceBase):
    """Represents an executable process that is invoked and nagged about if there are problems."""
    type: Literal["NagSourceExecutable"] = "NagSourceExecutable"
    command: str
    nag_interval_seconds: int = 30
    def identity(self) -> str:
        return self.command

    def check(self, prog: 'Program') -> NagCheckResult:
        # stub
        return ok()


NagSource = Annotated[
    NagSourceFile | NagSourceHTTPRequest | NagSourceSubprocess,
    Field(discriminator="type")
]

NagSourceAdapter: TypeAdapter[NagSource] = TypeAdapter(NagSource)
