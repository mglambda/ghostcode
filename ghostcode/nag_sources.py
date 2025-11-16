# ghostcode.nag_sources
from typing import *
from abc import ABC, abstractmethod, abstractproperty
import os
import json
from pydantic import BaseModel, Field, TypeAdapter
from ghostbox import Ghostbox

class NagCheckResult(BaseModel):
    """Returned by a NagSource check method to both retrieve the source content and an indicator whether it is generally ok or not.""" 
    source_content: str = Field(
        description = "A string representing the original source content that may be ok or not. This can be e.g. a file, a log excerpt, output of a shell command, or an HTTP query."
    )
    
    has_problems: bool = Field(
        description = "If true, the user should probably be nagged about the result."
    )


    
def ok(source_content: Optional[str] = None) -> NagCheckResult:
    """Return a default NagCheckResult that signals everything is ok."""
    return NagCheckResult(source_content = source_content if source_content is not None else "", has_problems = False)

def problem(source_content: Optional[str] = None) -> NagCheckResult:
    """Returns a NagCheckResult that has a problem and may need to be nagged about."""
    return NagCheckResult(source_content=source_content if source_content is not None else "", has_problems=True)
    
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
    def check(self, box: Ghostbox) -> NagCheckResult:
        """Returns the source content and whether the source is generally ok or not, potentially using a provided LLM request client.
        This does not deliver an accurate report, it merely classifies the source as being ok or not, allowing for further processing."""
        pass



class NagSourceFile(NagSourceBase):
    """Represents a file that should be periodically read to be monitored and potentially nagged about."""
    type: Literal["NagSourceFile"] = "NagSourceFile"
    filepath: str    
    nag_interval_seconds: int = Field(
        default = 60,
        description = "Number of seconds between file reads. 1 minute is the default because we don't expect files to change super frequently."
    )


    def identity(self) -> str:
        return self.filepath

    def check(self, box: Ghostbox) -> NagCheckResult:
        # stub
        return ok()


class NagSourceHTTPRequest(NagSourceBase):
    type: Literal["NagSourceHTTPRequest"] = "NagSourceHTTPRequest"
    url: str = Field(
        description = "The URL to send the http request to."
    )

    def identity(self) -> str:
        return self.url

    def check(self, box: Ghostbox) -> NagCheckResult:
        # stub
        return ok()

class NagSourceSubprocess(NagSourceBase):
    """Represents an executable process that is invoked and nagged about if there are problems."""
    type: Literal["NagSourceExecutable"] = "NagSourceExecutable"
    executable_filepath: str

    def identity(self) -> str:
        return self.executable_filepath

    def check(self, box: Ghostbox) -> NagCheckResult:
        # stub
        return ok()


NagSource = Annotated[
    NagSourceFile | NagSourceHTTPRequest | NagSourceSubprocess,
    Field(discriminator="type")
]

NagSourceAdapter: TypeAdapter[NagSource] = TypeAdapter(NagSource)
